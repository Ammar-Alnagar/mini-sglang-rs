use crate::layers::linear::Linear;
use crate::layers::norm::RMSNorm;
use crate::models::config::ModelConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone)]
struct RotaryEmbeddingCache {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbeddingCache {
    fn new(dim: usize, max_seq_len: usize, rope_theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_seq_len, 1), device)?;

        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let rotated = rotate_half(x)?;
        x.broadcast_mul(&cos)?
            .broadcast_add(&rotated.broadcast_mul(&sin)?)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)
}

struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbeddingCache,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen2Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj_weight = vb.get((num_heads * head_dim, hidden_size), "q_proj.weight")?;
        let q_proj_bias = vb.get(num_heads * head_dim, "q_proj.bias").ok();
        let q_proj = Linear::new(q_proj_weight, q_proj_bias);

        let k_proj_weight = vb.get((num_kv_heads * head_dim, hidden_size), "k_proj.weight")?;
        let k_proj_bias = vb.get(num_kv_heads * head_dim, "k_proj.bias").ok();
        let k_proj = Linear::new(k_proj_weight, k_proj_bias);

        let v_proj_weight = vb.get((num_kv_heads * head_dim, hidden_size), "v_proj.weight")?;
        let v_proj_bias = vb.get(num_kv_heads * head_dim, "v_proj.bias").ok();
        let v_proj = Linear::new(v_proj_weight, v_proj_bias);

        let o_proj_weight = vb.get((hidden_size, num_heads * head_dim), "o_proj.weight")?;
        let o_proj = Linear::new(o_proj_weight, None);

        let rotary = RotaryEmbeddingCache::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let pos = match &self.kv_cache {
            Some((prev_k, _)) => prev_k.dim(2)?,
            None => 0,
        };

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.rotary.apply_rotary_emb(&q, pos)?;
        let k = self.rotary.apply_rotary_emb(&k, pos)?;

        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let total_kv_len = k.dim(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let att = if seq_len > 1 {
            let mask = causal_mask(seq_len, total_kv_len, att.dtype(), att.device())?;
            att.broadcast_add(&mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((b_sz, n_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

fn causal_mask(seq_len: usize, total_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * total_len];
    let past_len = total_len - seq_len;
    for i in 0..seq_len {
        for j in 0..total_len {
            if j > past_len + i {
                mask[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?.to_dtype(dtype)
}

struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        let gate_weight = vb.get((intermediate_size, hidden_size), "gate_proj.weight")?;
        let gate_proj = Linear::new(gate_weight, None);

        let up_weight = vb.get((intermediate_size, hidden_size), "up_proj.weight")?;
        let up_proj = Linear::new(up_weight, None);

        let down_weight = vb.get((hidden_size, intermediate_size), "down_proj.weight")?;
        let down_proj = Linear::new(down_weight, None);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let intermediate = gate.mul(&up)?;
        self.down_proj.forward(&intermediate)
    }
}

struct Qwen2DecoderLayer {
    self_attn: Qwen2Attention,
    mlp: Qwen2MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen2DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm_w = vb.get(cfg.hidden_size, "input_layernorm.weight")?;
        let input_layernorm = RMSNorm::new(input_layernorm_w, cfg.rms_norm_eps);

        let post_attention_layernorm_w =
            vb.get(cfg.hidden_size, "post_attention_layernorm.weight")?;
        let post_attention_layernorm = RMSNorm::new(post_attention_layernorm_w, cfg.rms_norm_eps);

        let self_attn = Qwen2Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Qwen2MLP::new(cfg, vb.pp("mlp"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let hidden_states = self.input_layernorm.forward(x)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let post_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let post_states = self.mlp.forward(&post_states)?;
        residual + post_states
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

pub struct Qwen2Model {
    pub config: ModelConfig,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen2DecoderLayer>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl Qwen2Model {
    pub fn new(config: ModelConfig) -> Self {
        let device = Device::Cpu;
        let embed = candle_nn::Embedding::new(
            Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, &device).unwrap(),
            config.hidden_size,
        );
        let norm = RMSNorm::new(
            Tensor::ones(config.hidden_size, DType::F32, &device).unwrap(),
            config.rms_norm_eps,
        );
        let lm_head = Linear::new(
            Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, &device).unwrap(),
            None,
        );
        Self {
            config,
            embed_tokens: embed,
            layers: Vec::new(),
            norm,
            lm_head,
        }
    }

    pub fn load(config: ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_weight = vb_model.get(
            (config.vocab_size, config.hidden_size),
            "embed_tokens.weight",
        )?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight.clone(), config.hidden_size);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen2DecoderLayer::new(&config, vb_layers.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm_weight = vb_model.get(config.hidden_size, "norm.weight")?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);

        let lm_head_weight = if config.tie_word_embeddings.unwrap_or(true) {
            match vb.get((config.vocab_size, config.hidden_size), "lm_head.weight") {
                Ok(w) => w,
                Err(_) => embed_weight,
            }
        } else {
            vb.get((config.vocab_size, config.hidden_size), "lm_head.weight")?
        };
        let lm_head = Linear::new(lm_head_weight, None);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states)?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states)
    }

    pub fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}
