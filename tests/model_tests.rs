use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use minisgl::engine::config::EngineConfig;
use minisgl::llm::engine::LLMEngine;
use minisgl::models::config::ModelConfig;
use minisgl::models::qwen2::Qwen2Model;
use minisgl::models::qwen3::Qwen3Model;
use minisgl::tokenizer::worker::TokenizerWorker;
use std::collections::HashMap;

#[test]
fn test_qwen2_model_forward() {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        hidden_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_hidden_layers: 2,
        intermediate_size: 128,
        vocab_size: 1000,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        head_dim: Some(16),
        hidden_act: Some("silu".to_string()),
        tie_word_embeddings: Some(true),
        model_type: Some("qwen2".to_string()),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        architectures: Some(vec!["Qwen2ForCausalLM".to_string()]),
    };

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device).unwrap(),
    );

    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{}.", l);
        tensors.insert(
            format!("{}input_layernorm.weight", p),
            Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}post_attention_layernorm.weight", p),
            Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.q_proj.weight", p),
            Tensor::zeros(
                (cfg.num_attention_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.k_proj.weight", p),
            Tensor::zeros(
                (cfg.num_key_value_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.v_proj.weight", p),
            Tensor::zeros(
                (cfg.num_key_value_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.o_proj.weight", p),
            Tensor::zeros(
                (cfg.hidden_size, cfg.num_attention_heads * 16),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}mlp.gate_proj.weight", p),
            Tensor::zeros(
                (cfg.intermediate_size, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}mlp.up_proj.weight", p),
            Tensor::zeros(
                (cfg.intermediate_size, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}mlp.down_proj.weight", p),
            Tensor::zeros(
                (cfg.hidden_size, cfg.intermediate_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let mut model = Qwen2Model::load(cfg, vb).unwrap();

    let input_ids = Tensor::new(&[1u32, 10u32, 20u32], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits = model.forward(&input_ids).unwrap();
    assert_eq!(logits.dims(), &[1, 3, 1000]);
}

#[test]
fn test_qwen3_model_forward() {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        hidden_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_hidden_layers: 2,
        intermediate_size: 128,
        vocab_size: 1000,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        head_dim: Some(16),
        hidden_act: Some("silu".to_string()),
        tie_word_embeddings: Some(true),
        model_type: Some("qwen3".to_string()),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        architectures: Some(vec!["Qwen3ForCausalLM".to_string()]),
    };

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device).unwrap(),
    );

    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{}.", l);
        tensors.insert(
            format!("{}input_layernorm.weight", p),
            Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}post_attention_layernorm.weight", p),
            Tensor::ones(cfg.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.q_proj.weight", p),
            Tensor::zeros(
                (cfg.num_attention_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.k_proj.weight", p),
            Tensor::zeros(
                (cfg.num_key_value_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.v_proj.weight", p),
            Tensor::zeros(
                (cfg.num_key_value_heads * 16, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.o_proj.weight", p),
            Tensor::zeros(
                (cfg.hidden_size, cfg.num_attention_heads * 16),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.q_norm.weight", p),
            Tensor::ones(16, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}self_attn.k_norm.weight", p),
            Tensor::ones(16, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{}mlp.gate_proj.weight", p),
            Tensor::zeros(
                (cfg.intermediate_size, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}mlp.up_proj.weight", p),
            Tensor::zeros(
                (cfg.intermediate_size, cfg.hidden_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
        tensors.insert(
            format!("{}mlp.down_proj.weight", p),
            Tensor::zeros(
                (cfg.hidden_size, cfg.intermediate_size),
                DType::F32,
                &device,
            )
            .unwrap(),
        );
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let mut model = Qwen3Model::load(cfg, vb).unwrap();

    let input_ids = Tensor::new(&[1u32, 10u32, 20u32], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits = model.forward(&input_ids).unwrap();
    assert_eq!(logits.dims(), &[1, 3, 1000]);
}

#[tokio::test]
async fn test_tokenizer_and_engine_step() {
    let tokenizer = TokenizerWorker::new("test-model");
    let prompt = tokenizer.format_chat_prompt(&[("user".to_string(), "Hello".to_string())]);
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("<|im_start|>assistant"));

    let encoded = tokenizer.encode(&prompt).unwrap();
    let decoded = tokenizer.decode(&encoded).unwrap();
    assert_eq!(decoded, prompt);

    let mut engine = LLMEngine::new(EngineConfig::default()).unwrap();
    let req = minisgl::core::req::Req::new(
        1,
        encoded,
        0,
        0,
        5,
        minisgl::core::sampling::SamplingParams::default(),
    );
    let mut reqs = vec![req];
    let tokens = engine.step(&mut reqs).await.unwrap();
    assert_eq!(tokens.len(), 1);
}
