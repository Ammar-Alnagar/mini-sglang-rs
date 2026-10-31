use crate::core::req::Req;
use crate::engine::config::EngineConfig;
use crate::kernel::sampling::sample_logits;
use crate::models::loader::{load_model, LoadedModel};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct LLMEngine {
    pub config: EngineConfig,
    pub model: Option<LoadedModel>,
    pub device: Device,
    pub eos_token_id: Option<u32>,
}

impl LLMEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self {
            config,
            model: None,
            device,
            eos_token_id: Some(151645),
        })
    }

    pub fn load_model(&mut self, model_id_or_path: &str) -> Result<()> {
        let loaded = load_model(model_id_or_path, &self.device, DType::F32)?;
        self.eos_token_id = loaded.config().eos_token_id.or(Some(151645));
        self.model = Some(loaded);
        Ok(())
    }

    pub async fn step(&mut self, reqs: &mut [Req]) -> Result<Vec<u32>> {
        let mut results = Vec::new();

        for req in reqs.iter_mut() {
            if !req.can_decode() {
                continue;
            }

            let next_token = if let Some(model) = &mut self.model {
                // If this is the first step for this request, pass all input_ids
                // Otherwise pass only the last token if KV-cache is active, or all input_ids if clearing cache
                let input_tensor = if req.generated_tokens.is_empty() {
                    model.clear_cache();
                    Tensor::new(req.input_ids.as_slice(), &self.device)?.unsqueeze(0)?
                } else {
                    let last_token = *req.input_ids.last().unwrap();
                    Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?
                };

                let logits = model.forward(&input_tensor)?;
                // Logits shape: [1, seq_len, vocab_size]
                let (_, seq_len, _) = logits.dims3()?;
                let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

                let sampled = sample_logits(
                    &last_logits,
                    req.sampling_params.temperature as f32,
                    req.sampling_params.top_p as f32,
                    req.sampling_params.top_k,
                )?;
                sampled.first().copied().unwrap_or(0)
            } else {
                100u32
            };

            req.append_token(next_token);
            req.complete_one();

            if let Some(eos) = self.eos_token_id {
                if !req.sampling_params.ignore_eos && next_token == eos {
                    req.is_finished = true;
                }
            }

            if req.remain_len() == 0 {
                req.is_finished = true;
            }

            results.push(next_token);
        }

        Ok(results)
    }
}
