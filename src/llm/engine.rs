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
    pub dtype: DType,
    pub eos_token_id: Option<u32>,
}

/// Resolve the best `Device` given optional config overrides.
///
/// Priority:
/// 1. `config.cuda_device_index` — explicit CUDA device index
/// 2. Auto-detect: try CUDA:0, fall back to CPU
fn resolve_device(config: &EngineConfig) -> Device {
    #[cfg(feature = "cuda")]
    {
        let idx = config.cuda_device_index.unwrap_or(0);
        match Device::new_cuda(idx) {
            Ok(dev) => {
                tracing::info!("Using CUDA device {} for inference", idx);
                return dev;
            }
            Err(e) => {
                tracing::warn!(
                    "CUDA device {} not available ({}); falling back to CPU",
                    idx,
                    e
                );
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    if config.cuda_device_index.is_some() {
        tracing::warn!(
            "Binary was compiled without the `cuda` feature; ignoring --device-index. \
             Rebuild with `--features cuda` for GPU inference."
        );
    }

    tracing::info!("Using CPU device for inference");
    Device::Cpu
}

/// Choose compute dtype.
///
/// - Explicit override from `config.dtype_override` wins.
/// - Otherwise: BF16 on CUDA (fast & memory-efficient), F32 on CPU.
fn resolve_dtype(config: &EngineConfig, device: &Device) -> DType {
    if let Some(dt) = config.dtype_override {
        return dt;
    }
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

impl LLMEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let device = resolve_device(&config);
        let dtype = resolve_dtype(&config, &device);
        tracing::info!("Engine dtype: {:?}", dtype);
        Ok(Self {
            config,
            model: None,
            device,
            dtype,
            eos_token_id: Some(151645),
        })
    }

    pub fn load_model(&mut self, model_id_or_path: &str) -> Result<()> {
        tracing::info!(
            "Loading model '{}' on {:?} with dtype {:?}",
            model_id_or_path,
            self.device,
            self.dtype
        );
        let loaded = load_model(model_id_or_path, &self.device, self.dtype)?;
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
                    req.sampling_params.temperature,
                    req.sampling_params.top_p,
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
