use candle_core::{Result, Tensor};
use crate::models::config::ModelConfig;

pub struct Qwen3Model {
    pub config: ModelConfig,
}

impl Qwen3Model {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(input_ids.clone())
    }
}
