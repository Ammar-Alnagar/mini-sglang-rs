use crate::models::config::ModelConfig;
use candle_core::{Result, Tensor};

pub struct LlamaModel {
    pub config: ModelConfig,
}

impl LlamaModel {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(input_ids.clone())
    }
}
