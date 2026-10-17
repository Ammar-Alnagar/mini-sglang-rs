use candle_core::{Result, Tensor};
use crate::models::config::ModelConfig;

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
