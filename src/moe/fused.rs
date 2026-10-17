use candle_core::{Result, Tensor};

pub struct FusedMoE {
    pub num_experts: usize,
    pub top_k: usize,
}

impl FusedMoE {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}
