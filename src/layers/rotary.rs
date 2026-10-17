use candle_core::{Result, Tensor};

pub struct RotaryEmbedding {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cos: Tensor, sin: Tensor) -> Self {
        Self { cos, sin }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}
