use candle_core::{Result, Tensor};

pub fn silu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::silu(x)
}
