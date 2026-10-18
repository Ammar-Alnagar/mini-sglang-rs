use crate::distributed::info::DistributedInfo;
use candle_core::{Result, Tensor};

pub struct TensorParallel {
    pub info: DistributedInfo,
}

impl TensorParallel {
    pub fn new(info: DistributedInfo) -> Self {
        Self { info }
    }

    pub fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}
