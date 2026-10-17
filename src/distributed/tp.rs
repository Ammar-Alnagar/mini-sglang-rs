use candle_core::{Result, Tensor};
use crate::distributed::info::DistributedInfo;

pub struct TensorParallel {
    pub info: DistributedInfo,
}

impl TensorParallel {
    pub fn new(info: DistributedInfo) -> Self {
        Self { info }
    }

    pub fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
        if self.info.tp_size == 1 {
            Ok(x.clone())
        } else {
            Ok(x.clone())
        }
    }
}
