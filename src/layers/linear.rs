use candle_core::{Result, Tensor};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [.., in_features] => {
                let w_dims = self.weight.dims();
                if w_dims.len() == 2 && w_dims[0] == in_features {
                    self.weight.clone()
                } else {
                    self.weight.t()?
                }
            }
            _ => self.weight.t()?,
        };
        let out = x.broadcast_matmul(&w)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}
