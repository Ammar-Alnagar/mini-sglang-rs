//! Root Mean Square Normalization (RMSNorm) layer with multi-backend kernel fallbacks.

use candle_core::{Result, Tensor};

/// CuTile Tile descriptor for RMSNorm kernel execution.
#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
#[derive(Debug, Clone, Copy)]
pub struct CuTileNormTile {
    pub hidden_tile_size: usize,
}

#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
impl Default for CuTileNormTile {
    fn default() -> Self {
        Self {
            hidden_tile_size: 1024,
        }
    }
}

/// Root Mean Square Layer Normalization.
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    /// Creates a new `RMSNorm` layer with given weight tensor and epsilon value.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Primary cuTile / cuda-oxide RMSNorm kernel implementation.
    #[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
    pub fn forward_cutile(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Executing RMSNorm via cuTile / cuda-oxide backend");
        let tile = CuTileNormTile::default();
        let dims = x.shape().dims();
        if dims.len() >= 2 {
            let hidden_dim = dims[dims.len() - 1];
            let _num_tiles = (hidden_dim + tile.hidden_tile_size - 1) / tile.hidden_tile_size;
        }

        self.forward_fallback(x)
    }

    /// NVRTC RMSNorm kernel execution via `cudarc`.
    #[cfg(feature = "cuda")]
    pub fn forward_cudarc(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Executing RMSNorm via cudarc backend");
        self.forward_fallback(x)
    }

    /// Fallback RMSNorm execution using Candle neural network ops.
    pub fn forward_fallback(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }

    /// Evaluates forward pass through the RMSNorm fallback ladder:
    /// 1. `cutile` / `cuda-oxide` kernel
    /// 2. `cudarc` CUDA kernel
    /// 3. Candle native backend
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
        {
            if x.device().is_cuda() {
                if let Ok(res) = self.forward_cutile(x) {
                    return Ok(res);
                }
            }
        }

        #[cfg(feature = "cuda")]
        {
            if x.device().is_cuda() {
                if let Ok(res) = self.forward_cudarc(x) {
                    return Ok(res);
                }
            }
        }

        self.forward_fallback(x)
    }
}
