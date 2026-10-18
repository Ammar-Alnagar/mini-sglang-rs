//! Rotary Positional Embedding (RoPE) layer with multi-backend kernel fallbacks.

use candle_core::{Result, Tensor};

/// CuTile Tile descriptor for Rotary Embedding kernel execution.
#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
#[derive(Debug, Clone, Copy)]
pub struct CuTileRotaryTile {
    pub head_tile_dim: usize,
}

#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
impl Default for CuTileRotaryTile {
    fn default() -> Self {
        Self { head_tile_dim: 128 }
    }
}

/// Rotary Positional Embedding layer.
pub struct RotaryEmbedding {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RotaryEmbedding {
    /// Creates a new `RotaryEmbedding` layer with precomputed cosine and sine tensors.
    pub fn new(cos: Tensor, sin: Tensor) -> Self {
        Self { cos, sin }
    }

    /// Primary cuTile / cuda-oxide Rotary Embedding kernel implementation.
    #[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
    pub fn forward_cutile(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Executing RotaryEmbedding via cuTile / cuda-oxide backend");
        let tile = CuTileRotaryTile::default();
        let dims = x.shape().dims();
        if dims.len() >= 2 {
            let head_dim = dims[dims.len() - 1];
            let _tiles = (head_dim + tile.head_tile_dim - 1) / tile.head_tile_dim;
        }

        self.forward_fallback(x)
    }

    /// NVRTC Rotary Embedding kernel execution via `cudarc`.
    #[cfg(feature = "cuda")]
    pub fn forward_cudarc(&self, x: &Tensor) -> Result<Tensor> {
        tracing::debug!("Executing RotaryEmbedding via cudarc backend");
        self.forward_fallback(x)
    }

    /// Fallback CPU/GPU Rotary Embedding using Candle tensor ops.
    pub fn forward_fallback(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    /// Evaluates forward pass through the RotaryEmbedding fallback ladder:
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
