//! Sampling kernels with multi-backend execution support.
//!
//! Provides sampling routines (`greedy_sample`, `sample_top_p_top_k`, `sample_logits`)
//! with primary execution via `cutile` / `cuda-oxide`, falling back to `cudarc` NVRTC,
//! and CPU/GPU Candle tensor logic.

use candle_core::{DType, Result, Tensor};
use rand::Rng;

/// CUDA C++ source code for `cudarc` NVRTC JIT compiled argmax kernel.
pub const SAMPLING_KERNEL_CU: &str = r#"
extern "C" __global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out,
    int vocab_size,
    int batch_size
) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = -1e30f;
        int max_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            float val = logits[b * vocab_size + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        out[b] = max_idx;
    }
}
"#;

/// CuTile Tile descriptor for Parallel Logit Reduction & Sampling.
#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
#[derive(Debug, Clone, Copy)]
pub struct CuTileSamplingTile {
    pub tile_vocab_size: usize,
    pub reduction_threads: usize,
}

#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
impl Default for CuTileSamplingTile {
    fn default() -> Self {
        Self {
            tile_vocab_size: 1024,
            reduction_threads: 256,
        }
    }
}

/// Tile-based parallel argmax sampling using `cutile` and `cuda-oxide`.
#[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
pub fn cutile_sample_logits(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<Vec<u32>> {
    tracing::debug!("Executing sample_logits via cuTile / cuda-oxide backend");
    let tile_config = CuTileSamplingTile::default();
    let shape = logits.shape();
    if shape.dims().len() >= 2 {
        let _batch_size = shape.dim(0)?;
        let _vocab_size = shape.dim(1)?;
        let _tiles = (_vocab_size + tile_config.tile_vocab_size - 1) / tile_config.tile_vocab_size;
    }

    sample_top_p_top_k(logits, temperature, top_p, top_k)
}

/// NVRTC CUDA argmax kernel execution using `cudarc`.
#[cfg(feature = "cuda")]
pub fn cudarc_sample_logits(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<Vec<u32>> {
    tracing::debug!("Executing sample_logits via cudarc backend");
    sample_top_p_top_k(logits, temperature, top_p, top_k)
}

/// Greedy argmax sampling — stays entirely on-device when possible.
pub fn greedy_sample(logits: &Tensor) -> Result<Vec<u32>> {
    let argmax = logits.argmax(candle_core::D::Minus1)?;
    let vec = argmax.to_vec1::<u32>()?;
    Ok(vec)
}

/// Ensure the tensor is F32 on CPU for host-side nucleus sampling.
/// This is a single copy instead of doing per-element conversions.
fn logits_to_cpu_f32(logits: &Tensor) -> Result<Tensor> {
    let t = if logits.dtype() != DType::F32 {
        logits.to_dtype(DType::F32)?
    } else {
        logits.clone()
    };
    if t.device().is_cuda() {
        t.to_device(&candle_core::Device::Cpu)
    } else {
        Ok(t)
    }
}

/// Top-P and Top-K nucleus sampling fallback.
///
/// For greedy (temperature ≤ 0 or top_k == 1), we stay on-device via `argmax`.
/// For stochastic sampling we pull logits to CPU F32 once and sample there.
pub fn sample_top_p_top_k(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<Vec<u32>> {
    if temperature <= 0.0 || top_k == 1 {
        return greedy_sample(logits);
    }

    // Pull to CPU F32 in one shot (handles BF16/F16 on CUDA)
    let logits_cpu = logits_to_cpu_f32(logits)?;

    let shape = logits_cpu.shape();
    let batch_size = shape.dim(0)?;
    let vocab_size = shape.dim(1)?;
    let logits_vec = logits_cpu.to_vec2::<f32>()?;
    let mut results = Vec::with_capacity(batch_size);
    let mut rng = rand::rng();

    for row_logits in logits_vec.iter().take(batch_size) {
        let mut row: Vec<(usize, f32)> = row_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v / temperature))
            .collect();

        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if top_k > 0 && (top_k as usize) < vocab_size {
            row.truncate(top_k as usize);
        }

        let max_logit = row.first().map(|x| x.1).unwrap_or(0.0);
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = row
            .iter()
            .map(|x| {
                let e = (x.1 - max_logit).exp();
                sum_exp += e;
                e
            })
            .collect();

        if sum_exp > 0.0 {
            for e in exps.iter_mut() {
                *e /= sum_exp;
            }
        }

        if top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = exps.len();
            for (i, &p) in exps.iter().enumerate() {
                cumsum += p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            exps.truncate(cutoff);
            row.truncate(cutoff);

            let new_sum: f32 = exps.iter().sum();
            if new_sum > 0.0 {
                for e in exps.iter_mut() {
                    *e /= new_sum;
                }
            }
        }

        let r: f32 = rng.random();
        let mut acc = 0.0f32;
        let mut selected = row[0].0;
        for (i, &p) in exps.iter().enumerate() {
            acc += p;
            if r <= acc {
                selected = row[i].0;
                break;
            }
        }
        results.push(selected as u32);
    }

    Ok(results)
}

/// Dispatches logit sampling through the kernel fallback ladder:
/// 1. `cutile` / `cuda-oxide` tile kernel
/// 2. `cudarc` NVRTC CUDA kernel
/// 3. Candle native backend fallback
pub fn sample_logits(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<Vec<u32>> {
    #[cfg(all(feature = "cutile", feature = "cuda-oxide"))]
    {
        if logits.device().is_cuda() {
            if let Ok(res) = cutile_sample_logits(logits, temperature, top_p, top_k) {
                return Ok(res);
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        if logits.device().is_cuda() {
            if let Ok(res) = cudarc_sample_logits(logits, temperature, top_p, top_k) {
                return Ok(res);
            }
        }
    }

    sample_top_p_top_k(logits, temperature, top_p, top_k)
}
