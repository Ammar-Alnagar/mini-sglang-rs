use candle_core::{Result, Tensor};

pub const STORE_KERNEL_CU: &str = r#"
extern "C" __global__ void store_kv_cache_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ indices,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int stride,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length * stride) {
        int i = idx / stride;
        int d = idx % stride;
        int pos = indices[i];
        k_cache[pos * stride + d] = k[idx];
        v_cache[pos * stride + d] = v[idx];
    }
}
"#;

pub fn store_kv_fallback(
    _k_cache: &mut Tensor,
    _v_cache: &mut Tensor,
    _indices: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
) -> Result<()> {
    Ok(())
}
