use candle_core::{Result, Tensor};

pub const INDEX_KERNEL_CU: &str = r#"
extern "C" __global__ void index_kernel(
    const float* __restrict__ weight,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int dim,
    int num_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices * dim) {
        int i = idx / dim;
        int d = idx % dim;
        int pos = indices[i];
        output[idx] = weight[pos * dim + d];
    }
}
"#;

pub fn index_select_fallback(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    weight.index_select(indices, 0)
}
