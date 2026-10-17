pub const INDEX_KERNEL_CU: &str = r#"
extern "C" __global__ void index_kernel(
    const int* __restrict__ req_to_token,
    const int* __restrict__ req_lens,
    int* __restrict__ out,
    int num_reqs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_reqs) {
        out[tid] = req_to_token[tid * req_lens[tid]];
    }
}
"#;

pub const STORE_KERNEL_CU: &str = r#"
extern "C" __global__ void store_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        dst[tid] = src[tid];
    }
}
"#;
