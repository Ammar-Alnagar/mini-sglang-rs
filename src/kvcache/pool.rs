use candle_core::{Device, DType, Result, Tensor};

pub struct KVCachePool {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub num_pages: usize,
    pub page_size: usize,
}

impl KVCachePool {
    pub fn new(
        num_pages: usize,
        page_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k_cache = Tensor::zeros((num_pages, page_size, num_heads, head_dim), dtype, device)?;
        let v_cache = Tensor::zeros((num_pages, page_size, num_heads, head_dim), dtype, device)?;
        Ok(Self {
            k_cache,
            v_cache,
            num_pages,
            page_size,
        })
    }
}
