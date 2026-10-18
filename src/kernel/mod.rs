//! Inference kernel abstractions and CUDA execution backends.
//!
//! Includes cuTile / cuda-oxide high performance kernels with cudarc NVRTC
//! and Candle tensor backend fallbacks for indexing, KV caching, sampling, and radix cache.

pub mod index;
pub mod radix;
pub mod sampling;
pub mod store;

pub use index::{index_select, index_select_fallback, INDEX_KERNEL_CU};
pub use radix::{
    radix_index, radix_index_fallback, radix_store, radix_store_fallback, RADIX_INDEX_KERNEL_CU,
    RADIX_STORE_KERNEL_CU,
};
pub use sampling::{
    greedy_sample, sample_logits, sample_top_p_top_k, SAMPLING_KERNEL_CU,
};
pub use store::{store_kv_cache, store_kv_fallback, STORE_KERNEL_CU};
