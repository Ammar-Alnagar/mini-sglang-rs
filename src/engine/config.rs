use crate::distributed::DistributedInfo;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: String,
    pub tp_info: DistributedInfo,
    pub max_running_req: usize,
    pub attention_backend: String,
    pub moe_backend: String,
    pub cuda_graph_bs: Option<Vec<usize>>,
    pub cuda_graph_max_bs: Option<usize>,
    pub page_size: usize,
    pub memory_ratio: f32,
    pub distributed_timeout: f32,
    pub use_dummy_weight: bool,
    pub max_seq_len_override: Option<usize>,
    pub num_page_override: Option<usize>,
    /// Force a specific CUDA device index; `None` means auto-detect.
    pub cuda_device_index: Option<usize>,
    /// Override compute dtype; `None` means auto (bf16 on CUDA, f32 on CPU).
    pub dtype_override: Option<candle_core::DType>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: "Qwen/Qwen3-0.6B".to_string(),
            tp_info: DistributedInfo::default(),
            max_running_req: 256,
            attention_backend: "auto".to_string(),
            moe_backend: "auto".to_string(),
            cuda_graph_bs: None,
            cuda_graph_max_bs: None,
            page_size: 1,
            memory_ratio: 0.9,
            distributed_timeout: 60.0,
            use_dummy_weight: false,
            max_seq_len_override: None,
            num_page_override: None,
            cuda_device_index: None,
            dtype_override: None,
        }
    }
}

impl EngineConfig {
    pub fn max_seq_len(&self, model_max_pos: usize) -> usize {
        self.max_seq_len_override.unwrap_or(model_max_pos)
    }
}
