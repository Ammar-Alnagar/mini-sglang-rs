use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            num_hidden_layers: 14,
            intermediate_size: 2816,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
        }
    }
}
