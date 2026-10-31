use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_key_value_heads() -> usize {
    16
}
fn default_num_hidden_layers() -> usize {
    14
}
fn default_intermediate_size() -> usize {
    2816
}
fn default_vocab_size() -> usize {
    151936
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            num_hidden_layers: default_num_hidden_layers(),
            intermediate_size: default_intermediate_size(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            head_dim: None,
            hidden_act: Some("silu".to_string()),
            tie_word_embeddings: Some(true),
            model_type: Some("qwen".to_string()),
            bos_token_id: Some(151643),
            eos_token_id: Some(151645),
            architectures: None,
        }
    }
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
