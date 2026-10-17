use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_temperature() -> f32 { 0.0 }
fn default_top_k() -> i32 { -1 }
fn default_top_p() -> f32 { 1.0 }
fn default_max_tokens() -> usize { 1024 }

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
            max_tokens: 1024,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        (self.temperature <= 0.0 || self.top_k == 1) && (self.top_p >= 1.0)
    }
}
