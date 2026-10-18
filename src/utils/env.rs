use std::env;

pub struct EnvConfig {
    pub shell_max_tokens: usize,
    pub shell_top_k: i32,
    pub shell_top_p: f32,
    pub shell_temperature: f32,
    pub flashinfer_use_tensor_cores: Option<bool>,
    pub disable_overlap_scheduling: bool,
    pub pynccl_max_buffer_size: usize,
}

impl EnvConfig {
    pub fn from_env() -> Self {
        Self {
            shell_max_tokens: parse_env("MINISGL_SHELL_MAX_TOKENS", 2048),
            shell_top_k: parse_env("MINISGL_SHELL_TOP_K", -1),
            shell_top_p: parse_env("MINISGL_SHELL_TOP_P", 1.0),
            shell_temperature: parse_env("MINISGL_SHELL_TEMPERATURE", 0.6),
            flashinfer_use_tensor_cores: parse_env_opt_bool("MINISGL_FLASHINFER_USE_TENSOR_CORES"),
            disable_overlap_scheduling: parse_env_bool("MINISGL_DISABLE_OVERLAP_SCHEDULING", false),
            pynccl_max_buffer_size: parse_env_mem(
                "MINISGL_PYNCCL_MAX_BUFFER_SIZE",
                1024 * 1024 * 1024,
            ),
        }
    }
}

fn parse_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .map(|v| match v.to_lowercase().as_str() {
            "1" | "true" | "yes" => true,
            "0" | "false" | "no" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn parse_env_opt_bool(key: &str) -> Option<bool> {
    env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
}

fn parse_env_mem(key: &str, default: usize) -> usize {
    env::var(key)
        .map(|v| parse_mem_bytes(&v).unwrap_or(default))
        .unwrap_or(default)
}

fn parse_mem_bytes(mem: &str) -> Option<usize> {
    let s = mem.trim().to_uppercase();
    if s.is_empty() {
        return None;
    }
    if s.ends_with("GB") || s.ends_with('G') {
        let val = s.trim_end_matches("GB").trim_end_matches('G');
        Some((val.parse::<f64>().ok()? * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if s.ends_with("MB") || s.ends_with('M') {
        let val = s.trim_end_matches("MB").trim_end_matches('M');
        Some((val.parse::<f64>().ok()? * 1024.0 * 1024.0) as usize)
    } else if s.ends_with("KB") || s.ends_with('K') {
        let val = s.trim_end_matches("KB").trim_end_matches('K');
        Some((val.parse::<f64>().ok()? * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
}
