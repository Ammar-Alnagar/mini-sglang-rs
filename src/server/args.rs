use clap::Parser;

/// Device selection for inference.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceArg {
    /// Automatically select CUDA if available, otherwise CPU.
    #[default]
    Auto,
    /// Force CUDA GPU (index selectable via `--device-index`).
    Cuda,
    /// Force CPU.
    Cpu,
}

impl std::str::FromStr for DeviceArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" | "gpu" => Ok(Self::Cuda),
            "cpu" => Ok(Self::Cpu),
            _ => Err(format!("Unknown device '{}'. Use: auto, cuda, cpu", s)),
        }
    }
}

/// Compute dtype for model weights & activations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DTypeArg {
    /// Pick automatically: bf16 on CUDA, f32 on CPU.
    #[default]
    Auto,
    BF16,
    F16,
    F32,
}

impl std::str::FromStr for DTypeArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            "f16" | "float16" | "fp16" => Ok(Self::F16),
            "f32" | "float32" | "fp32" => Ok(Self::F32),
            _ => Err(format!(
                "Unknown dtype '{}'. Use: auto, bf16, f16, f32",
                s
            )),
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Mini-SGLang Pure Rust LLM Inference Server")]
pub struct Args {
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    #[arg(long, default_value_t = 1)]
    pub tp: usize,

    #[arg(long)]
    pub shell: bool,

    /// Device to use for inference: auto, cuda, cpu  [default: auto]
    #[arg(long, default_value = "auto")]
    pub device: DeviceArg,

    /// CUDA device index (for multi-GPU systems).  [default: 0]
    #[arg(long, default_value_t = 0)]
    pub device_index: usize,

    /// Compute dtype: auto, bf16, f16, f32  [default: auto]
    #[arg(long, default_value = "auto")]
    pub dtype: DTypeArg,
}
