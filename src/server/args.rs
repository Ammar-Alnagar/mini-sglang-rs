use clap::Parser;

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
}
