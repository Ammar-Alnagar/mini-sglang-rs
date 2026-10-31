use crate::core::req::Req;
use crate::core::sampling::SamplingParams;
use crate::engine::config::EngineConfig;
use crate::llm::engine::LLMEngine;
use crate::server::api_server;
use crate::server::args::{Args, DeviceArg, DTypeArg};
use crate::tokenizer::worker::TokenizerWorker;
use anyhow::Result;
use candle_core::DType;
use std::io::{self, BufRead, Write};

/// Build an `EngineConfig` from CLI args.
fn engine_config_from_args(args: &Args) -> EngineConfig {
    let mut cfg = EngineConfig::default();

    // Resolve CUDA device index
    match &args.device {
        DeviceArg::Cuda => {
            cfg.cuda_device_index = Some(args.device_index);
        }
        DeviceArg::Auto => {
            // auto: try CUDA:device_index, fall back to CPU inside the engine
            cfg.cuda_device_index = Some(args.device_index);
        }
        DeviceArg::Cpu => {
            cfg.cuda_device_index = None;
        }
    }

    // Resolve explicit dtype override
    cfg.dtype_override = match &args.dtype {
        DTypeArg::Auto => None,
        DTypeArg::BF16 => Some(DType::BF16),
        DTypeArg::F16 => Some(DType::F16),
        DTypeArg::F32 => Some(DType::F32),
    };

    cfg
}

pub async fn run(args: Args) -> Result<()> {
    if args.shell {
        println!("Interactive Shell Mode (Mini-SGLang Pure Rust)");
        println!("Model: {}", args.model);
        println!("Type '/reset' to clear chat history, or press Ctrl+C to exit.\n");

        println!("Loading model weights and tokenizer...");
        let tokenizer = TokenizerWorker::new(&args.model);
        let cfg = engine_config_from_args(&args);
        let mut engine = LLMEngine::new(cfg)?;
        if let Err(e) = engine.load_model(&args.model) {
            eprintln!(
                "Warning: Failed to load model weights ({}). Running in fallback mode.",
                e
            );
        } else {
            println!("Model loaded successfully.\n");
        }

        let mut history: Vec<(String, String)> = Vec::new();
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            print!("User: ");
            stdout.flush()?;

            let mut line = String::new();
            if stdin.lock().read_line(&mut line)? == 0 {
                // EOF reached
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "/reset" {
                history.clear();
                println!("Chat history cleared.\n");
                continue;
            }

            history.push(("user".to_string(), trimmed.to_string()));
            let prompt = tokenizer.format_chat_prompt(&history);
            let token_ids = tokenizer.encode(&prompt)?;

            let params = SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                ignore_eos: false,
                max_tokens: 512,
            };

            let mut req = Req::new(1, token_ids, 0, 0, params.max_tokens, params);

            print!("Assistant: ");
            stdout.flush()?;

            let mut generated = Vec::new();
            while req.can_decode() {
                let mut batch_reqs = vec![req];
                let tokens = engine.step(&mut batch_reqs).await?;
                req = batch_reqs.pop().unwrap();

                if let Some(&tok) = tokens.first() {
                    generated.push(tok);
                    if let Ok(piece) = tokenizer.decode(&[tok]) {
                        print!("{}", piece);
                        stdout.flush()?;
                    }
                }
            }

            println!("\n");
            let mut response_text = tokenizer.decode(&generated).unwrap_or_default();
            if let Some(stripped) = response_text.strip_suffix("<|im_end|>") {
                response_text = stripped.to_string();
            } else if let Some(stripped) = response_text.strip_suffix("<|endoftext|>") {
                response_text = stripped.to_string();
            }
            history.push(("assistant".to_string(), response_text));
        }

        return Ok(());
    }

    let cfg = engine_config_from_args(&args);
    api_server::start_server(&args.host, args.port, &args.model, cfg).await
}
