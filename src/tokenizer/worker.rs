use crate::models::loader::resolve_model_files;
use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct TokenizerWorker {
    pub model_path: String,
    tokenizer: Option<Tokenizer>,
    pub eos_token_id: Option<u32>,
}

impl TokenizerWorker {
    pub fn new(model_path: impl Into<String>) -> Self {
        let model_str = model_path.into();
        let mut tokenizer = None;
        let mut eos_token_id = Some(151645); // default Qwen eos_token_id (<|im_end|>)

        // Try to load tokenizer from local path or hf-hub
        if let Ok(paths) = resolve_model_files(&model_str) {
            if paths.tokenizer_path.exists() {
                if let Ok(tok) = Tokenizer::from_file(&paths.tokenizer_path) {
                    if let Some(id) = tok.token_to_id("<|im_end|>") {
                        eos_token_id = Some(id);
                    } else if let Some(id) = tok.token_to_id("<|endoftext|>") {
                        eos_token_id = Some(id);
                    }
                    tokenizer = Some(tok);
                }
            }
        } else if Path::new(&model_str).exists() {
            let tok_file = Path::new(&model_str).join("tokenizer.json");
            if tok_file.exists() {
                if let Ok(tok) = Tokenizer::from_file(tok_file) {
                    if let Some(id) = tok.token_to_id("<|im_end|>") {
                        eos_token_id = Some(id);
                    } else if let Some(id) = tok.token_to_id("<|endoftext|>") {
                        eos_token_id = Some(id);
                    }
                    tokenizer = Some(tok);
                }
            }
        }

        Self {
            model_path: model_str,
            tokenizer,
            eos_token_id,
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(tok) = &self.tokenizer {
            let encoding = tok
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization encode error: {}", e))?;
            Ok(encoding.get_ids().to_vec())
        } else {
            // Fallback byte-level encoding if tokenizer.json is absent
            Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        if let Some(tok) = &self.tokenizer {
            tok.decode(tokens, true)
                .map_err(|e| anyhow::anyhow!("Tokenization decode error: {}", e))
        } else {
            let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
    }

    pub fn format_chat_prompt(&self, messages: &[(String, String)]) -> String {
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}
