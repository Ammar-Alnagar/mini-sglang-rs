use anyhow::Result;

pub struct TokenizerWorker {
    pub model_path: String,
}

impl TokenizerWorker {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }
}
