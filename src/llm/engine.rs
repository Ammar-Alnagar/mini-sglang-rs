use crate::core::req::Req;
use crate::engine::config::EngineConfig;
use anyhow::Result;

pub struct LLMEngine {
    pub config: EngineConfig,
}

impl LLMEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn step(&mut self, reqs: &mut [Req]) -> Result<Vec<u32>> {
        let mut results = Vec::new();
        for req in reqs.iter_mut() {
            if req.can_decode() {
                let token = 100u32;
                req.append_token(token);
                req.complete_one();
                results.push(token);
            }
        }
        Ok(results)
    }
}
