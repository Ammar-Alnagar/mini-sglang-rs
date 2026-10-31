use crate::models::config::ModelConfig;
use crate::models::qwen2::Qwen2Model;
use crate::models::qwen3::Qwen3Model;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use std::path::{Path, PathBuf};

pub enum LoadedModel {
    Qwen2(Qwen2Model),
    Qwen3(Qwen3Model),
}

impl LoadedModel {
    pub fn forward(&mut self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.forward(input_ids),
            Self::Qwen3(m) => m.forward(input_ids),
        }
    }

    pub fn clear_cache(&mut self) {
        match self {
            Self::Qwen2(m) => m.clear_cache(),
            Self::Qwen3(m) => m.clear_cache(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        match self {
            Self::Qwen2(m) => &m.config,
            Self::Qwen3(m) => &m.config,
        }
    }
}

pub struct ModelPaths {
    pub config_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
    pub tokenizer_path: PathBuf,
}

pub fn resolve_model_files(model_id_or_path: &str) -> Result<ModelPaths> {
    // If model_id_or_path ends with -FP8, check if local or hub repo exists, or fallback to base
    let candidate_ids = if model_id_or_path.ends_with("-FP8") {
        vec![
            model_id_or_path.to_string(),
            model_id_or_path.trim_end_matches("-FP8").to_string(),
        ]
    } else {
        vec![model_id_or_path.to_string()]
    };

    let mut last_err = None;
    for id in candidate_ids {
        let path = Path::new(&id);
        if path.exists() {
            let config_path = path.join("config.json");
            let tokenizer_path = path.join("tokenizer.json");
            let single_safetensor = path.join("model.safetensors");
            let weight_paths = if single_safetensor.exists() {
                vec![single_safetensor]
            } else {
                let mut shards = Vec::new();
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.extension().is_some_and(|ext| ext == "safetensors") {
                            shards.push(p);
                        }
                    }
                }
                shards.sort();
                shards
            };

            if config_path.exists() && !weight_paths.is_empty() {
                return Ok(ModelPaths {
                    config_path,
                    weight_paths,
                    tokenizer_path,
                });
            }
        }

        // Try HuggingFace Hub
        let api = Api::new().context("Failed to initialize HuggingFace API client")?;
        let repo = api.repo(Repo::model(id.clone()));

        let config_path = match repo.get("config.json") {
            Ok(p) => p,
            Err(e) => {
                last_err = Some(anyhow::anyhow!(
                    "Failed to fetch config.json for {}: {}",
                    id,
                    e
                ));
                continue;
            }
        };

        let tokenizer_path = repo.get("tokenizer.json").unwrap_or_else(|_| {
            repo.get("vocab.json")
                .unwrap_or_else(|_| config_path.clone())
        });

        let mut weight_paths = Vec::new();
        if let Ok(p) = repo.get("model.safetensors") {
            weight_paths.push(p);
        } else if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let content = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut shard_names: Vec<String> = map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                shard_names.sort();
                shard_names.dedup();
                for shard in shard_names {
                    if let Ok(sp) = repo.get(&shard) {
                        weight_paths.push(sp);
                    }
                }
            }
        }

        if !weight_paths.is_empty() {
            return Ok(ModelPaths {
                config_path,
                weight_paths,
                tokenizer_path,
            });
        }
    }

    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!("Could not find model weights for {}", model_id_or_path)
    }))
}

pub fn load_model(model_id_or_path: &str, device: &Device, dtype: DType) -> Result<LoadedModel> {
    let paths = resolve_model_files(model_id_or_path)?;
    tracing::info!("Loading model config from {:?}", paths.config_path);
    let config_str = std::fs::read_to_string(&paths.config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    tracing::info!("Loading safetensors weights from {:?}", paths.weight_paths);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths.weight_paths, dtype, device)? };

    let is_qwen3 = config
        .model_type
        .as_deref()
        .is_some_and(|t| t.to_lowercase().contains("qwen3"))
        || config
            .architectures
            .as_ref()
            .is_some_and(|archs| archs.iter().any(|a| a.to_lowercase().contains("qwen3")));

    if is_qwen3 {
        tracing::info!("Detected Qwen3 architecture");
        let model = Qwen3Model::load(config, vb)?;
        Ok(LoadedModel::Qwen3(model))
    } else {
        tracing::info!("Detected Qwen2 architecture");
        let model = Qwen2Model::load(config, vb)?;
        Ok(LoadedModel::Qwen2(model))
    }
}
