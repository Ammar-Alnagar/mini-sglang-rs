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

/// Detect if any safetensors file contains F8_E4M3 tensors, indicating FP8 quantized weights.
fn detect_fp8_weights(paths: &[PathBuf]) -> bool {
    for path in paths {
        if let Ok(file) = std::fs::File::open(path) {
            use std::io::Read;
            let mut reader = std::io::BufReader::new(file);
            let mut header_size_buf = [0u8; 8];
            if reader.read_exact(&mut header_size_buf).is_ok() {
                let header_size = u64::from_le_bytes(header_size_buf) as usize;
                let mut header_buf = vec![0u8; header_size];
                if reader.read_exact(&mut header_buf).is_ok() {
                    if let Ok(header_str) = std::str::from_utf8(&header_buf) {
                        if header_str.contains("\"F8_E4M3\"") {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Dequantize a single F8E4M3 weight tensor using its per-block scale_inv tensor.
///
/// FP8 block-quantized format:
///   weight_dequant = weight_fp8.to(target_dtype) * scale_inv_tiled
///
/// `scale_inv` has shape `[rows/block_size, cols/block_size]` where `block_size` is
/// determined by dividing the weight shape by the scale shape.
fn dequantize_fp8_weight(
    weight_fp8: &Tensor,
    scale_inv: &Tensor,
    target_dtype: DType,
) -> candle_core::Result<Tensor> {
    let weight_bf16 = weight_fp8.to_dtype(target_dtype)?;
    let w_shape = weight_bf16.dims();
    let s_shape = scale_inv.dims();

    if w_shape.len() != 2 || s_shape.len() != 2 {
        // Fallback: just return casted weight without scaling
        tracing::warn!(
            "FP8 dequant: unexpected shapes w={:?} s={:?}, returning raw cast",
            w_shape,
            s_shape
        );
        return Ok(weight_bf16);
    }

    let (w_rows, w_cols) = (w_shape[0], w_shape[1]);
    let (s_rows, s_cols) = (s_shape[0], s_shape[1]);

    let block_r = w_rows / s_rows;
    let block_c = w_cols / s_cols;

    // Reshape weight to [s_rows, block_r, s_cols, block_c]
    // then broadcast-multiply with scale_inv [s_rows, 1, s_cols, 1]
    // then reshape back to [w_rows, w_cols]
    let weight_reshaped = weight_bf16.reshape((s_rows, block_r, s_cols, block_c))?;
    let scale_expanded = scale_inv
        .to_dtype(target_dtype)?
        .reshape((s_rows, 1, s_cols, 1))?;
    let dequantized = weight_reshaped.broadcast_mul(&scale_expanded)?;
    dequantized.reshape((w_rows, w_cols))
}

/// Convert a safetensors TensorView to a candle Tensor on the given device.
fn safetensor_view_to_tensor(
    view: &safetensors::tensor::TensorView,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = match view.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F8_E4M3 => DType::F8E4M3,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::U32 => DType::U32,
        safetensors::Dtype::U8 => DType::U8,
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported safetensors dtype: {:?}",
                view.dtype()
            )));
        }
    };

    let data = view.data();
    Tensor::from_raw_buffer(data, dtype, &shape, device)
}

/// Load FP8-quantized model: loads weights natively, dequantizes F8E4M3 tensors
/// using their `weight_scale_inv` counterparts, then constructs the model.
fn load_model_fp8(
    config: ModelConfig,
    paths: &[PathBuf],
    device: &Device,
    target_dtype: DType,
) -> Result<LoadedModel> {
    tracing::info!(
        "Loading FP8 quantized model with dequantization to {:?}",
        target_dtype
    );

    // Read all safetensors files into memory
    let all_data: Vec<Vec<u8>> = paths
        .iter()
        .map(|p| std::fs::read(p))
        .collect::<std::io::Result<Vec<_>>>()
        .context("Failed to read safetensors files")?;

    let all_safetensors: Vec<safetensors::SafeTensors> = all_data
        .iter()
        .map(|data| safetensors::SafeTensors::deserialize(data))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {}", e))?;

    // Collect all tensor names
    let mut all_names: Vec<String> = Vec::new();
    for st in &all_safetensors {
        for (name, _) in st.tensors() {
            if name != "__metadata__" {
                all_names.push(name.to_string());
            }
        }
    }
    all_names.sort();
    all_names.dedup();

    // Identify FP8 weight/scale_inv pairs
    let mut fp8_weight_names = std::collections::HashSet::<String>::new();
    let mut scale_inv_names = std::collections::HashSet::<String>::new();

    for name in &all_names {
        if name.ends_with(".weight") {
            let base = name.strip_suffix(".weight").unwrap();
            let scale_name = format!("{}.weight_scale_inv", base);
            if all_names.contains(&scale_name) {
                fp8_weight_names.insert(name.clone());
                scale_inv_names.insert(scale_name);
            }
        }
    }

    tracing::info!(
        "Found {} FP8-quantized weight pairs to dequantize",
        fp8_weight_names.len()
    );

    // Build dequantized tensor map
    let mut tensor_map = std::collections::HashMap::<String, Tensor>::new();

    for name in &all_names {
        if scale_inv_names.contains(name) {
            // Skip scale_inv — consumed during dequantization
            continue;
        }

        let view = find_tensor_in_safetensors(&all_safetensors, name)?;

        if fp8_weight_names.contains(name) {
            // FP8 weight: load raw + scale_inv and dequantize
            let base = name.strip_suffix(".weight").unwrap();
            let scale_name = format!("{}.weight_scale_inv", base);
            let scale_view = find_tensor_in_safetensors(&all_safetensors, &scale_name)?;

            let weight_raw = safetensor_view_to_tensor(&view, device)?;
            let scale_inv = safetensor_view_to_tensor(&scale_view, device)?;

            let dequantized = dequantize_fp8_weight(&weight_raw, &scale_inv, target_dtype)?;
            tracing::debug!(
                "Dequantized FP8 tensor {} ({:?} -> {:?})",
                name,
                view.dtype(),
                target_dtype
            );
            tensor_map.insert(name.clone(), dequantized);
        } else {
            // Normal tensor: load and cast to target dtype
            let tensor = safetensor_view_to_tensor(&view, device)?
                .to_dtype(target_dtype)?;
            tensor_map.insert(name.clone(), tensor);
        }
    }

    let vb = VarBuilder::from_tensors(tensor_map, target_dtype, device);
    load_model_from_vb(config, vb)
}

/// Find a tensor by name across multiple safetensors shards.
fn find_tensor_in_safetensors<'a>(
    shards: &'a [safetensors::SafeTensors<'a>],
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    for st in shards {
        if let Ok(view) = st.tensor(name) {
            return Ok(view);
        }
    }
    anyhow::bail!("Tensor '{}' not found in any safetensors shard", name)
}

/// Load model from a VarBuilder (shared logic for normal and FP8 paths).
fn load_model_from_vb(config: ModelConfig, vb: VarBuilder) -> Result<LoadedModel> {
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

pub fn load_model(model_id_or_path: &str, device: &Device, dtype: DType) -> Result<LoadedModel> {
    let paths = resolve_model_files(model_id_or_path)?;
    tracing::info!("Loading model config from {:?}", paths.config_path);
    let config_str = std::fs::read_to_string(&paths.config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    tracing::info!("Loading safetensors weights from {:?}", paths.weight_paths);

    // Check if this is an FP8 quantized model
    if detect_fp8_weights(&paths.weight_paths) {
        tracing::info!(
            "Detected FP8 (F8_E4M3) quantized weights — will dequantize to {:?}",
            dtype
        );
        return load_model_fp8(config, &paths.weight_paths, device, dtype);
    }

    // Standard loading path
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths.weight_paths, dtype, device)? };
    load_model_from_vb(config, vb)
}
