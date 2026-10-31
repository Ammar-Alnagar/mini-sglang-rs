use crate::models::config::ModelConfig;
use crate::models::qwen2::Qwen2Model;
use crate::models::qwen3::Qwen3Model;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use std::collections::HashMap;
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

/// Precomputed lookup table mapping any `u8` byte representation of an IEEE FP8 E4M3FN float
/// directly to its standard 32-bit float value.
fn fp8_e4m3_to_f32_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    for b in 0..256 {
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 3) & 0x0F;
        let mant = b & 0x07;
        let val = if exp == 0 {
            // subnormal
            sign * 2.0f32.powi(-6) * (mant as f32 / 8.0)
        } else if exp == 15 && mant == 7 {
            // NaN
            0.0f32
        } else {
            // normal
            sign * 2.0f32.powi(exp as i32 - 7) * (1.0 + mant as f32 / 8.0)
        };
        table[b] = val;
    }
    table
}

/// Parse raw bytes of a safetensors BF16 tensor into a `Vec<f32>`.
fn parse_bf16_as_f32(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>().0.iter()
        .map(|chunk| {
            let u16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((u16_val as u32) << 16)
        })
        .collect()
}

/// Convert a non-FP8 safetensors `TensorView` directly into a Candle `Tensor` on `device`
/// with `target_dtype`.
fn convert_raw_tensor(
    view: &safetensors::tensor::TensorView,
    target_dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let shape = view.shape();
    let data = view.data();

    match view.dtype() {
        safetensors::Dtype::BF16 => match target_dtype {
            DType::BF16 => {
                let vec: Vec<half::bf16> = data
                    .as_chunks::<2>().0.iter()
                    .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Tensor::from_vec(vec, shape, device)
            }
            DType::F32 => {
                let vec = parse_bf16_as_f32(data);
                Tensor::from_vec(vec, shape, device)
            }
            DType::F16 => {
                let vec: Vec<half::f16> = parse_bf16_as_f32(data)
                    .into_iter()
                    .map(half::f16::from_f32)
                    .collect();
                Tensor::from_vec(vec, shape, device)
            }
            _ => {
                let vec = parse_bf16_as_f32(data);
                Tensor::from_vec(vec, shape, device)?.to_dtype(target_dtype)
            }
        },
        safetensors::Dtype::F32 => {
            let f32_vec: Vec<f32> = data
                .as_chunks::<4>().0.iter()
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            match target_dtype {
                DType::F32 => Tensor::from_vec(f32_vec, shape, device),
                DType::BF16 => {
                    let vec: Vec<half::bf16> =
                        f32_vec.into_iter().map(half::bf16::from_f32).collect();
                    Tensor::from_vec(vec, shape, device)
                }
                DType::F16 => {
                    let vec: Vec<half::f16> =
                        f32_vec.into_iter().map(half::f16::from_f32).collect();
                    Tensor::from_vec(vec, shape, device)
                }
                _ => Tensor::from_vec(f32_vec, shape, device)?.to_dtype(target_dtype),
            }
        }
        safetensors::Dtype::F16 => {
            let f16_vec: Vec<half::f16> = data
                .as_chunks::<2>().0.iter()
                .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect();
            match target_dtype {
                DType::F16 => Tensor::from_vec(f16_vec, shape, device),
                DType::F32 => {
                    let vec: Vec<f32> = f16_vec.into_iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(vec, shape, device)
                }
                DType::BF16 => {
                    let vec: Vec<half::bf16> = f16_vec
                        .into_iter()
                        .map(|h| half::bf16::from_f32(h.to_f32()))
                        .collect();
                    Tensor::from_vec(vec, shape, device)
                }
                _ => {
                    let vec: Vec<f32> = f16_vec.into_iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(vec, shape, device)?.to_dtype(target_dtype)
                }
            }
        }
        safetensors::Dtype::I64 => {
            let vec: Vec<i64> = data
                .as_chunks::<8>().0.iter()
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Tensor::from_vec(vec, shape, device)
        }
        safetensors::Dtype::U32 => {
            let vec: Vec<u32> = data
                .as_chunks::<4>().0.iter()
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(vec, shape, device)
        }
        safetensors::Dtype::U8 => {
            let vec: Vec<u8> = data.to_vec();
            Tensor::from_vec(vec, shape, device)
        }
        _ => Err(candle_core::Error::Msg(format!(
            "Unsupported safetensors dtype: {:?}",
            view.dtype()
        ))),
    }
}

/// Dequantize FP8 weight tensor using per-block `scale_inv` tensor directly into GPU/CPU memory.
fn dequantize_fp8_block(
    weight_view: &safetensors::tensor::TensorView,
    scale_view: &safetensors::tensor::TensorView,
    target_dtype: DType,
    device: &Device,
    fp8_table: &[f32; 256],
) -> candle_core::Result<Tensor> {
    let w_shape = weight_view.shape();
    let s_shape = scale_view.shape();

    let (rows, cols) = if w_shape.len() == 2 {
        (w_shape[0], w_shape[1])
    } else {
        return Err(candle_core::Error::Msg(format!(
            "Expected 2D shape for FP8 weight, got {:?}",
            w_shape
        )));
    };

    let (s_rows, s_cols) = if s_shape.len() == 2 {
        (s_shape[0], s_shape[1])
    } else {
        return Err(candle_core::Error::Msg(format!(
            "Expected 2D shape for scale_inv, got {:?}",
            s_shape
        )));
    };

    let block_r = rows / s_rows;
    let block_c = cols / s_cols;

    let scale_f32 = match scale_view.dtype() {
        safetensors::Dtype::BF16 => parse_bf16_as_f32(scale_view.data()),
        safetensors::Dtype::F32 => scale_view
            .data()
            .as_chunks::<4>().0.iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported scale_inv dtype: {:?}",
                scale_view.dtype()
            )));
        }
    };

    let weight_bytes = weight_view.data();
    let total_elems = rows * cols;

    match target_dtype {
        DType::BF16 => {
            let mut out = Vec::with_capacity(total_elems);
            for r in 0..rows {
                let sr = r / block_r;
                let scale_row_offset = sr * s_cols;
                let weight_row_offset = r * cols;
                for c in 0..cols {
                    let sc = c / block_c;
                    let scale = scale_f32[scale_row_offset + sc];
                    let raw_b = weight_bytes[weight_row_offset + c];
                    let dequant = fp8_table[raw_b as usize] * scale;
                    out.push(half::bf16::from_f32(dequant));
                }
            }
            Tensor::from_vec(out, (rows, cols), device)
        }
        DType::F16 => {
            let mut out = Vec::with_capacity(total_elems);
            for r in 0..rows {
                let sr = r / block_r;
                let scale_row_offset = sr * s_cols;
                let weight_row_offset = r * cols;
                for c in 0..cols {
                    let sc = c / block_c;
                    let scale = scale_f32[scale_row_offset + sc];
                    let raw_b = weight_bytes[weight_row_offset + c];
                    let dequant = fp8_table[raw_b as usize] * scale;
                    out.push(half::f16::from_f32(dequant));
                }
            }
            Tensor::from_vec(out, (rows, cols), device)
        }
        _ => {
            // F32 and other dtypes
            let mut out = Vec::with_capacity(total_elems);
            for r in 0..rows {
                let sr = r / block_r;
                let scale_row_offset = sr * s_cols;
                let weight_row_offset = r * cols;
                for c in 0..cols {
                    let sc = c / block_c;
                    let scale = scale_f32[scale_row_offset + sc];
                    let raw_b = weight_bytes[weight_row_offset + c];
                    let dequant = fp8_table[raw_b as usize] * scale;
                    out.push(dequant);
                }
            }
            let t = Tensor::from_vec(out, (rows, cols), device)?;
            if target_dtype != DType::F32 {
                t.to_dtype(target_dtype)
            } else {
                Ok(t)
            }
        }
    }
}

/// Load FP8-quantized model: parses safetensors, dequantizes F8E4M3 tensors
/// using their `weight_scale_inv` counterparts, loads directly onto `device` with `target_dtype`.
fn load_model_fp8(
    config: ModelConfig,
    paths: &[PathBuf],
    device: &Device,
    target_dtype: DType,
) -> Result<LoadedModel> {
    tracing::info!(
        "Loading FP8 quantized model with block-dequantization to {:?} on {:?}",
        target_dtype,
        device
    );

    let all_data: Vec<Vec<u8>> = paths
        .iter()
        .map(std::fs::read)
        .collect::<std::io::Result<Vec<_>>>()
        .context("Failed to read safetensors files")?;

    let all_safetensors: Vec<safetensors::SafeTensors> = all_data
        .iter()
        .map(|data| safetensors::SafeTensors::deserialize(data))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {}", e))?;

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
        "Found {} FP8-quantized weight pairs to dequantize directly to {:?}",
        fp8_weight_names.len(),
        device
    );

    let fp8_table = fp8_e4m3_to_f32_table();
    let mut tensor_map = HashMap::<String, Tensor>::new();

    for name in &all_names {
        if scale_inv_names.contains(name) {
            // Skip scale_inv — consumed during dequantization
            continue;
        }

        let view = find_tensor_in_safetensors(&all_safetensors, name)?;

        if fp8_weight_names.contains(name) {
            let base = name.strip_suffix(".weight").unwrap();
            let scale_name = format!("{}.weight_scale_inv", base);
            let scale_view = find_tensor_in_safetensors(&all_safetensors, &scale_name)?;

            let tensor =
                dequantize_fp8_block(&view, &scale_view, target_dtype, device, &fp8_table)?;
            tensor_map.insert(name.clone(), tensor);
        } else {
            let tensor = convert_raw_tensor(&view, target_dtype, device)?;
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
