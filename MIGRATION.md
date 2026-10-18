# Mini-SGLang Python to Pure Rust Migration Guide

## Architectural Overview

Mini-SGLang has been completely rewritten from Python into a pure Rust high-performance LLM inference workspace.

### Module Mapping

| Python Component (`python/minisgl/*`) | Pure Rust Component (`src/*`) | Description |
|--------------------------------------|--------------------------------|-------------|
| `core.py` | `src/core/req.rs`, `src/core/sampling.rs` | `Req`, `Batch`, `SamplingParams` |
| `kvcache/radix_cache.py` | `src/kvcache/radix_cache.rs` | Trie Radix Cache matching algorithm |
| `kvcache/mha_pool.py` | `src/kvcache/pool.rs` | Paged MHA KV memory allocation |
| `distributed/` | `src/distributed/info.rs`, `src/distributed/tp.rs` | DistributedInfo and TensorParallel primitives |
| `kernel/` | `src/kernel/` | High-performance kernels via `cutile` & `cuda-oxide`, `cudarc::nvrtc`, & Candle fallbacks |
| `layers/` | `src/layers/` | Type-safe `Linear`, `RMSNorm`, `RotaryEmbedding` with multi-backend execution |
| `models/` | `src/models/` | Llama-3, Qwen-2.5, Qwen-3, MoE architecture implementations |
| `scheduler/` | `src/scheduler/scheduler.rs` | Overlap scheduler & prefill/decode logic |
| `tokenizer/` | `src/tokenizer/worker.rs` | High-throughput HuggingFace tokenization worker |
| `server/` | `src/server/api_server.rs`, `src/server/launch.rs` | Axum OpenAI API server (`/v1/chat/completions`) & CLI shell |

## Kernel Execution & Fallback Architecture

1. **cuTile & CUDA Oxide (`cutile`, `cuda-oxide`)**:
   - Primary tile-based kernel execution layer built on NVIDIA's `cuda-rust` repository.
   - Provides optimized GPU execution for index selection, store KV cache, logit sampling, radix cache operations, RMSNorm, Rotary Embeddings, and Fused MoE.
2. **cuDarc (`cudarc`)**:
   - Secondary fallback utilizing NVRTC JIT compilation for CUDA C++ source kernels.
3. **Candle Fallback**:
   - High-portability fallback layer executing tensor operations on Candle (CPU/GPU).

## Build and Installation Commands

```bash
# Build release binary (default feature enables cuda, cutile, and cuda-oxide)
cargo build --release

# Build for CPU-only environments without GPU dependencies
cargo build --no-default-features

# Install locally
cargo install --path .

# Run server
minisgl --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000

# Run in interactive shell mode
minisgl --model Qwen/Qwen3-0.6B --shell
```
