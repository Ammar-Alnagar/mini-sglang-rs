# Mini-SGLang Python to Pure Rust Migration Guide

## Architectural Overview

Mini-SGLang is rewritten from Python into a pure Rust workspace for High-Performance LLM Inference.

### Module Mapping Table

| Python Module (`python/minisgl/*`) | Pure Rust Module (`src/*`) | Description |
|--------------------------------------|--------------------------------|-------------|
| `core.py` | `src/core/req.rs`, `src/core/sampling.rs` | Request structures, batch processing, and sampling parameters |
| `kvcache/radix_cache.py` | `src/kvcache/radix_cache.rs` | Radix trie algorithm for KV cache reuse |
| `kvcache/mha_pool.py` | `src/kvcache/pool.rs` | Memory pool allocation for multi-head attention KV blocks |
| `distributed/` | `src/distributed/info.rs`, `src/distributed/tp.rs` | Tensor Parallelism communication primitives |
| `kernel/` | `src/kernel/` | Tile kernels using `cutile` and `cuda-oxide`, `cudarc` NVRTC JIT compilation, and Candle fallbacks |
| `layers/` | `src/layers/` | Neural network components: `Linear`, `RMSNorm`, `RotaryEmbedding` |
| `models/` | `src/models/` | Implementations for Llama-3, Qwen-2.5, Qwen-3, and MoE models |
| `scheduler/` | `src/scheduler/scheduler.rs` | Overlap scheduler for prefill and decode batch operations |
| `tokenizer/` | `src/tokenizer/worker.rs` | HuggingFace tokenizer wrapper for text processing |
| `server/` | `src/server/api_server.rs`, `src/server/launch.rs` | Axum OpenAI API server endpoints and interactive CLI shell |

---

## Kernel Execution and Fallback Architecture

1. **cuTile and CUDA Oxide (`cutile`, `cuda-oxide`)**:
   - Primary execution engine based on NVIDIA CUDA Rust tools.
   - Executes GPU kernels for index selection, paged KV cache storage, logit sampling, radix cache operations, RMSNorm, Rotary Embeddings, and Fused MoE.
2. **cuDarc (`cudarc`)**:
   - Secondary execution fallback using NVRTC JIT compilation for CUDA C++ source files.
3. **Candle Fallback (`Candle`)**:
   - High-compatibility fallback executing tensor operations on CPU or GPU.

---

## Build Commands

```bash
# Build release binary with default features
cargo build --release

# Build for CPU environments without CUDA dependencies
cargo build --no-default-features

# Install binary to cargo path
cargo install --path .

# Run API server
minisgl --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000

# Run interactive shell
minisgl --model Qwen/Qwen3-0.6B --shell
```
