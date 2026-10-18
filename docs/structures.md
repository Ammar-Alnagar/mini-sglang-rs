# Structure of Mini-SGLang

## System Architecture

Mini-SGLang is designed as a high-performance system to handle Large Language Model (LLM) inference efficiently in pure Rust.

### Key Components

- **API Server**: Built with Axum. It provides an OpenAI-compatible API (`/v1/chat/completions`) and `/health`.
- **Tokenizer Worker**: High-throughput HuggingFace tokenization worker.
- **Scheduler Worker**: Manages computation, prefill/decode batches, and KV cache allocation.
- **Engine**: Handles model weights, Candle tensor operations, and CUDA execution.

## Code Organization (Rust Workspace)

The source code is located in `src/`. Here is a breakdown of the Rust modules:

- `src/core`: Defines `Req`, `Batch`, and `SamplingParams`.
- `src/distributed`: Provides Tensor Parallelism primitives (`tp.rs`) and `DistributedInfo`.
- `src/layers`: Implements building blocks like `Linear`, `RMSNorm`, `RotaryEmbedding`.
- `src/models`: Implements LLM model architectures (`llama.rs`, `qwen2.rs`, `qwen3.rs`).
- `src/kvcache`: Implements `RadixCache` and KV pool allocations.
- `src/utils`: Logging and environment configurations.
- `src/engine`: Engine initialization and Candle tensor context.
- `src/message`: Axum protocol types for OpenAI API compatibility.
- `src/scheduler`: Implements `Scheduler` with overlap prefill/decode logic.
- `src/server`: Axum API server (`api_server.rs`) and CLI launcher (`launch.rs`).
- `src/tokenizer`: HuggingFace tokenizer wrapper worker.
- `src/kernel`: CUDA NVRTC kernel bindings and Candle fallback kernels.
