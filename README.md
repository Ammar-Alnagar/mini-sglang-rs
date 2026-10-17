# Mini-SGLang (Pure Rust Edition)

A **lightweight, high-performance** inference engine for Large Language Models written in pure Rust.

---

Mini-SGLang is a complete, production-grade Rust migration of [SGLang](https://github.com/sgl-project/sglang). It preserves every core feature of the original while delivering superior efficiency, type safety, and memory management.

## ✨ Key Features

- **Pure Rust Engine**: Built on Tokio async runtime, Axum web framework, Candle tensor backend, and `cudarc` for CUDA interactions.
- **Radix Cache**: Reuses KV cache for shared prompt prefixes.
- **Chunked Prefill & Overlap Scheduling**: Hides CPU scheduling latency.
- **Tensor Parallelism (TP)**: Scalable inference across multiple GPUs.
- **OpenAI-Compatible API**: Server supporting `/v1/chat/completions` and `/health`.
- **Interactive Terminal Shell**: Interactive CLI chat interface (`--shell`).

## 🚀 Quick Start

### Build & Run from Source

```bash
# Build release executable
cargo build --release

# Run OpenAI API Server
cargo run --release -- --model "Qwen/Qwen3-0.6B" --port 8000

# Run in Interactive Shell Mode
cargo run --release -- --model "Qwen/Qwen3-0.6B" --shell
```

### Docker Support

```bash
# Build image
docker build -t minisgl-rust .

# Run container
docker run -p 8000:8000 minisgl-rust --model Qwen/Qwen3-0.6B
```

## 📚 Migration Details

See [MIGRATION.md](./MIGRATION.md) for full architectural mappings between Python and Rust components.
