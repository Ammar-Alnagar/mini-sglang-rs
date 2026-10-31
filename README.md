# Mini-SGLang (Pure Rust Edition)

A lightweight inference engine for Large Language Models written in pure Rust.

---

Mini-SGLang is a Rust migration of SGLang. It preserves core features of the original engine while delivering memory management, type safety, and execution through the Tokio async runtime, Axum framework, Candle tensor engine, `cutile` and `cuda-oxide` from NVIDIA CUDA Rust tools, and `cudarc` CUDA operations.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        HTTP["HTTP Client (curl or OpenAI SDK)"]
        CLI["Interactive CLI Shell (--shell)"]
    end

    subgraph Server ["Axum Web Server Layer"]
        Health["GET /health"]
        Chat["POST /v1/chat/completions"]
        Chan["Tokio mpsc Channel Buffer"]
        Args["CLI Arguments Parser (clap)"]
    end

    subgraph Pipeline ["Inference Pipeline"]
        Worker["Background Engine Loop"]
        Tok["Tokenizer Worker"]
        Radix["Radix Cache"]
        Sched["Scheduler"]
    end

    subgraph Compute ["LLM Compute Engine"]
        Eng["LLM Engine"]
        CuTile["NVIDIA cuTile and cuda-oxide"]
        CuDarc["cudarc NVRTC Fallback"]
        Candle["Candle Tensor Fallback"]
        KVPool["KV Cache Pool"]
    end

    HTTP -->|JSON Request| Chat
    CLI -->|Prompt Input| Args
    Args -->|Initialize| Server
    Chat -->|Send to Channel| Chan
    Chan -->|Batch Intake| Worker
    Worker -->|Extract Prompt| Tok
    Tok -->|Token Sequence| Radix
    Radix -->|Prefix Allocation| Sched
    Sched -->|Batch Construction| Eng
    Eng -->|Tile Kernels| CuTile
    CuTile -.->|Fallback| CuDarc
    CuDarc -.->|Fallback| Candle
    Eng <-->|KV Memory Management| KVPool
    Eng -->|Sampled Output Token| Tok
    Tok -->|Decoded Response| Worker
    Worker -->|oneshot Response| Chat
    Chat -->|JSON Response| HTTP
```

---

## Kernel Execution Hierarchy

Mini-SGLang uses a three-tier kernel execution pipeline for performance across hardware configurations:

1. **Primary Backend (`cuTile` and `cuda-oxide`)**:
   - Built on NVIDIA CUDA Rust tools (`cutile` and `cuda-oxide`).
   - Executes tile-based GPU kernels for index selection, paged KV storage, logit sampling, radix cache operations, RMSNorm, RoPE, and fused MoE.
2. **Secondary Fallback Backend (`cudarc` NVRTC)**:
   - Uses `cudarc` NVRTC runtime compilation for CUDA C++ source kernels when cuTile kernels are inactive.
3. **Tertiary Fallback (`Candle`)**:
   - Executes tensor operations on CPU or GPU via Candle when custom CUDA kernels are disabled.

---

## Request Execution Flow

The sequence diagram below shows the request lifecycle from entry to completion.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Axum as Axum Web Server
    participant Channel as mpsc Buffer Channel
    participant Worker as Background Engine Worker
    participant Tokenizer as Tokenizer Worker
    participant Cache as Radix Cache
    participant Scheduler as Scheduler
    participant Engine as LLM Engine
    participant CUDA as CUDA Execution Backend

    Client->>Axum: POST /v1/chat/completions
    Axum->>Axum: Parse ChatCompletionRequest
    Axum->>Channel: Enqueue ServerRequest with oneshot handle
    Channel->>Worker: Collect request batch
    Worker->>Tokenizer: Encode prompt text
    Tokenizer-->>Worker: Token IDs
    Worker->>Cache: Match prompt prefix
    Cache-->>Worker: Cached page index data
    Worker->>Scheduler: Register Request
    
    loop Inference Execution
        Scheduler->>Scheduler: Build request Batch
        Scheduler-->>Engine: Process Batch
        Engine->>CUDA: Run forward pass
        CUDA-->>Engine: Compute Logits
        Engine->>Engine: Sample next token
        Engine->>Scheduler: Update Request status
    end

    Scheduler-->>Worker: Completed Request tokens
    Worker->>Tokenizer: Decode token sequence
    Tokenizer-->>Worker: Decoded string
    Worker->>Axum: Return via oneshot handle
    Axum-->>Client: Return 200 OK Response
```

### Request Lifecycle Steps

1. **Request Entry**:
   - The client sends an HTTP POST request to `/v1/chat/completions`.
   - Axum converts the JSON payload into a `ChatCompletionRequest` structure.
2. **Async Channel Enqueue**:
   - Axum enqueues the request into an asynchronous `mpsc` channel and awaits response notification via a `oneshot` channel handle.
3. **Dynamic Batching**:
   - The background engine loop collects pending requests from the channel buffer into dynamic execution batches.
4. **Tokenization**:
   - The engine sends prompt text to `TokenizerWorker` to generate numerical token IDs.
5. **Radix Cache Prefix Matching**:
   - `RadixCache` checks existing prefix entries to reuse stored KV cache blocks.
6. **Scheduling**:
   - The engine creates a `Req` structure and queues it. `Scheduler` selects active requests and forms execution batches.
7. **Model Forward Pass**:
   - `LLMEngine` processes the batch using standard neural network layers.
   - GPU operations execute on `cutile`, `cudarc`, or `Candle`.
8. **Token Sampling**:
   - The engine samples the next token according to `SamplingParams`.
   - Generation continues until a stop condition occurs or max token count is reached.
9. **Response Delivery**:
   - `TokenizerWorker` converts generated tokens back into text.
   - The background worker transmits the completed response through the `oneshot` handle to Axum, which delivers a `ChatCompletionResponse` JSON object to the client.

---

## Build and Deployment Flow

```mermaid
flowchart LR
    subgraph Build ["Build Pipeline"]
        Source["Rust Source"] --> Cargo["cargo build --release"]
        Cargo --> Binary["minisgl binary"]
    end

    subgraph Docker ["Container Deployment"]
        DockerFile["Dockerfile"] --> DockerBuild["docker build"]
        DockerBuild --> Image["minisgl-rust image"]
    end

    subgraph Execution ["Execution Modes"]
        Binary --> RunServer["Server Mode"]
        Binary --> RunShell["Shell Mode"]
        Binary --> RunTP["Tensor Parallel Mode"]
        Image --> RunContainer["Docker Container"]
    end
```

### Build Instructions

The engine executes on CPU by default. You can enable CUDA acceleration with Cargo feature flags on supported NVIDIA hardware.

```bash
# Build release binary (CPU mode by default)
cargo build --release

# Build release binary with CUDA acceleration
cargo build --release --features cuda

# Run OpenAI API Server on CPU
cargo run --release -- --model "Qwen/Qwen3-0.6B" --port 8000

# Run OpenAI API Server with CUDA
cargo run --release --features cuda -- --model "Qwen/Qwen3-0.6B" --port 8000

# Run in Interactive Shell Mode
cargo run --release -- --model "Qwen/Qwen3-0.6B" --shell

# Run with Tensor Parallelism
cargo run --release --features cuda -- --model "Qwen/Qwen3-0.6B" --tp 2
```

### Docker Deployment

```bash
# Build Docker image
docker build -t minisgl-rust .

# Run Docker container
docker run -p 8000:8000 minisgl-rust --model Qwen/Qwen3-0.6B
```

---

## Application Workflows

```mermaid
flowchart TD
    Start["Launch Application"] --> CheckMode{"CLI Flag Check"}
    
    CheckMode -->|"--shell"| Interactive["Interactive Shell Mode"]
    Interactive --> UserInput["Prompt Input"]
    UserInput --> ResetCheck{"Input == '/reset'?"}
    ResetCheck -->|Yes| ClearHistory["Clear Context History"]
    ResetCheck -->|No| EngineInference["Execute Inference"]
    EngineInference --> UserInput
    ClearHistory --> UserInput

    CheckMode -->|"Default Server"| APIServer["Axum API Server"]
    APIServer --> Bind["Bind TCP Listener"]
    Bind --> Listen["Listen for HTTP Requests"]
    Listen --> RouteHealth["GET /health"]
    Listen --> RouteChat["POST /v1/chat/completions"]
```

### Command Line Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--model` | String | `"Qwen/Qwen3-0.6B"` | Model ID or file path |
| `--host` | String | `"127.0.0.1"` | Network address for API server |
| `--port` | u16 | `8000` | Port number for API server |
| `--tp` | usize | `1` | Number of GPUs for Tensor Parallelism |
| `--shell` | bool | `false` | Enable interactive shell mode |

---

## Migration Details

Read [MIGRATION.md](MIGRATION.md) for full mapping between Python and Rust components.
