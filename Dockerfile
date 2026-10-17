# Multi-stage Dockerfile for Mini-SGLang Rust

# Builder Stage
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY Cargo.toml ./
COPY vendor/ ./vendor/
COPY src/ ./src/
COPY tests/ ./tests/

RUN cargo build --release

# Runtime Stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1001 minisgl

COPY --from=builder /app/target/release/minisgl /usr/local/bin/minisgl

USER minisgl
WORKDIR /home/minisgl

EXPOSE 8000

ENTRYPOINT ["minisgl"]
CMD ["--help"]
