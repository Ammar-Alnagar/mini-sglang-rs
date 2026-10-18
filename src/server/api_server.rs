use crate::message::protocol::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
};
use axum::{
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;

pub async fn start_server(host: &str, port: u16) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/chat/completions", post(chat_completions));

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server running on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> Json<ChatCompletionResponse> {
    let response = ChatCompletionResponse {
        id: "chatcmpl-minisgl-rust".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello from Mini-SGLang pure Rust engine!".to_string(),
            },
            finish_reason: Some("stop".to_string()),
        }],
    };
    Json(response)
}
