use crate::core::req::Req;
use crate::engine::config::EngineConfig;
use crate::kvcache::radix_cache::RadixCache;
use crate::llm::engine::LLMEngine;
use crate::message::protocol::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
};
use crate::scheduler::scheduler::Scheduler;
use crate::tokenizer::worker::TokenizerWorker;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{mpsc, oneshot};

static REQ_ID_GEN: AtomicU64 = AtomicU64::new(1);

struct ServerRequest {
    req: ChatCompletionRequest,
    tx: oneshot::Sender<ChatCompletionResponse>,
}

#[derive(Clone)]
pub struct AppState {
    tx: mpsc::Sender<ServerRequest>,
}

pub async fn start_server(host: &str, port: u16, model_name_or_path: &str, cfg: EngineConfig) -> anyhow::Result<()> {
    let (tx, rx) = mpsc::channel::<ServerRequest>(1024);
    let state = AppState { tx };
    let model_str = model_name_or_path.to_string();

    // Spawn background engine worker loop for batching and execution
    tokio::spawn(async move {
        if let Err(e) = run_engine_loop(rx, &model_str, cfg).await {
            tracing::error!("Engine loop error: {:?}", e);
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Optimized server running on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    let (tx, rx) = oneshot::channel();
    let server_req = ServerRequest { req, tx };

    if state.tx.send(server_req).await.is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match rx.await {
        Ok(resp) => Ok(Json(resp)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn run_engine_loop(
    mut rx: mpsc::Receiver<ServerRequest>,
    model_name_or_path: &str,
    cfg: EngineConfig,
) -> anyhow::Result<()> {
    let tokenizer = TokenizerWorker::new(model_name_or_path);
    let cache = RadixCache::new();
    let scheduler = Scheduler::new(32);
    let mut engine = LLMEngine::new(cfg)?;

    tracing::info!("Loading model {} into engine...", model_name_or_path);
    if let Err(e) = engine.load_model(model_name_or_path) {
        tracing::warn!(
            "Failed to load weights for {}: {}. Running in fallback mode.",
            model_name_or_path,
            e
        );
    }

    let mut pending_handlers: Vec<(u64, String, oneshot::Sender<ChatCompletionResponse>)> =
        Vec::new();
    let mut active_reqs: Vec<Req> = Vec::new();

    loop {
        // Collect batch of incoming requests from channel non-blockingly if active requests exist, or block if empty
        let mut new_incoming = Vec::new();

        if active_reqs.is_empty() {
            if let Some(req_item) = rx.recv().await {
                new_incoming.push(req_item);
            } else {
                break; // Channel closed
            }
        }

        while let Ok(req_item) = rx.try_recv() {
            new_incoming.push(req_item);
            if new_incoming.len() >= 32 {
                break;
            }
        }

        // Process incoming requests into active_reqs
        for item in new_incoming {
            let uid = REQ_ID_GEN.fetch_add(1, Ordering::SeqCst);
            let prompt = if item.req.messages.len() == 1 && item.req.messages[0].role == "user" {
                tokenizer.format_chat_prompt(&[(
                    "user".to_string(),
                    item.req.messages[0].content.clone(),
                )])
            } else {
                let msgs: Vec<(String, String)> = item
                    .req
                    .messages
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone()))
                    .collect();
                tokenizer.format_chat_prompt(&msgs)
            };

            let token_ids = tokenizer.encode(&prompt).unwrap_or_default();
            let (cached_len, page_indices) = cache.match_prefix(&token_ids);
            let table_idx = page_indices.first().copied().unwrap_or(0);
            let max_tokens = item.req.sampling_params.max_tokens;

            let req_obj = Req::new(
                uid,
                token_ids,
                table_idx,
                cached_len,
                max_tokens,
                item.req.sampling_params.clone(),
            );

            active_reqs.push(req_obj);
            pending_handlers.push((uid, item.req.model, item.tx));
        }

        if active_reqs.is_empty() {
            continue;
        }

        // Schedule and run batch execution step
        if let Some(_batch) = scheduler.schedule(&mut active_reqs) {
            let _tokens = engine.step(&mut active_reqs).await?;

            // Check finished requests and deliver responses
            let mut remaining_reqs = Vec::new();
            for req_item in active_reqs {
                if req_item.is_finished || !req_item.can_decode() {
                    let mut decoded_text = tokenizer
                        .decode(&req_item.generated_tokens)
                        .unwrap_or_default();

                    // Strip any trailing assistant tokens / tags if present
                    if let Some(stripped) = decoded_text.strip_suffix("<|im_end|>") {
                        decoded_text = stripped.to_string();
                    } else if let Some(stripped) = decoded_text.strip_suffix("<|endoftext|>") {
                        decoded_text = stripped.to_string();
                    }

                    if let Some(pos) = pending_handlers
                        .iter()
                        .position(|(id, _, _)| *id == req_item.uid)
                    {
                        let (_, model_name, tx) = pending_handlers.remove(pos);
                        let resp = ChatCompletionResponse {
                            id: format!("chatcmpl-{}", req_item.uid),
                            object: "chat.completion".to_string(),
                            created: 1700000000,
                            model: model_name,
                            choices: vec![ChatCompletionChoice {
                                index: 0,
                                message: ChatMessage {
                                    role: "assistant".to_string(),
                                    content: decoded_text,
                                },
                                finish_reason: Some("stop".to_string()),
                            }],
                        };
                        let _ = tx.send(resp);
                    }
                } else {
                    remaining_reqs.push(req_item);
                }
            }
            active_reqs = remaining_reqs;
        }

        tokio::task::yield_now().await;
    }

    Ok(())
}
