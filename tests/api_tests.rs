use minisgl::message::protocol::{ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage};

#[test]
fn test_chat_completion_protocol_serde() {
    let req = ChatCompletionRequest {
        model: "Qwen/Qwen3-0.6B".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(64),
        stream: Some(false),
    };

    let serialized = serde_json::to_string(&req).expect("Serialization failed");
    let deserialized: ChatCompletionRequest = serde_json::from_str(&serialized).expect("Deserialization failed");

    assert_eq!(deserialized.model, "Qwen/Qwen3-0.6B");
    assert_eq!(deserialized.messages.len(), 1);
    assert_eq!(deserialized.messages[0].content, "Hello");
}

#[test]
fn test_chat_completion_response() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "Qwen/Qwen3-0.6B".to_string(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
            finish_reason: Some("stop".to_string()),
        }],
    };

    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.content, "Hi there!");
}
