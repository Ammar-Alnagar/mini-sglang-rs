use minisgl::core::sampling::SamplingParams;

#[test]
fn test_default_sampling_params() {
    let params = SamplingParams::default();
    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_k, -1);
    assert_eq!(params.top_p, 1.0);
    assert!(params.is_greedy());
}

#[test]
fn test_non_greedy_sampling_params() {
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        ignore_eos: false,
        max_tokens: 128,
    };
    assert!(!params.is_greedy());
}
