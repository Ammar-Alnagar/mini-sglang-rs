use minisgl::core::req::Req;
use minisgl::core::sampling::SamplingParams;
use minisgl::kvcache::radix_cache::RadixCache;

#[test]
fn test_req_lifecycle() {
    let params = SamplingParams::default();
    let mut req = Req::new(1, vec![1, 2, 3], 0, 0, 10, params);
    assert_eq!(req.remain_len(), 10);
    assert!(req.can_decode());

    req.append_token(42);
    req.complete_one();
    assert_eq!(req.generated_tokens, vec![42]);
}

#[test]
fn test_radix_cache() {
    let mut cache = RadixCache::new();
    let page = cache.insert(&[10, 20, 30], 1);
    assert_eq!(page, 1);

    let (matched, pages) = cache.match_prefix(&[10, 20, 30, 40]);
    assert_eq!(matched, 3);
    assert_eq!(pages, vec![1]);
}
