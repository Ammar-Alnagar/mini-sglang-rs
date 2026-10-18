use candle_core::{Device, Tensor};
use minisgl::kernel::{
    greedy_sample, index_select, index_select_fallback, radix_index, radix_store,
    sample_logits, sample_top_p_top_k, store_kv_cache, store_kv_fallback,
};
use minisgl::layers::norm::RMSNorm;
use minisgl::layers::rotary::RotaryEmbedding;
use minisgl::moe::fused::FusedMoE;

#[test]
fn test_index_select_kernel() {
    let device = Device::Cpu;
    let weight = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).unwrap();
    let indices = Tensor::new(&[2u32, 0u32], &device).unwrap();

    let out_fallback = index_select_fallback(&weight, &indices).unwrap();
    let out_dispatch = index_select(&weight, &indices).unwrap();

    let vec_fallback = out_fallback.to_vec2::<f32>().unwrap();
    let vec_dispatch = out_dispatch.to_vec2::<f32>().unwrap();

    assert_eq!(vec_fallback, vec![vec![5.0, 6.0], vec![1.0, 2.0]]);
    assert_eq!(vec_dispatch, vec_fallback);
}

#[test]
fn test_store_kv_cache_kernel() {
    let device = Device::Cpu;
    let mut k_cache = Tensor::zeros((4, 2), candle_core::DType::F32, &device).unwrap();
    let mut v_cache = Tensor::zeros((4, 2), candle_core::DType::F32, &device).unwrap();
    let indices = Tensor::new(&[1u32, 3u32], &device).unwrap();
    let k = Tensor::new(&[[1.0f32, 1.0], [2.0, 2.0]], &device).unwrap();
    let v = Tensor::new(&[[3.0f32, 3.0], [4.0, 4.0]], &device).unwrap();

    assert!(store_kv_fallback(&mut k_cache, &mut v_cache, &indices, &k, &v).is_ok());
    assert!(store_kv_cache(&mut k_cache, &mut v_cache, &indices, &k, &v).is_ok());
}

#[test]
fn test_sampling_kernels() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[1.0f32, 10.0, 2.0], [5.0, 0.0, 1.0]], &device).unwrap();

    let greedy = greedy_sample(&logits).unwrap();
    assert_eq!(greedy, vec![1, 0]);

    let sampled = sample_logits(&logits, 0.0, 1.0, -1).unwrap();
    assert_eq!(sampled, vec![1, 0]);

    let sampled_top_p = sample_top_p_top_k(&logits, 0.7, 0.9, 10).unwrap();
    assert_eq!(sampled_top_p.len(), 2);
}

#[test]
fn test_radix_kernels() {
    let device = Device::Cpu;
    let req_to_token = Tensor::new(&[10u32, 20u32, 30u32], &device).unwrap();
    let req_lens = Tensor::new(&[1u32, 1u32, 1u32], &device).unwrap();

    let out = radix_index(&req_to_token, &req_lens).unwrap();
    assert_eq!(out.to_vec1::<u32>().unwrap(), vec![10, 20, 30]);

    let src = Tensor::new(&[1.0f32, 2.0], &device).unwrap();
    let mut dst = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
    assert!(radix_store(&src, &mut dst).is_ok());
}

#[test]
fn test_layers_and_moe_kernels() {
    let device = Device::Cpu;
    let weight = Tensor::new(&[1.0f32, 1.0f32], &device).unwrap();
    let norm = RMSNorm::new(weight, 1e-5);
    let x = Tensor::new(&[[2.0f32, 4.0f32]], &device).unwrap();
    let norm_out = norm.forward(&x).unwrap();
    assert_eq!(norm_out.shape().dims(), &[1, 2]);

    let cos = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
    let sin = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
    let rope = RotaryEmbedding::new(cos, sin);
    let rope_out = rope.forward(&x).unwrap();
    assert_eq!(rope_out.shape().dims(), &[1, 2]);

    let moe = FusedMoE::new(8, 2);
    let moe_out = moe.forward(&x).unwrap();
    assert_eq!(moe_out.shape().dims(), &[1, 2]);
}
