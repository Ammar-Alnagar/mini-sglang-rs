use candle_core::{Result, Tensor};
use rand::Rng;

pub const SAMPLING_KERNEL_CU: &str = r#"
extern "C" __global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out,
    int vocab_size,
    int batch_size
) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = -1e30f;
        int max_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            float val = logits[b * vocab_size + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        out[b] = max_idx;
    }
}
"#;

pub fn greedy_sample(logits: &Tensor) -> Result<Vec<u32>> {
    let argmax = logits.argmax(candle_core::D::Minus1)?;
    let vec = argmax.to_vec1::<u32>()?;
    Ok(vec)
}

pub fn sample_top_p_top_k(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: i32,
) -> Result<Vec<u32>> {
    if temperature <= 0.0 || top_k == 1 {
        return greedy_sample(logits);
    }

    let shape = logits.shape();
    let batch_size = shape.dim(0)?;
    let vocab_size = shape.dim(1)?;
    let logits_vec = logits.to_vec2::<f32>()?;
    let mut results = Vec::with_capacity(batch_size);
    let mut rng = rand::rng();

    for b in 0..batch_size {
        let mut row: Vec<(usize, f32)> = logits_vec[b]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v / temperature))
            .collect();

        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if top_k > 0 && (top_k as usize) < vocab_size {
            row.truncate(top_k as usize);
        }

        let max_logit = row.first().map(|x| x.1).unwrap_or(0.0);
        let mut sum_exp = 0.0f32;
        let mut exps: Vec<f32> = row.iter().map(|x| {
            let e = (x.1 - max_logit).exp();
            sum_exp += e;
            e
        }).collect();

        if sum_exp > 0.0 {
            for e in exps.iter_mut() {
                *e /= sum_exp;
            }
        }

        if top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = exps.len();
            for (i, &p) in exps.iter().enumerate() {
                cumsum += p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            exps.truncate(cutoff);
            row.truncate(cutoff);

            let new_sum: f32 = exps.iter().sum();
            if new_sum > 0.0 {
                for e in exps.iter_mut() {
                    *e /= new_sum;
                }
            }
        }

        let r: f32 = rng.random();
        let mut acc = 0.0f32;
        let mut selected = row[0].0;
        for (i, &p) in exps.iter().enumerate() {
            acc += p;
            if r <= acc {
                selected = row[i].0;
                break;
            }
        }
        results.push(selected as u32);
    }

    Ok(results)
}
