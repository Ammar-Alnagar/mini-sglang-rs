use crate::core::sampling::SamplingParams;

#[derive(Debug, Clone)]
pub struct Req {
    pub uid: u64,
    pub input_ids: Vec<u32>,
    pub table_idx: usize,
    pub cached_len: usize,
    pub device_len: usize,
    pub output_len: usize,
    pub max_device_len: usize,
    pub sampling_params: SamplingParams,
    pub generated_tokens: Vec<u32>,
    pub is_finished: bool,
}

impl Req {
    pub fn new(
        uid: u64,
        input_ids: Vec<u32>,
        table_idx: usize,
        cached_len: usize,
        output_len: usize,
        sampling_params: SamplingParams,
    ) -> Self {
        let device_len = input_ids.len();
        let max_device_len = device_len + output_len;
        assert!(cached_len <= device_len);
        assert!(device_len <= max_device_len);

        Self {
            uid,
            input_ids,
            table_idx,
            cached_len,
            device_len,
            output_len,
            max_device_len,
            sampling_params,
            generated_tokens: Vec::new(),
            is_finished: false,
        }
    }

    pub fn remain_len(&self) -> usize {
        self.max_device_len.saturating_sub(self.device_len)
    }

    pub fn extend_len(&self) -> usize {
        self.device_len.saturating_sub(self.cached_len)
    }

    pub fn complete_one(&mut self) {
        self.cached_len = self.device_len;
        self.device_len += 1;
    }

    pub fn append_token(&mut self, token: u32) {
        self.input_ids.push(token);
        self.generated_tokens.push(token);
    }

    pub fn can_decode(&self) -> bool {
        self.remain_len() > 0 && !self.is_finished
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub reqs: Vec<Req>,
    pub phase: BatchPhase,
}

impl Batch {
    pub fn new(reqs: Vec<Req>, phase: BatchPhase) -> Self {
        Self { reqs, phase }
    }

    pub fn is_prefill(&self) -> bool {
        self.phase == BatchPhase::Prefill
    }

    pub fn is_decode(&self) -> bool {
        self.phase == BatchPhase::Decode
    }

    pub fn size(&self) -> usize {
        self.reqs.len()
    }
}
