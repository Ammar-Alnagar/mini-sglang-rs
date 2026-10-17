use crate::core::req::{Batch, BatchPhase, Req};

pub struct Scheduler {
    pub max_running_reqs: usize,
}

impl Scheduler {
    pub fn new(max_running_reqs: usize) -> Self {
        Self { max_running_reqs }
    }

    pub fn schedule(&self, reqs: &mut [Req]) -> Option<Batch> {
        let active: Vec<_> = reqs.iter().filter(|r| r.can_decode()).cloned().collect();
        if active.is_empty() {
            None
        } else {
            Some(Batch::new(active, BatchPhase::Decode))
        }
    }
}
