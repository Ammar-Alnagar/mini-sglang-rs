use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedInfo {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub node_rank: usize,
    pub node_size: usize,
    pub local_rank: usize,
    pub local_size: usize,
}

impl Default for DistributedInfo {
    fn default() -> Self {
        Self {
            tp_rank: 0,
            tp_size: 1,
            node_rank: 0,
            node_size: 1,
            local_rank: 0,
            local_size: 1,
        }
    }
}

impl DistributedInfo {
    pub fn new(tp_rank: usize, tp_size: usize) -> Self {
        Self {
            tp_rank,
            tp_size,
            node_rank: 0,
            node_size: 1,
            local_rank: tp_rank,
            local_size: tp_size,
        }
    }

    pub fn is_primary(&self) -> bool {
        self.tp_rank == 0
    }
}
