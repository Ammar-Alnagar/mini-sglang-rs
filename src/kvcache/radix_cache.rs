use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RadixNode {
    pub key: Vec<u32>,
    pub value_page_idx: usize,
    pub children: HashMap<u32, usize>,
    pub parent: Option<usize>,
    pub ref_count: usize,
}

pub struct RadixCache {
    pub nodes: Vec<RadixNode>,
    pub root_idx: usize,
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixCache {
    pub fn new() -> Self {
        let root = RadixNode {
            key: Vec::new(),
            value_page_idx: 0,
            children: HashMap::new(),
            parent: None,
            ref_count: 1,
        };
        Self {
            nodes: vec![root],
            root_idx: 0,
        }
    }

    pub fn match_prefix(&self, tokens: &[u32]) -> (usize, Vec<usize>) {
        let mut curr = self.root_idx;
        let mut matched_len = 0;
        let mut page_indices = Vec::new();

        while matched_len < tokens.len() {
            let next_token = tokens[matched_len];
            if let Some(&child_idx) = self.nodes[curr].children.get(&next_token) {
                let child = &self.nodes[child_idx];
                if tokens[matched_len..].starts_with(&child.key) {
                    matched_len += child.key.len();
                    page_indices.push(child.value_page_idx);
                    curr = child_idx;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        (matched_len, page_indices)
    }

    pub fn insert(&mut self, tokens: &[u32], page_idx: usize) -> usize {
        let new_node = RadixNode {
            key: tokens.to_vec(),
            value_page_idx: page_idx,
            children: HashMap::new(),
            parent: Some(self.root_idx),
            ref_count: 1,
        };
        let idx = self.nodes.len();
        self.nodes.push(new_node);
        if let Some(first_token) = tokens.first() {
            self.nodes[self.root_idx].children.insert(*first_token, idx);
        }
        idx
    }
}
