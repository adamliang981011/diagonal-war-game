use std::collections::HashMap;

use crate::ai::AiMove;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Debug, Clone)]
pub struct TTEntry {
    pub depth: u8,
    pub score: f32,
    pub flag: TTFlag,
    pub best_move: Option<AiMove>,
}

/// Transposition Table：以盤面 hash 為 key 快取搜尋結果
pub struct TranspositionTable {
    entries: HashMap<u64, TTEntry>,
    max_size: usize,
}

impl TranspositionTable {
    pub fn new(max_size: usize) -> Self {
        Self { entries: HashMap::with_capacity(max_size / 2), max_size }
    }

    pub fn lookup(&self, hash: u64) -> Option<&TTEntry> {
        self.entries.get(&hash)
    }

    pub fn insert(&mut self, hash: u64, entry: TTEntry) {
        if self.entries.len() >= self.max_size {
            // 超過上限時清空（簡單策略）
            self.entries.clear();
        }
        self.entries.insert(hash, entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
