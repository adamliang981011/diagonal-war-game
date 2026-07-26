use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 開局書中的一個 entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningEntry {
    pub visits: u32,
    pub score: f32,          // 平均得分（0.0 ~ 1.0）
    pub best_piece: usize,
    pub best_variant: usize,
    pub best_x: i32,
    pub best_y: i32,
}

/// 開局書：盤面 hash → 統計資料
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeningBook {
    pub entries: HashMap<u64, OpeningEntry>,
}

impl OpeningBook {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// 查詢盤面是否有已知的最佳步
    pub fn lookup(&self, board_hash: u64) -> Option<&OpeningEntry> {
        self.entries.get(&board_hash)
    }

    /// 插入或更新一個 entry
    pub fn insert(&mut self, board_hash: u64, entry: OpeningEntry) {
        self.entries.insert(board_hash, entry);
    }

    /// 從 JSON 檔案載入
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// 儲存至 JSON 檔案
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// entry 數量
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
