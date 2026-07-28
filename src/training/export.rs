use serde::{Deserialize, Serialize};

/// 版本資訊
pub const GAME_RULE_VERSION: u16 = 1;
pub const AI_VERSION: u32 = 2026072801;

/// 未正規化的 MCTS child（含 raw visits，用於 chosen_move）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisitRecord {
    pub piece: u8,
    pub variant: u8,
    pub x: i8,
    pub y: i8,
    pub visits: u32,
}

/// 正規化後的 policy target（probability = visits / total_visits）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRecord {
    pub piece: u8,
    pub variant: u8,
    pub x: i8,
    pub y: i8,
    pub probability: f32,
}

/// 訓練資料：單一步驟的快照（v3：含 MCTS value + policy target + metadata）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// 棋盤狀態 20×20，0=空格，1~4=玩家編號
    pub board: [[u8; 20]; 20],
    /// 當前玩家編號 (0~3)
    pub current_player: u8,
    /// 玩家人數 (2, 3, 4)
    pub player_count: u8,
    /// 每位玩家剩餘棋子的 bitmask（最多 4 玩家，各 64 bits）
    pub remaining_mask: [u64; 4],
    /// 局數步數
    pub turn: u16,
    /// 最終贏家：255=尚未結束，0~3=玩家編號
    pub winner: u8,

    // MCTS value target（visit-weighted root Q）
    pub mcts_value: f32,
    pub total_visits: u32,

    // Policy target
    pub chosen_move: VisitRecord,
    pub root_visits: Vec<PolicyRecord>,

    // Metadata（資料溯源）
    pub game_rule_version: u16,
    pub ai_version: u32,
    pub random_seed: u64,
}

/// 一場完整的對局紀錄
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub game_id: u64,
    pub player_count: u8,
    pub steps: Vec<StepRecord>,
    pub final_winner: u8, // 255=平手
}

/// 將多場對局寫入二進位檔案
pub fn write_games(path: &str, games: &[GameRecord]) -> Result<(), Box<dyn std::error::Error>> {
    let encoded: Vec<u8> = bincode::serialize(games)?;
    eprintln!("  wrote {} games, {} bytes to {}", games.len(), encoded.len(), path);
    std::fs::write(path, encoded)?;
    Ok(())
}

/// 從二進位檔案讀取對局
pub fn read_games(path: &str) -> Result<Vec<GameRecord>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let games: Vec<GameRecord> = bincode::deserialize(&data)?;
    Ok(games)
}

/// 將 Board<20> 轉為 20×20 u8 陣列
pub fn board_to_array<const N: usize>(board: &crate::game::board::Board<N>) -> [[u8; 20]; 20] {
    let mut arr = [[0u8; 20]; 20];
    for y in 0..N.min(20) {
        for x in 0..N.min(20) {
            arr[y][x] = match board.cells[y][x] {
                crate::game::board::CellState::Empty => 0,
                crate::game::board::CellState::Occupied(pid) => pid.0 as u8 + 1,
            };
        }
    }
    arr
}

/// 將剩餘棋子清單轉為 [u64; 4] bitmask（每玩家一個）
pub fn remaining_to_masks(remaining: &[Vec<usize>], player_count: usize) -> [u64; 4] {
    let mut masks = [0u64; 4];
    for p in 0..player_count.min(4) {
        for &idx in &remaining[p] {
            if idx < 64 {
                masks[p] |= 1u64 << idx;
            }
        }
    }
    masks
}
