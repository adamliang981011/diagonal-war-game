pub mod config;
pub mod elo;
pub mod endgame;
pub mod evaluate;
pub mod greedy;
pub mod mcts;
pub mod move_ordering;
pub mod opening;
pub mod opening_book;
pub mod random;
pub mod search;
pub mod tournament;
pub mod train;
pub mod transposition;
pub mod value;
pub mod zobrist;

use std::sync::Mutex;

use crate::game::board::{Board, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// Tree Reuse 的跨回合搜尋狀態
pub struct SearchState {
    pub tree: Option<Box<crate::ai::mcts::Tree>>,
}

static GLOBAL_SEARCH: Mutex<Option<SearchState>> = Mutex::new(None);

/// 存取全域 SearchState（首次使用時自動初始化）
pub fn with_search_state<F, R>(f: F) -> R
where
    F: FnOnce(&mut Option<SearchState>) -> R,
{
    let mut guard = GLOBAL_SEARCH.lock().unwrap();
    if guard.is_none() {
        *guard = Some(SearchState { tree: None });
    }
    f(&mut *guard)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AiDifficulty {
    Random,
    Greedy,
    GreedyWithTemp(f32),
    Search1Ply,
    Search2Ply,
    Mcts { iterations: usize },
}

#[derive(Debug, Clone)]
pub struct AiMove {
    pub piece_index: usize,
    pub variant_index: usize,
    pub x: i32,
    pub y: i32,
    pub score: i32,
}

/// MCTS 完整輸出（含統計資訊，供 self-play 訓練資料使用）
#[derive(Debug, Clone)]
pub struct MctsOutput {
    pub best: AiMove,
    /// (encoded_action, visits) — 被訪問過的 root children
    pub visits: Vec<(u32, u32)>,
    pub total_visits: u32,
    /// Visit-weighted root value
    pub value: f32,
}

/// 將 (piece, variant, x, y) 編碼為單一 u32 action id
pub fn encode_action(piece: usize, variant: usize, x: i32, y: i32) -> u32 {
    (piece as u32 & 0x1F)
        | ((variant as u32 & 0x1F) << 5)
        | ((x as u32 & 0x1F) << 10)
        | ((y as u32 & 0x1F) << 15)
}

pub fn decode_action(a: u32) -> (usize, usize, i32, i32) {
    let piece = (a & 0x1F) as usize;
    let variant = ((a >> 5) & 0x1F) as usize;
    let x = ((a >> 10) & 0x1F) as i32;
    let y = ((a >> 15) & 0x1F) as i32;
    (piece, variant, x, y)
}

/// 列出所有合法放置
pub fn list_legal_moves<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> Vec<(usize, usize, i32, i32, i32)> {
    let mut moves = Vec::new();
    for (pi, shape) in remaining_pieces.iter().enumerate() {
        for (vi, variant) in shape.variants.iter().enumerate() {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    if x + variant.width > N as i32 || y + variant.height > N as i32 {
                        continue;
                    }
                    if board
                        .is_valid(variant, x, y, player, is_first_move, starting_corner)
                        .is_ok()
                    {
                        let base = variant.cells.len() as i32 * 100
                            + evaluate::count_corner_contacts(board, variant, x, y, player);
                        moves.push((pi, vi, x, y, base));
                    }
                }
            }
        }
    }
    moves
}

/// 選擇一步並回傳 MCTS 統計資訊（供 self-play 使用）
pub fn choose_move_with_stats<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    difficulty: AiDifficulty,
    player_count: usize,
) -> Option<(AiMove, MctsOutput)> {
    match difficulty {
        AiDifficulty::Mcts { iterations } => {
            let mut cfg = crate::ai::config::official_config();
            cfg.iterations = iterations;
            let mut stats = None;
            let mv = mcts::choose_move(board, player, remaining_pieces, is_first_move, starting_corner, &cfg, player_count, &mut stats, &mut None);
            mv.map(|m| (m, stats.unwrap()))
        },
        _ => choose_move(board, player, remaining_pieces, is_first_move, starting_corner, difficulty, player_count)
            .map(|m| {
                let out = MctsOutput { best: m.clone(), visits: vec![], total_visits: 0, value: 0.5 };
                (m, out)
            }),
    }
}

/// 根據難度選擇 AI 策略
pub fn choose_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    difficulty: AiDifficulty,
    player_count: usize,
) -> Option<AiMove> {
    // Endgame detection: 剩餘總格數 ≤ 25 時使用 Alpha-Beta 搜尋
    let total_remaining: usize = remaining_pieces.iter().map(|p| p.base.cells.len()).sum();
    if total_remaining <= 25 && remaining_pieces.len() <= 5 {
        let all_players: Vec<PlayerId> = (0..player_count).map(PlayerId).collect();
        if let Some(mv) = endgame::solve(board, player, remaining_pieces, &all_players) {
            return Some(mv);
        }
    }

    match difficulty {
        AiDifficulty::Random => random::choose_move(board, player, remaining_pieces, is_first_move, starting_corner),
        AiDifficulty::Greedy => greedy::choose_move(board, player, remaining_pieces, is_first_move, starting_corner),
        AiDifficulty::GreedyWithTemp(temp) => greedy::choose_move_with_temp(board, player, remaining_pieces, is_first_move, starting_corner, temp),
        AiDifficulty::Search1Ply => search::choose_move_with_temp(board, player, remaining_pieces, is_first_move, starting_corner, 1, 0.2, player_count),
        AiDifficulty::Search2Ply => search::choose_move_with_temp(board, player, remaining_pieces, is_first_move, starting_corner, 2, 0.2, player_count),
        AiDifficulty::Mcts { iterations } => {
            let mut cfg = crate::ai::config::official_config();
            cfg.iterations = iterations;
            with_search_state(|search_state| {
                mcts::choose_move(board, player, remaining_pieces, is_first_move, starting_corner, &cfg, player_count, &mut None, search_state)
            })
        },
    }
}
