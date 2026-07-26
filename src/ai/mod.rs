pub mod elo;
pub mod endgame;
pub mod evaluate;
pub mod greedy;
pub mod mcts;
pub mod move_ordering;
pub mod opening_book;
pub mod random;
pub mod search;
pub mod train;
pub mod transposition;
pub mod zobrist;

use crate::game::board::{Board, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

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
        AiDifficulty::Mcts { iterations } => mcts::choose_move(board, player, remaining_pieces, is_first_move, starting_corner, iterations, player_count),
    }
}
