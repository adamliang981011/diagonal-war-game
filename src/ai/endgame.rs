use crate::ai::list_legal_moves;
use crate::ai::AiMove;
use crate::game::board::{Board, CellState, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// Endgame Search：剩餘格數 ≤ 25 時使用 DFS + Alpha-Beta 窮舉
/// 由於搜尋空間有限，可找到絕對最佳步
pub fn solve<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    all_players: &[PlayerId],
) -> Option<AiMove> {
    let mut best_move: Option<AiMove> = None;
    let mut best_score = f32::MIN;

    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));
    let corner = match player.0 {
        0 => Corner::TopLeft, 1 => Corner::TopRight,
        2 => Corner::BottomRight, _ => Corner::BottomLeft,
    };

    let moves = list_legal_moves(board, player, remaining_pieces, is_first, Some(corner));
    if moves.is_empty() {
        return None;
    }

    for &(pi, vi, x, y, _) in &moves {
        let variant = &remaining_pieces[pi].variants[vi];
        let mut sim_board = board.clone();
        sim_board.place_piece(variant, x, y, player);

        // Alpha-Beta 評估對方回應
        let score = -negamax(&sim_board, PlayerId((player.0 + 1) % all_players.len() as usize),
                             remaining_pieces, all_players,
                             f32::MIN, f32::MAX, 3);

        if score > best_score {
            best_score = score;
            best_move = Some(AiMove {
                piece_index: pi, variant_index: vi, x, y,
                score: (score * 1000.0) as i32,
            });
        }
    }

    best_move
}

/// Negamax + Alpha-Beta Pruning
/// depth: 剩餘搜尋深度
fn negamax<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    all_pieces: &[PieceShape],
    all_players: &[PlayerId],
    mut alpha: f32,
    beta: f32,
    depth: usize,
) -> f32 {
    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));
    let corner = match player.0 {
        0 => Corner::TopLeft, 1 => Corner::TopRight,
        2 => Corner::BottomRight, _ => Corner::BottomLeft,
    };

    let moves = list_legal_moves(board, player, all_pieces, is_first, Some(corner));

    if moves.is_empty() || depth == 0 {
        // 終止搜尋：使用 heuristic 評估
        let occupied = board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;
        let weights = crate::ai::evaluate::compute_phase_weights(occupied, crate::ai::evaluate::TOTAL_PIECE_AREA);
        return crate::ai::evaluate::heuristic_evaluate_with_weights(board, player, all_players, &weights);
    }

    for &(pi, vi, x, y, _) in &moves {
        let variant = &all_pieces[pi].variants[vi];
        let mut sim_board = board.clone();
        sim_board.place_piece(variant, x, y, player);

        let score = -negamax(&sim_board, PlayerId((player.0 + 1) % all_players.len() as usize),
                             all_pieces, all_players,
                             -beta, -alpha, depth - 1);

        if score >= beta {
            return beta; // prune (fail-soft)
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}
