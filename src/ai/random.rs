use crate::ai::list_legal_moves;
use crate::ai::AiMove;
use crate::game::board::{Board, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// 隨機 AI：從所有合法放置中隨機挑選
pub fn choose_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> Option<AiMove> {
    use rand::prelude::IndexedRandom;

    let moves = list_legal_moves(board, player, remaining_pieces, is_first_move, starting_corner);
    let mut rng = rand::rng();
    moves.choose(&mut rng).map(|&(pi, vi, x, y, score)| AiMove {
        piece_index: pi,
        variant_index: vi,
        x,
        y,
        score,
    })
}
