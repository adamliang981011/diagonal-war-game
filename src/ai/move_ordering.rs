use crate::ai::evaluate;
use crate::game::board::{Board, CellState};
use crate::game::piece::{PieceShape, PieceVariant};
use crate::game::player::PlayerId;

/// 對合法步進行多因子排序（動態權重）
pub fn order_moves<const N: usize>(
    moves: &mut [(usize, usize, i32, i32, i32)],
    board: &Board<N>,
    player: PlayerId,
    pieces: &[PieceShape],
    used_squares: f32,
    total_squares: f32,
) {
    let scores: Vec<i32> = moves.iter().map(|&(pi, vi, x, y, _base)| {
        let variant = &pieces[pi].variants[vi];
        let progress = (used_squares / total_squares).clamp(0.0, 1.0);
        score_move(board, variant, x, y, player, progress)
    }).collect();

    let mut indices: Vec<usize> = (0..moves.len()).collect();
    indices.sort_by(|&a, &b| scores[b].cmp(&scores[a]));
    let sorted: Vec<_> = indices.iter().map(|&i| moves[i].clone()).collect();
    moves.copy_from_slice(&sorted);
}

/// 單一棋步的綜合評分（使用動態權重）
fn score_move<const N: usize>(
    board: &Board<N>,
    variant: &PieceVariant,
    x: i32,
    y: i32,
    player: PlayerId,
    progress: f32,
) -> i32 {
    let size_w = (35.0 + progress * 25.0) as i32;
    let corner_w = (35.0 - progress * 25.0) as i32;
    let block_w = (2.0 - progress) as i32;
    let center_w = (5.0 * (1.0 - progress)) as i32;

    let mut score = 0;
    score += variant.cells.len() as i32 * size_w;
    score += evaluate::count_corner_contacts(board, variant, x, y, player) * corner_w;

    for &(dx, dy) in &variant.cells {
        let dist = ((x + dx - (N as i32 / 2)).abs() + (y + dy - (N as i32 / 2)).abs()).max(1);
        score += (40 - dist.min(39)) * center_w;
    }

    let mut empty = 0;
    for &(dx, dy) in &variant.cells {
        let ax = x + dx; let ay = y + dy;
        for (nx, ny) in &[(ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)] {
            if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty { empty += 1; }
        }
    }
    score += empty * block_w;
    score
}
