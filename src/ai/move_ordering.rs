use crate::ai::evaluate;
use crate::ai::opening;
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

/// 形狀別棋值表（依進度變化）
pub fn piece_value_table(progress: f32, cells: &[(i32, i32)]) -> f32 {
    let n = cells.len();
    let w = cells.iter().map(|&(x, _)| x).max().unwrap_or(0)
             - cells.iter().map(|&(x, _)| x).min().unwrap_or(0) + 1;
    let h = cells.iter().map(|&(_, y)| y).max().unwrap_or(0)
             - cells.iter().map(|&(_, y)| y).min().unwrap_or(0) + 1;
    let spread = (w * h) as f32 / n as f32;

    let base = match n {
        7 => 120.0, 6 => 100.0, 5 => 80.0,
        4 => 60.0, 3 => 40.0, 2 => 20.0, _ => 10.0,
    };
    let shape_bonus = if spread > 2.0 { 20.0 } else { 0.0 };

    let decay = (progress * 3.0).min(1.0);
    base * (1.0 - decay) + (base * 0.3 + shape_bonus * 0.5) * decay
}

/// Action encoding constants (must match python/dataset_loader.py)
pub const MAX_PIECES: usize = 26;
pub const MAX_VARIANTS: usize = 8;
pub const BOARD_SIZE: usize = 20;
pub const MAX_ACTIONS: usize = MAX_PIECES * MAX_VARIANTS * BOARD_SIZE * BOARD_SIZE;

/// 將 (piece, variant, x, y) 編碼為 action_id (0..70400)
pub fn action_id(pi: usize, vi: usize, x: i32, y: i32) -> usize {
    pi * MAX_VARIANTS * BOARD_SIZE * BOARD_SIZE
        + vi * BOARD_SIZE * BOARD_SIZE
        + (y as usize) * BOARD_SIZE
        + (x as usize)
}

/// 計算先驗機率 prior（固定公式，獨立於 score_move）
pub fn compute_prior<const N: usize>(
    board: &Board<N>,
    variant: &PieceVariant,
    x: i32, y: i32,
    player: PlayerId,
    progress: f32,
) -> f32 {
    let size_val = piece_value_table(progress, &variant.cells);
    let heat_val = opening::centroid_heat(variant, x, y) * 25.0;
    let corner_val = evaluate::count_corner_contacts(board, variant, x, y, player) as f32 * 5.0;
    0.45 * size_val + 0.25 * heat_val + 0.20 * corner_val + 0.10 * 0.0
}

/// 單一棋步的綜合評分（使用動態權重，僅用於 order_moves 排序）
pub fn score_move<const N: usize>(
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
