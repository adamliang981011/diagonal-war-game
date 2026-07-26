use std::collections::VecDeque;

use crate::game::board::{Board, CellState, Corner};
use crate::game::piece::{PieceShape, PieceVariant};
use crate::game::player::PlayerId;

/// 盤面階段權重
#[derive(Debug, Clone, Copy)]
pub struct GamePhaseWeights {
    pub mobility: f32,
    pub corner: f32,
    pub blocking: f32,
    pub center: f32,
    pub future_mobility: f32,
    pub frontier_quality: f32,
    pub dead_region: f32,
}

pub fn compute_phase_weights(used_squares: f32, total_squares: f32) -> GamePhaseWeights {
    let progress = (used_squares / total_squares).clamp(0.0, 1.0);
    GamePhaseWeights {
        mobility: 5.0 + progress * 3.0,
        corner:   5.0 - progress * 3.0,
        blocking: 1.0 + (progress * std::f32::consts::PI).sin() * 2.5,
        center:   3.0 * (1.0 - progress),
        future_mobility: 1.0 + progress * 1.5,
        frontier_quality: 1.0 + (progress * std::f32::consts::PI).sin() * 1.5,
        dead_region: progress * 1.5,
    }
}

pub fn opening_weights() -> GamePhaseWeights {
    GamePhaseWeights {
        mobility: 5.0, corner: 5.0, blocking: 1.0, center: 3.0,
        future_mobility: 1.0, frontier_quality: 1.0, dead_region: 0.0,
    }
}

// ============================================================
// 棋子放置評分（用於 greedy/placement）
// ============================================================

/// 計算放置後與自己棋子的角接觸數量
pub fn count_corner_contacts<const N: usize>(
    board: &Board<N>,
    variant: &PieceVariant,
    pos_x: i32,
    pos_y: i32,
    player: PlayerId,
) -> i32 {
    let mut count = 0;
    for &(dx, dy) in &variant.cells {
        let ax = pos_x + dx;
        let ay = pos_y + dy;
        for (nx, ny) in &[(ax + 1, ay + 1), (ax + 1, ay - 1), (ax - 1, ay + 1), (ax - 1, ay - 1)] {
            if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Occupied(player) {
                count += 1;
            }
        }
    }
    count
}

/// 基礎評分：棋子大小 × 100 + 角接觸數 × 5
pub fn score_placement<const N: usize>(
    board: &Board<N>,
    variant: &PieceVariant,
    x: i32,
    y: i32,
    player: PlayerId,
) -> i32 {
    let size_score = variant.cells.len() as i32 * 100;
    let contact_score = count_corner_contacts(board, variant, x, y, player) * 5;
    size_score + contact_score
}

/// Temperature 取樣
pub fn temperature_sample(scores: &[i32], temp: f32, rng: &mut impl rand::Rng) -> usize {
    if scores.is_empty() { return 0; }
    if temp <= 0.0 {
        scores.iter().enumerate().max_by_key(|(_, s)| **s).unwrap().0
    } else {
        let max = *scores.iter().max().unwrap_or(&0) as f32;
        let weights: Vec<f32> = scores.iter().map(|s| ((*s as f32 - max) / temp).exp()).collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 { return scores.len() - 1; }
        let mut roll: f32 = rng.random::<f32>() * total;
        for (i, w) in weights.iter().enumerate() {
            roll -= w; if roll <= 0.0 { return i; }
        }
        scores.len() - 1
    }
}

// ============================================================
// 進階放置評分
// ============================================================

fn center_distance<const N: usize>(x: i32, y: i32) -> i32 {
    let cx = (N as i32) / 2; let cy = (N as i32) / 2;
    let dist = ((x - cx).abs() + (y - cy).abs()).max(1);
    20 - dist.min(19)
}

pub fn score_placement_advanced<const N: usize>(
    board: &Board<N>, variant: &PieceVariant, x: i32, y: i32,
    player: PlayerId, opponent_ids: &[PlayerId],
) -> i32 {
    let mut score = score_placement(board, variant, x, y, player);
    for &(dx, dy) in &variant.cells { score += center_distance::<N>(x + dx, y + dy) * 3; }
    for &_opp in opponent_ids {
        let before = empty_neighbors::<N>(board, x, y, variant);
        if before > 0 { score += 2; }
    }
    score
}

fn empty_neighbors<const N: usize>(
    board: &Board<N>, pos_x: i32, pos_y: i32, variant: &PieceVariant,
) -> i32 {
    let mut count = 0;
    for &(dx, dy) in &variant.cells {
        let ax = pos_x + dx; let ay = pos_y + dy;
        for (nx, ny) in &[(ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)] {
            if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty { count += 1; }
        }
    }
    count
}

// ============================================================
// 盤面整體評估（MCTS playout 用）
// ============================================================

fn count_expandable_corners<const N: usize>(board: &Board<N>, player: PlayerId) -> i32 {
    let mut count = 0;
    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Occupied(player) {
            for (nx, ny) in &[(x as i32 + 1, y as i32 + 1), (x as i32 + 1, y as i32 - 1),
                              (x as i32 - 1, y as i32 + 1), (x as i32 - 1, y as i32 - 1)] {
                if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty { count += 1; }
            }
        }
    }}
    count
}

fn center_proximity<const N: usize>(board: &Board<N>, player: PlayerId) -> f32 {
    let cx = (N / 2) as i32; let cy = (N / 2) as i32;
    let max_dist = (cx + cy) as f32;
    let mut total = 0.0; let mut cells = 0;
    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Occupied(player) {
            total += 1.0 - (((x as i32 - cx).abs() + (y as i32 - cy).abs()) as f32 / max_dist);
            cells += 1;
        }
    }}
    if cells == 0 { 0.0 } else { total / cells as f32 }
}

/// 估算某玩家的合法走法數（邊界空位數）
pub fn estimate_mobility<const N: usize>(board: &Board<N>, player: PlayerId) -> f32 {
    let mut frontier = 0;
    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Occupied(player) {
            for (nx, ny) in &[(x as i32 + 1, y as i32), (x as i32 - 1, y as i32),
                              (x as i32, y as i32 + 1), (x as i32, y as i32 - 1)] {
                if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty { frontier += 1; }
            }
        }
    }}
    frontier as f32
}

/// 未來 Mobility：從每個角出發 BFS 計算可達空格總數
/// 反映未來數步的擴張潛力
pub fn future_mobility<const N: usize>(board: &Board<N>, player: PlayerId) -> f32 {
    let mut visited: Vec<Vec<bool>> = vec![vec![false; N]; N];
    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    let mut total = 0.0;

    // 從所有角（對角空格）出發
    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Occupied(player) {
            for (nx, ny) in &[(x as i32 + 1, y as i32 + 1), (x as i32 + 1, y as i32 - 1),
                              (x as i32 - 1, y as i32 + 1), (x as i32 - 1, y as i32 - 1)] {
                if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty
                    && !visited[*ny as usize][*nx as usize] {
                    visited[*ny as usize][*nx as usize] = true;
                    queue.push_back((*nx, *ny));
                    total += 1.0;
                }
            }
        }
    }}

    while let Some((cx, cy)) = queue.pop_front() {
        for (nx, ny) in &[(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
            if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty
                && !visited[*ny as usize][*nx as usize] {
                visited[*ny as usize][*nx as usize] = true;
                queue.push_back((*nx, *ny));
                total += 1.0;
            }
        }
    }

    total
}

/// Reachable Area：與 future_mobility 相同
/// 計算每個角透過 BFS 可到達的空格總數
pub fn reachable_area<const N: usize>(board: &Board<N>, player: PlayerId) -> f32 {
    future_mobility(board, player)
}

/// 估算剩餘大型棋子的兼容性（不需完整掃描 all variants × positions）
/// 透過檢查最大棋子的可用性來推估
/// remaining_pieces: 玩家剩餘的棋子清單
pub fn piece_compatibility_estimate<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
) -> f32 {
    if remaining_pieces.is_empty() { return 0.5; }

    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));
    let corner = match player.0 {
        0 => Corner::TopLeft, 1 => Corner::TopRight,
        2 => Corner::BottomRight, _ => Corner::BottomLeft,
    };

    // 只檢查最大的 3 顆棋
    let mut sorted: Vec<&PieceShape> = remaining_pieces.iter().collect();
    sorted.sort_by(|a, b| b.base.cells.len().cmp(&a.base.cells.len()));
    let check_count = sorted.len().min(3);

    let mut fit_count = 0;
    for &shape in sorted.iter().take(check_count) {
        // 快速檢查：每個 variant 掃描直到找到一個合法位置
        let has_place = shape.variants.iter().any(|variant| {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    if x + variant.width > N as i32 || y + variant.height > N as i32 { continue; }
                    if board.is_valid(variant, x, y, player, is_first, Some(corner)).is_ok() { return true; }
                }
            }
            false
        });
        if has_place { fit_count += 1; }
    }

    fit_count as f32 / check_count as f32
}

/// Frontier Quality：corner/frontier 比例，越高代表外圍品質越好
pub fn frontier_quality<const N: usize>(board: &Board<N>, player: PlayerId) -> f32 {
    let corners = count_expandable_corners(board, player) as f32;
    let frontier = estimate_mobility(board, player);
    if frontier == 0.0 { 0.5 } else { (corners / frontier).min(1.0) }
}

/// Dead Region 懲罰：偵測無法被任何剩餘棋子有效利用的死區
/// 計算所有連續空區域中，小於某閾值的區域總面積比例
pub fn dead_region_penalty<const N: usize>(board: &Board<N>, min_piece_size: usize) -> f32 {
    let mut visited = vec![vec![false; N]; N];
    let mut dead_cells = 0;
    let mut total_empty = 0;

    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Empty && !visited[y][x] {
            // BFS 計算連續空區域大小
            let mut region_size = 0;
            let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
            queue.push_back((x as i32, y as i32));
            visited[y][x] = true;
            while let Some((cx, cy)) = queue.pop_front() {
                region_size += 1;
                for (nx, ny) in &[(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
                    if board.is_in_bounds(*nx, *ny) && board.cells[*ny as usize][*nx as usize] == CellState::Empty
                        && !visited[*ny as usize][*nx as usize] {
                        visited[*ny as usize][*nx as usize] = true;
                        queue.push_back((*nx, *ny));
                    }
                }
            }
            if region_size < min_piece_size as i32 {
                dead_cells += region_size;
            }
            total_empty += region_size;
        }
    }}

    if total_empty == 0 { 0.0 } else { dead_cells as f32 / total_empty as f32 }
}

/// 盤面評估：使用動態階段權重
pub fn heuristic_evaluate_with_weights<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    all_players: &[PlayerId],
    weights: &GamePhaseWeights,
) -> f32 {
    let mut my_score = 0.0;
    let mut opp_score = 0.0;

    for &p in all_players {
        let is_me = p == player;
        let corners = count_expandable_corners(board, p) as f32;
        let frontier = estimate_mobility(board, p);
        let center = center_proximity(board, p);
        let future = future_mobility(board, p);
        let fq = frontier_quality(board, p);
        apply_weight(&mut my_score, &mut opp_score, is_me, corners, weights.corner);
        apply_weight(&mut my_score, &mut opp_score, is_me, frontier, weights.mobility);
        apply_weight(&mut my_score, &mut opp_score, is_me, center, weights.center);
        apply_weight(&mut my_score, &mut opp_score, is_me, future, weights.future_mobility);
        apply_weight(&mut my_score, &mut opp_score, is_me, fq, weights.frontier_quality);
    }

    // Blocking：我方角數 - 對手平均角數
    let my_corners = count_expandable_corners(board, player) as f32;
    let opp_corners: f32 = all_players.iter()
        .filter(|p| **p != player)
        .map(|p| count_expandable_corners(board, *p) as f32)
        .sum::<f32>() / (all_players.len() - 1) as f32;
    my_score += (my_corners - opp_corners) * weights.blocking;

    // Dead Region：全局懲罰（對所有玩家相同）
    let dead = dead_region_penalty(board, 2); // 最小有效棋子 = 2 (domino)
    my_score -= dead * weights.dead_region * my_score.abs().max(1.0);
    opp_score += dead * weights.dead_region * opp_score.abs().max(1.0);

    let total = my_score + opp_score.abs();
    if total == 0.0 { 0.0 } else { my_score / total }
}

/// 相容舊介面（使用預設權重：中盤）
pub fn heuristic_evaluate<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    all_players: &[PlayerId],
) -> f32 {
    let weights = GamePhaseWeights { mobility: 6.0, corner: 3.5, blocking: 3.5, center: 0.2, future_mobility: 1.5, frontier_quality: 1.5, dead_region: 0.5 };
    heuristic_evaluate_with_weights(board, player, all_players, &weights)
}

fn apply_weight(my: &mut f32, opp: &mut f32, is_me: bool, value: f32, weight: f32) {
    if is_me { *my += value * weight; } else { *opp += value * weight; }
}

/// 終局盤面評估
pub fn evaluate_board<const N: usize>(
    board: &Board<N>, player: PlayerId, all_players: &[PlayerId],
) -> f32 {
    let mut score = 0.0;
    for &p in all_players {
        let owned = board.cells.iter().flatten()
            .filter(|&&c| c == CellState::Occupied(p)).count() as f32;
        if p == player { score += owned * 1.0; } else { score -= owned * 0.8; }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Board;
    use crate::game::piece::PieceVariant;
    use crate::game::player::PlayerId;

    #[test]
    fn test_score_placement_basic() {
        let board: Board<20> = Board::new();
        let variant = PieceVariant::new(vec![(0, 0)]);
        let score = score_placement(&board, &variant, 0, 0, PlayerId(0));
        assert!(score > 0);
    }

    #[test]
    fn test_count_corner_contacts() {
        let mut board: Board<20> = Board::new();
        let mono = PieceVariant::new(vec![(0, 0)]);
        board.try_place(&mono, 0, 0, PlayerId(0), true, Some(crate::game::board::Corner::TopLeft)).unwrap();
        let contacts = count_corner_contacts(&board, &mono, 1, 1, PlayerId(0));
        assert_eq!(contacts, 1);
    }

    #[test]
    fn test_heuristic_evaluate_empty_board() {
        let board: Board<20> = Board::new();
        let players = vec![PlayerId(0), PlayerId(1)];
        let weights = compute_phase_weights(0.0, 119.0);
    let score = heuristic_evaluate_with_weights(&board, PlayerId(0), &players, &weights);
    assert!((score - 0.5).abs() < 0.01, "Empty board both players equal, should be 0.5, got {}", score);
    }

    #[test]
    fn test_heuristic_evaluate_with_pieces() {
        let mut board: Board<20> = Board::new();
        let players = vec![PlayerId(0), PlayerId(1)];
        let mono = PieceVariant::new(vec![(0, 0)]);
        board.try_place(&mono, 0, 0, PlayerId(0), true, Some(crate::game::board::Corner::TopLeft)).unwrap();
        let weights = compute_phase_weights(1.0, 119.0);
        let score = heuristic_evaluate_with_weights(&board, PlayerId(0), &players, &weights);
        assert!(score > 0.0);
    }

    #[test]
    fn test_phase_weights_smooth() {
        let w = compute_phase_weights(0.0, 119.0);
        assert!((w.mobility - 5.0).abs() < 0.01, "Opening mobility should be 5.0");
        let w2 = compute_phase_weights(119.0, 119.0);
        assert!((w2.mobility - 8.0).abs() < 0.01, "Endgame mobility should be 8.0");
        let w3 = compute_phase_weights(59.5, 119.0);
        assert!(w3.blocking > 3.0, "Midgame blocking should peak above 3.0");
    }
}
