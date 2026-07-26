use crate::ai::evaluate;
use crate::ai::{list_legal_moves, AiMove};
use crate::game::board::{Board, Corner, CellState};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// 前瞻搜尋 AI（無 temperature，永遠選最高分）
pub fn choose_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    depth: usize,
    player_count: usize,
) -> Option<AiMove> {
    let scored = evaluate_candidates(board, player, remaining_pieces, is_first_move, starting_corner, depth, player_count);
    scored.into_iter().max_by(|a, b| a.4.cmp(&b.4)).map(|(pi, vi, x, y, sc)| AiMove {
        piece_index: pi, variant_index: vi, x, y, score: sc,
    })
}

/// 前瞻搜尋 AI + Temperature：各候選步評估後，依 net score 的 softmax 權重隨機選取
pub fn choose_move_with_temp<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    depth: usize,
    temp: f32,
    player_count: usize,
) -> Option<AiMove> {
    let scored = evaluate_candidates(board, player, remaining_pieces, is_first_move, starting_corner, depth, player_count);
    if scored.is_empty() {
        return None;
    }
    let scores: Vec<i32> = scored.iter().map(|(_, _, _, _, s)| *s).collect();
    let mut rng = rand::rng();
    let idx = evaluate::temperature_sample(&scores, temp, &mut rng);
    let (pi, vi, x, y, sc) = scored[idx];
    Some(AiMove { piece_index: pi, variant_index: vi, x, y, score: sc })
}

/// 評估候選步，回傳 Vec<(pi, vi, x, y, net_score)>
fn evaluate_candidates<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    depth: usize,
    player_count: usize,
) -> Vec<(usize, usize, i32, i32, i32)> {
    let candidates = list_legal_moves(board, player, remaining_pieces, is_first_move, starting_corner);
    if candidates.is_empty() {
        return vec![];
    }

    const CANDIDATE_LIMIT: usize = 10;
    let mut candidates = candidates;
    candidates.sort_by(|a, b| b.4.cmp(&a.4));
    candidates.truncate(CANDIDATE_LIMIT);

    let mut results = Vec::with_capacity(candidates.len());

    for &(pi, vi, x, y, _base) in &candidates {
        let variant = &remaining_pieces[pi].variants[vi];
        let mut sim_board = board.clone();
        sim_board.place_piece(variant, x, y, player);

        let opponents: Vec<PlayerId> = (0..player_count).map(PlayerId).filter(|p| *p != player).collect();
        let mut worst_opponent_score = f32::MIN;

        for &opp in &opponents {
            let opp_moves = find_best_response(&sim_board, opp, remaining_pieces, depth.saturating_sub(1), player_count);
            let opp_score = opp_moves.unwrap_or(0.0);
            if opp_score > worst_opponent_score {
                worst_opponent_score = opp_score;
            }
        }

        let my_score = evaluate::score_placement(board, variant, x, y, player) as f32;
        let net = (my_score - worst_opponent_score) as i32;
        results.push((pi, vi, x, y, net));
    }

    results
}

/// 找對手的最佳回應分數（遞迴 depth）
fn find_best_response<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    all_pieces: &[PieceShape],
    depth: usize,
    player_count: usize,
) -> Option<f32> {
    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));
    let remaining: Vec<PieceShape> = all_pieces.to_vec();

    let moves = list_legal_moves(board, player, &remaining, is_first, Some(match player.0 {
        0 => Corner::TopLeft,
        1 => Corner::TopRight,
        2 => Corner::BottomRight,
        _ => Corner::BottomLeft,
    }));

    if moves.is_empty() {
        return None;
    }

    const OPP_CANDIDATES: usize = 3;
    let mut candidates = moves;
    candidates.sort_by(|a, b| b.4.cmp(&a.4));
    candidates.truncate(OPP_CANDIDATES);

    let mut best_score = f32::MIN;

    for &(pi, vi, x, y, _base) in &candidates {
        let variant = &remaining[pi].variants[vi];
        let mut sim = board.clone();
        sim.place_piece(variant, x, y, player);

        let base_score = evaluate::score_placement(board, variant, x, y, player) as f32;

        let total = if depth > 0 {
            let next_opp: PlayerId = PlayerId((player.0 + 1) % player_count);
            if let Some(opp_score) = find_best_response(&sim, next_opp, &remaining, depth - 1, player_count) {
                base_score - opp_score * 0.5
            } else {
                base_score
            }
        } else {
            base_score
        };

        if total > best_score {
            best_score = total;
        }
    }

    Some(best_score)
}
