use rand::seq::SliceRandom;

use crate::ai::evaluate;
use crate::ai::list_legal_moves;
use crate::ai::move_ordering;
use crate::ai::transposition::{TranspositionTable, TTEntry, TTFlag};
use crate::ai::zobrist::ZobristTable;
use crate::ai::AiMove;
use crate::game::board::{Board, CellState, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

const UCB_C: f32 = 3.0;
const MAX_TT_SIZE: usize = 50_000;
const PARALLEL_THREADS: usize = 4;

// ============================================================
// Progressive Widening
// ============================================================

fn max_children(visits: u32) -> usize {
    if visits < 5 { 2 }
    else if visits < 15 { 5 }
    else if visits < 40 { 10 }
    else { usize::MAX }
}

// ============================================================
// 樹節點
// ============================================================

#[derive(Clone)]
struct Edge {
    piece_index: usize,
    variant_index: usize,
    x: i32,
    y: i32,
    child_idx: usize,
    visits: u32,
    total_score: f32,
}

#[derive(Clone)]
struct TreeNode {
    visits: u32,
    total_score: f32,
    children: Vec<Edge>,
    unexpanded: Vec<(usize, usize, i32, i32)>,
}

#[derive(Clone)]
struct Tree {
    nodes: Vec<TreeNode>,
}

impl Tree {
    fn new() -> Self {
        Self { nodes: vec![TreeNode { visits: 0, total_score: 0.0, children: vec![], unexpanded: vec![] }] }
    }

    fn add_child(&mut self, parent: usize, pi: usize, vi: usize, x: i32, y: i32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode { visits: 0, total_score: 0.0, children: vec![], unexpanded: vec![] });
        self.nodes[parent].children.push(Edge {
            piece_index: pi, variant_index: vi, x, y,
            child_idx: idx, visits: 0, total_score: 0.0,
        });
        idx
    }

    fn ucb1_select(&self, node: &TreeNode) -> usize {
        let log_parent = (node.visits as f32).ln();
        let mut best = 0;
        let mut best_val = f32::MIN;
        for (i, c) in node.children.iter().enumerate() {
            let val = if c.visits == 0 {
                f32::MAX
            } else {
                c.total_score / c.visits as f32 + (log_parent / c.visits as f32).sqrt() * UCB_C
            };
            if val > best_val { best_val = val; best = i; }
        }
        best
    }
}

// ============================================================
// 主入口
// ============================================================

pub fn choose_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    iterations: usize,
    player_count: usize,
) -> Option<AiMove> {
    let mut candidates = list_legal_moves(board, player, remaining_pieces, is_first_move, starting_corner);
    if candidates.is_empty() { return None; }
    let occupied = board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;
    move_ordering::order_moves(&mut candidates, board, player, remaining_pieces, occupied, 119.0);

    let all_players: Vec<PlayerId> = (0..player_count).map(PlayerId).collect();
    let per_thread = iterations / PARALLEL_THREADS;

    let tree_results: Vec<Tree> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..PARALLEL_THREADS {
            let board_c = board.clone();
            let remaining_c = remaining_pieces.to_vec();
            let all_players_c = all_players.clone();
            let candidates_c = candidates.clone();
            handles.push(s.spawn(move || {
                run_mcts::<N>(&board_c, player, &remaining_c, &all_players_c, &candidates_c, per_thread)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>()
    });

    // 合併多棵樹 (children visits + total_score)
    let mut merged = Tree::new();
    merged.nodes[0].unexpanded = candidates.iter().map(|&(pi, vi, x, y, _)| (pi, vi, x, y)).collect();

    for tree in &tree_results {
        for child in &tree.nodes[0].children {
            if let Some(mc) = merged.nodes[0].children.iter_mut()
                .find(|c| c.piece_index == child.piece_index && c.variant_index == child.variant_index
                      && c.x == child.x && c.y == child.y) {
                mc.visits += child.visits;
                mc.total_score += child.total_score;
            } else {
                // 未合併過的子節點，直接加入
                let idx = merged.nodes.len();
                merged.nodes.push(TreeNode { visits: child.visits, total_score: child.total_score, children: vec![], unexpanded: vec![] });
                merged.nodes[0].children.push(Edge {
                    piece_index: child.piece_index, variant_index: child.variant_index,
                    x: child.x, y: child.y, child_idx: idx,
                    visits: child.visits, total_score: child.total_score,
                });
            }
        }
    }

    merged.nodes[0].children.iter()
        .max_by_key(|c| c.visits)
        .map(|c| {
            let avg = if c.visits > 0 { c.total_score / c.visits as f32 } else { 0.0 };
            AiMove {
                piece_index: c.piece_index, variant_index: c.variant_index,
                x: c.x, y: c.y, score: (avg * 1000.0) as i32,
            }
        })
}

// ============================================================
// 單一執行緒 MCTS
// ============================================================

fn run_mcts<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    all_players: &[PlayerId],
    candidates: &[(usize, usize, i32, i32, i32)],
    iterations: usize,
) -> Tree {
    let mut tree = Tree::new();
    let mut tt = TranspositionTable::new(MAX_TT_SIZE);
    let zobrist = ZobristTable::new();

    // Root 不一次全部展開，改為 Progressive Widening
    tree.nodes[0].unexpanded = candidates.iter().map(|&(pi, vi, x, y, _)| (pi, vi, x, y)).collect();

    for _ in 0..iterations {
        let mut path: Vec<(usize, Option<usize>)> = vec![(0, None)];
        let mut sim_board = board.clone();
        let mut sim_player = player;
        let mut node_idx = 0;
        let mut tt_hit = false;

        // Selection
        loop {
            let node = &tree.nodes[node_idx];

            // 若還有未展開的步，且尚未超過 Progressive Widening 上限 → 此時展開
            if !node.unexpanded.is_empty() && node.children.len() < max_children(node.visits) {
                break;
            }

            // 無子節點 → rollout
            if node.children.is_empty() { break; }

            // TT 查詢
            let h = zobrist.hash(&sim_board);
            if let Some(entry) = tt.lookup(h) {
                if entry.depth >= 1 {
                    let score = entry.score;
                    for &(nidx, p_opt) in path.iter().rev() {
                        tree.nodes[nidx].visits += 1; tree.nodes[nidx].total_score += score;
                        if let Some(pidx) = p_opt {
                            if let Some(edge) = tree.nodes[pidx].children.iter_mut().find(|e| e.child_idx == nidx) {
                                edge.visits += 1; edge.total_score += score;
                            }
                        }
                    }
                    tt_hit = true; break;
                }
            }

            let choice = tree.ucb1_select(node);
            let child = &node.children[choice];
            let cidx = child.child_idx;
            let variant = &remaining_pieces[child.piece_index].variants[child.variant_index];
            sim_board.place_piece(variant, child.x, child.y, sim_player);
            sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
            path.push((cidx, Some(node_idx)));
            node_idx = cidx;
        }
        if tt_hit { continue; }

        // Expansion：從 unexpanded 取一個步展開
        let node = &mut tree.nodes[node_idx];
        if !node.unexpanded.is_empty() && node.children.len() < max_children(node.visits) {
            if let Some((pi, vi, x, y)) = node.unexpanded.pop() {
                let c = tree.add_child(node_idx, pi, vi, x, y);
                let variant = &remaining_pieces[pi].variants[vi];
                sim_board.place_piece(variant, x, y, sim_player);
                path.push((c, Some(node_idx)));
            }
        }

        // Playout
        let result = fast_playout(&sim_board, player, all_players, remaining_pieces);

        // Backpropagation
        for &(nidx, p_opt) in path.iter().rev() {
            tree.nodes[nidx].visits += 1;
            tree.nodes[nidx].total_score += result;
            if let Some(pidx) = p_opt {
                if let Some(edge) = tree.nodes[pidx].children.iter_mut().find(|e| e.child_idx == nidx) {
                    edge.visits += 1;
                    edge.total_score += result;
                }
            }
        }

        // 存入 TT
        let h = zobrist.hash(&sim_board);
        let score = tree.nodes[node_idx].total_score / tree.nodes[node_idx].visits.max(1) as f32;
        tt.insert(h, TTEntry {
            depth: 1, score, flag: TTFlag::Exact,
            best_move: tree.nodes[0].children.iter().max_by_key(|c| c.visits).map(|c| {
                let avg = if c.visits > 0 { c.total_score / c.visits as f32 } else { 0.0 };
                AiMove {
                    piece_index: c.piece_index, variant_index: c.variant_index,
                    x: c.x, y: c.y, score: (avg * 1000.0) as i32,
                }
            }),
        });
    }

    tree
}

// ============================================================
// Playout
// ============================================================

/// 根據遊戲進度動態決定 playout 深度
fn get_playout_depth(occupied: f32, total: f32) -> usize {
    let progress = occupied / total;
    if progress < 0.3 { 8 }        // Opening
    else if progress < 0.7 { 12 }  // Midgame
    else { 20 }                    // Endgame
}

/// 收集 rollout 候選步（最多 24 個，依棋子大小順序掃描）
fn collect_candidates<const N: usize>(
    board: &Board<N>, player: PlayerId,
    all_pieces: &[PieceShape], used: &[usize],
    piece_order: &[usize],
) -> Vec<(usize, usize, i32, i32)> {
    const MAX_CAND: usize = 24;
    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));
    let corner = match player.0 {
        0 => Corner::TopLeft, 1 => Corner::TopRight,
        2 => Corner::BottomRight, _ => Corner::BottomLeft,
    };
    let mut candidates = Vec::with_capacity(MAX_CAND);
    for &pi in piece_order {
        if used.contains(&pi) { continue; }
        let shape = &all_pieces[pi];
        for vi in 0..shape.variants.len() {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    let v = &shape.variants[vi];
                    if x + v.width > N as i32 || y + v.height > N as i32 { continue; }
                    if board.is_valid(v, x, y, player, is_first, Some(corner)).is_ok() {
                        candidates.push((pi, vi, x, y));
                        if candidates.len() >= MAX_CAND { return candidates; }
                    }
                }
            }
        }
    }
    candidates
}

/// 加權隨機選取 Top N（第一名 40% → 第五名 5%）
fn weighted_random_top5(scored: &[(usize, usize, i32, i32, i32)]) -> usize {
    let n = scored.len().min(5);
    if n == 0 { return 0; }
    let weights = [0.40, 0.30, 0.15, 0.10, 0.05];
    let total: f32 = weights[..n].iter().sum();
    let mut roll: f32 = rand::random::<f32>() * total;
    for i in 0..n { roll -= weights[i]; if roll <= 0.0 { return i; } }
    n - 1
}

/// 建立依棋子大小降序排列的索引（同大小內 shuffle）
fn build_piece_order(all_pieces: &[PieceShape]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..all_pieces.len()).collect();
    order.sort_by(|&a, &b| all_pieces[b].base.cells.len().cmp(&all_pieces[a].base.cells.len()));
    let mut i = 0;
    while i < order.len() {
        let size = all_pieces[order[i]].base.cells.len();
        let mut j = i;
        while j < order.len() && all_pieces[order[j]].base.cells.len() == size { j += 1; }
        order[i..j].shuffle(&mut rand::rng());
        i = j;
    }
    order
}

fn fast_playout<const N: usize>(
    board: &Board<N>,
    root_player: PlayerId,
    all_players: &[PlayerId],
    all_pieces: &[PieceShape],
) -> f32 {
    let occupied = board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;
    let weights = evaluate::compute_phase_weights(occupied, 119.0);
    let mut sim_board = board.clone();
    let mut sim_player = root_player;
    let mut passes = 0;
    let mut used: Vec<usize> = Vec::new();
    let mut steps: [usize; 4] = [0; 4];

    let piece_order = build_piece_order(all_pieces);
    let mut total_occupied = occupied;

    loop {
        let depth = get_playout_depth(total_occupied, 119.0);
        if steps[sim_player.0] >= depth {
            return evaluate::heuristic_evaluate_with_weights(&sim_board, root_player, all_players, &weights);
        }

        // Candidate Sampling + Top5 Weighted Random
        let raw = collect_candidates(&sim_board, sim_player, all_pieces, &used, &piece_order);

        if raw.is_empty() {
            passes += 1;
            if passes >= 4 * all_players.len() { break; }
            sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
            continue;
        }

        // 對候選步快速評分（corner contact ×15）
        let mut scored: Vec<(usize, usize, i32, i32, i32)> = raw.iter().map(|&(pi, vi, x, y)| {
            let variant = &all_pieces[pi].variants[vi];
            let s = evaluate::count_corner_contacts(&sim_board, variant, x, y, sim_player) * 15;
            (pi, vi, x, y, s)
        }).collect();
        scored.sort_by(|a, b| b.4.cmp(&a.4));

        let idx = weighted_random_top5(&scored);
        let (pi, vi, x, y, _) = scored[idx];
        let variant = &all_pieces[pi].variants[vi];
        sim_board.place_piece(variant, x, y, sim_player);
        used.push(pi);
        steps[sim_player.0] += 1;
        passes = 0;
        total_occupied = sim_board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;

        sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
    }

    evaluate::heuristic_evaluate_with_weights(&sim_board, root_player, all_players, &weights)
}
