use std::sync::atomic::{AtomicBool, Ordering};

use rand::seq::SliceRandom;

use crate::ai::evaluate;
use crate::ai::list_legal_moves;
use crate::ai::move_ordering;
use crate::ai::transposition::{TranspositionTable, TTEntry, TTFlag};
use crate::ai::zobrist::ZobristTable;
use crate::ai::AiMove;
use crate::game::board::{Board, CellState, Corner};
use crate::game::piece::{PieceShape, PieceVariant};
use crate::game::player::PlayerId;

const PUCT_C: f32 = 1.25;
const VIRTUAL_LOSS: f32 = 0.05;
const MAX_TT_SIZE: usize = 50_000;

const DIAGONAL_DIRS: [(i32, i32); 4] = [
    (1,1),(1,-1),(-1,1),(-1,-1),
];

// ============================================================
// Progressive Widening
// ============================================================

fn max_children(visits: u32) -> usize {
    let cap = (visits as f32 * 0.3 + 1.0).sqrt() as usize;
    cap.clamp(2, 50)
}

// ============================================================
// 樹節點
// ============================================================

#[derive(Clone)]
pub(crate) struct Edge {
    pub(crate) piece_index: usize,
    pub(crate) variant_index: usize,
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) child_idx: usize,
    pub(crate) visits: u32,
    pub(crate) total_score: f32,
    pub(crate) prior: f32,
    pub(crate) virtual_loss: f32,
}

#[derive(Clone)]
pub(crate) struct TreeNode {
    pub(crate) visits: u32,
    pub(crate) total_score: f32,
    pub(crate) children: Vec<Edge>,
    pub(crate) unexpanded: Vec<(usize, usize, i32, i32)>,
    pub(crate) nn_value: f32, // -1.0 = 未計算
}

#[derive(Clone)]
pub(crate) struct Tree {
    pub(crate) nodes: Vec<TreeNode>,
}

impl Tree {
    fn new() -> Self {
        Self { nodes: vec![TreeNode { visits: 0, total_score: 0.0, children: vec![], unexpanded: vec![], nn_value: -1.0 }] }
    }

    fn add_child(&mut self, parent: usize, pi: usize, vi: usize, x: i32, y: i32, prior: f32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode { visits: 0, total_score: 0.0, children: vec![], unexpanded: vec![], nn_value: -1.0 });
        self.nodes[parent].children.push(Edge {
            piece_index: pi, variant_index: vi, x, y,
            child_idx: idx, visits: 0, total_score: 0.0,
            prior, virtual_loss: 0.0,
        });
        idx
    }

    fn puct_select(&self, node: &TreeNode) -> usize {
        let sqrt_parent = (node.visits as f32).sqrt();
        let mut best = 0;
        let mut best_val = f32::MIN;
        for (i, c) in node.children.iter().enumerate() {
            let q = if c.visits > 0 { c.total_score / c.visits as f32 } else { 0.0 };
            let u = PUCT_C * c.prior * sqrt_parent / (1.0 + c.visits as f32);
            let vl = c.virtual_loss / (c.visits.max(1) as f32 + 1.0);
            let val = q + u - vl;
            if val > best_val { best_val = val; best = i; }
        }
        best
    }
}

// ============================================================
// RolloutBoard（輕量棋盤，用於 rollout）
// ============================================================

#[derive(Clone)]
/// Rollout profiling（最小版本）
#[derive(Default)]
struct RolloutProfile {
    _frontier_us: u64,
    _candidates_us: u64,
    _score_us: u64,
    _eval_us: u64,
    _total_us: u64,
    _count: u64,
}

struct RolloutBoard<const N: usize> {
    cells: [[u8; N]; N],
    row_masks: [u32; N],
    occupied: f32,
}

impl<const N: usize> RolloutBoard<N> {
    fn from_board(board: &Board<N>) -> Self {
        let mut cells = [[0u8; N]; N];
        let mut row_masks = [0u32; N];
        let mut occupied = 0.0;
        for y in 0..N {
            for x in 0..N {
                let c = match board.cells[y][x] {
                    CellState::Empty => 0,
                    CellState::Occupied(p) => { occupied += 1.0; p.0 as u8 + 1 }
                };
                cells[y][x] = c;
                if c != 0 { row_masks[y] |= 1u32 << x; }
            }
        }
        Self { cells, row_masks, occupied }
    }

    fn overlap_check(&self, variant: &PieceVariant, pos_x: i32, pos_y: i32) -> bool {
        for &(dx, dy) in &variant.cells {
            let ax = pos_x + dx;
            let ay = pos_y + dy;
            if ax < 0 || ax >= N as i32 || ay < 0 || ay >= N as i32 { return true; }
            if (self.row_masks[ay as usize] & (1u32 << (ax as usize))) != 0 { return true; }
        }
        false
    }

    fn occupied_count(&self) -> f32 { self.occupied }

    fn to_board<const M: usize>(&self) -> Board<M> {
        let mut b = Board::new();
        for y in 0..M.min(N) {
            for x in 0..M.min(N) {
                let v = self.cells[y][x];
                if v != 0 {
                    b.cells[y][x] = CellState::Occupied(PlayerId(v as usize - 1));
                }
            }
        }
        b
    }
}

/// 建立 frontier（當前玩家的 diagonal 空格）
fn build_frontier_fast<const N: usize>(board: &RolloutBoard<N>, player: PlayerId) -> Vec<(i32, i32)> {
    let pid = player.0 + 1;
    let mut frontier = Vec::new();
    for y in 0..N {
        for x in 0..N {
            if board.cells[y][x] == pid as u8 {
                let px = x as i32; let py = y as i32;
                for &(dx, dy) in &DIAGONAL_DIRS {
                    let nx = px + dx; let ny = py + dy;
                    if nx < 0 || nx >= N as i32 || ny < 0 || ny >= N as i32 { continue; }
                    if board.cells[ny as usize][nx as usize] != 0 { continue; }
                    frontier.push((nx, ny));
                }
            }
        }
    }
    frontier.sort_unstable();
    frontier.dedup();
    frontier
}

/// 收集 rollout 階段候選步
fn collect_candidates_rollout<const N: usize>(
    board: &RolloutBoard<N>, _player: PlayerId,
    remaining_pieces: &[PieceShape], used: &[usize],
    piece_order: &[usize], frontier: &[(i32, i32)],
    max_cand: usize,
) -> Vec<(usize, usize, i32, i32)> {
    let frontier_limit = frontier.len().min(32);
    let mut candidates = Vec::new();

    for &pi in piece_order {
        if used.contains(&pi) { continue; }
        if candidates.len() >= max_cand { break; }
        let shape = &remaining_pieces[pi];
        let per_piece_limit = shape.base.cells.len().max(2) * 3;

        let mut piece_cand = 0;
        let variants: Vec<&PieceVariant> = shape.variants.iter().take(2).collect();
        'outer: for (vi_idx, v) in variants.iter().enumerate() {
            for &(fx, fy) in frontier.iter().take(frontier_limit) {
                if piece_cand >= per_piece_limit { break 'outer; }
                for &(cdx, cdy) in &v.cells {
                    let px = fx - cdx;
                    let py = fy - cdy;
                    if px < 0 || px >= N as i32 || py < 0 || py >= N as i32 { continue; }
                    if board.overlap_check(v, px, py) { continue; }
                    candidates.push((pi, vi_idx, px, py));
                    piece_cand += 1;
                    if piece_cand >= per_piece_limit { break 'outer; }
                    break;
                }
            }
        }
    }
    candidates
}

/// 更新 rollout frontier（新棋子放置後加入新 frontier）
fn update_frontier_rollout<const N: usize>(
    board: &RolloutBoard<N>, frontier: &mut Vec<(i32, i32)>, variant: &PieceVariant, ax: i32, ay: i32,
) {
    for &(dx, dy) in &variant.cells {
        let px = ax + dx; let py = ay + dy;
        for &(ndx, ndy) in &DIAGONAL_DIRS {
            let nx = px + ndx; let ny = py + ndy;
            if nx >= 0 && nx < N as i32 && ny >= 0 && ny < N as i32
                && board.cells[ny as usize][nx as usize] == 0 {
                let pos = (nx, ny);
                if !frontier.contains(&pos) { frontier.push(pos); }
            }
        }
    }
}

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

impl Tree {
    /// 依據實際落子，將 root 推進到對應子節點（保留 subtree）
    #[allow(dead_code)]
    pub(crate)     fn advance_root(&mut self, mv: &crate::ai::AiMove) -> bool {
        if self.nodes.is_empty() { return false; }
        let child_idx = self.nodes[0].children.iter()
            .position(|c| c.piece_index == mv.piece_index && c.variant_index == mv.variant_index && c.x == mv.x && c.y == mv.y);
        match child_idx {
            Some(idx) => {
                let child = &self.nodes[0].children[idx];
                let new_root = child.child_idx;
                let subtree: Vec<TreeNode> = self.nodes.drain(new_root..).collect();
                if subtree.is_empty() {
                    *self = Tree::new();
                } else {
                    self.nodes = subtree;
                    self.nodes[0].visits = 0;
                    self.nodes[0].total_score = 0.0;
                }
                true
            }
            None => false,
        }
    }
}

// ============================================================
// Ponder（背景搜尋）
// ============================================================

static PONDER_ACTIVE: AtomicBool = AtomicBool::new(false);
static PONDER_STOP: AtomicBool = AtomicBool::new(false);
static PONDER_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn set_ponder_enabled(enabled: bool) {
    PONDER_ENABLED.store(enabled, Ordering::SeqCst);
}
fn available_ponder_threads() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get()).unwrap_or(4);
    1usize.min(cores.saturating_sub(1).max(1))
}

/// 啟動背景搜尋（在等待對手時預先計算）
fn start_ponder<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    all_players: &[PlayerId],
    merged: &Tree,
    iterations: usize,
) {
    if !PONDER_ENABLED.load(Ordering::SeqCst) { return; }
    if PONDER_ACTIVE.load(Ordering::SeqCst) { return; }
    PONDER_STOP.store(false, Ordering::SeqCst);
    PONDER_ACTIVE.store(true, Ordering::SeqCst);

    let board_c = board.clone();
    let remaining_c = remaining_pieces.to_vec();
    let all_players_c = all_players.to_vec();
    let tree = merged.clone();

    std::thread::spawn(move || {
        // 取 top-5 對手回應，從對手視角輕量搜尋
        let mut children: Vec<_> = tree.nodes[0].children.iter()
            .filter(|c| c.visits > 0)
            .collect();
        children.sort_by(|a, b| b.visits.cmp(&a.visits));

        let mut ponder_tree = Tree::new();
        for child in children.iter().take(5) {
            if PONDER_STOP.load(Ordering::SeqCst) { break; }
            let mut sub_tree = tree.clone();
            let child_mv = crate::ai::AiMove {
                piece_index: child.piece_index, variant_index: child.variant_index,
                x: child.x, y: child.y, score: 0,
            };
            if !sub_tree.advance_root(&child_mv) {
                continue;
            }
            // 從對手視角執行簡短 MCTS
            let next_player = PlayerId((player.0 + 1) % all_players_c.len());
            let sim_board = {
                let mut b = board_c.clone();
                let variant = &remaining_c[child.piece_index].variants[child.variant_index];
                b.place_piece(variant, child.x, child.y, player);
                b
            };
            let cand = list_legal_moves(&sim_board, next_player, &remaining_c, false, None);
            if cand.is_empty() { continue; }
            let pt = available_ponder_threads();
            let per_thread = (iterations / 10).max(5) / pt.max(1);
            let _r = std::thread::scope(|s| {
                for _ in 0..pt {
                    let bc = sim_board.clone();
                    let rc = remaining_c.clone();
                    let apc = all_players_c.clone();
                    let cc = cand.clone();
                    s.spawn(move || {
                        let _t = run_mcts::<N>(&bc, next_player, &rc, &apc, &cc, per_thread);
                    });
                }
            });
            // 合併結果
            if sub_tree.nodes.len() > 1 {
                for sc in &sub_tree.nodes[0].children {
                    let idx = ponder_tree.nodes.len();
                    ponder_tree.nodes.push(TreeNode {
                        visits: sc.visits,
                        total_score: sc.total_score,
                        children: vec![],
                        unexpanded: vec![],
                        nn_value: -1.0,
                    });
                    ponder_tree.nodes[0].children.push(Edge {
                        piece_index: sc.piece_index,
                        variant_index: sc.variant_index,
                        x: sc.x, y: sc.y,
                        child_idx: idx,
                        visits: sc.visits,
                        total_score: sc.total_score,
                        prior: 0.0, virtual_loss: 0.0,
                    });
                }
            }
        }
        PONDER_ACTIVE.store(false, Ordering::SeqCst);
    });
}

/// 停止背景搜尋
fn stop_ponder() {
    PONDER_STOP.store(true, Ordering::SeqCst);
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
    config: &crate::ai::config::MctsConfig,
    player_count: usize,
    _stats: &mut Option<crate::ai::MctsOutput>,
    search_state: &mut Option<crate::ai::SearchState>,
) -> Option<AiMove> {
    crate::ai::value::clear_nn_cache();
    stop_ponder();

    let iterations = config.iterations;
    let mut candidates = list_legal_moves(board, player, remaining_pieces, is_first_move, starting_corner);
    if candidates.is_empty() { return None; }
    let occupied = board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;
    move_ordering::order_moves(&mut candidates, board, player, remaining_pieces, occupied, 119.0);

    let all_players: Vec<PlayerId> = (0..player_count).map(PlayerId).collect();
    let n_threads = config.parallel_threads.max(1);
    let per_thread = iterations / n_threads;

    let tree_results: Vec<Tree> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..n_threads {
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
    merged.nodes[0].unexpanded = candidates.iter().map(|&(pi, vi, x, y, _)| (pi, vi, x, y)).rev().collect();

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
                merged.nodes.push(TreeNode { visits: child.visits, total_score: child.total_score, children: vec![], unexpanded: vec![], nn_value: -1.0 });
                merged.nodes[0].children.push(Edge {
                    piece_index: child.piece_index, variant_index: child.variant_index,
                    x: child.x, y: child.y, child_idx: idx,
                    visits: child.visits, total_score: child.total_score,
                    prior: child.prior, virtual_loss: child.virtual_loss,
                });
            }
        }
    }

    let best = merged.nodes[0].children.iter()
        .max_by_key(|c| c.visits)
        .map(|c| {
            let avg = if c.visits > 0 { c.total_score / c.visits as f32 } else { 0.0 };
            AiMove {
                piece_index: c.piece_index, variant_index: c.variant_index,
                x: c.x, y: c.y, score: (avg * 1000.0) as i32,
            }
        });

    // Profile 輸出（僅 print_profile 開啟時）
    if config.print_profile {
        let total_visits: u32 = merged.nodes[0].children.iter().map(|c| c.visits).sum();
        eprintln!("\nAI Turn Profile (pid={}, {} iters, {} players):", player.0, config.iterations, player_count);
        eprintln!("  Candidate checks: {} total, best visits={}  total_visits={}", candidates.len(), total_visits, iterations);
        eprintln!("Root candidates (Top 10):");
        let mut children: Vec<_> = merged.nodes[0].children.iter().enumerate().collect();
        children.sort_by(|(_, a), (_, b)| b.visits.cmp(&a.visits));
        for (i, child) in children.iter().take(10) {
            let q = if child.visits > 0 { child.total_score / child.visits as f32 } else { 0.0 };
            eprintln!("  {}. piece={:2} prior={:.3} visit={:4} Q={:.3}", i+1, child.piece_index, child.prior, child.visits, q);
        }
        crate::ai::value::print_cache_stats();
    }

    // 啟動背景搜尋
    start_ponder(board, player, remaining_pieces, &all_players, &merged, iterations);

    // 儲存樹供下一回合 Tree Reuse
    if let Some(state) = search_state {
        state.tree = Some(Box::new(merged));
    }

    best
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
    tree.nodes[0].unexpanded = candidates.iter().map(|&(pi, vi, x, y, _)| (pi, vi, x, y)).rev().collect();

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

            let choice = tree.puct_select(node);
            let (pi, vi, cx, cy, cidx) = {
                let c = &node.children[choice];
                (c.piece_index, c.variant_index, c.x, c.y, c.child_idx)
            };
            tree.nodes[node_idx].children[choice].virtual_loss += VIRTUAL_LOSS;
            let variant = &remaining_pieces[pi].variants[vi];
            sim_board.place_piece(variant, cx, cy, sim_player);
            sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
            path.push((cidx, Some(node_idx)));
            node_idx = cidx;
        }
        if tt_hit { continue; }

        // Expansion：softmax 正規化 prior 後展開一步
        let mut expanded_nn = -1.0;
        if let Some(node) = tree.nodes.get_mut(node_idx) {
            if !node.unexpanded.is_empty() && node.children.len() < max_children(node.visits) {
                // 計算所有 unexpanded 的 raw prior
                let progress = evaluate::compute_progress(board.cells.iter().flatten()
                    .filter(|&&c| c != CellState::Empty).count() as f32, evaluate::TOTAL_PIECE_AREA);
                let raw_priors: Vec<f32> = node.unexpanded.iter()
                    .map(|&(pi, vi, x, y)| {
                        let v = &remaining_pieces[pi].variants[vi];
                        move_ordering::compute_prior(board, v, x, y, player, progress).max(0.0)
                    })
                    .collect();
                // Softmax 正規化
                let tau = 15.0 * (1.0 - progress) + 5.0 * progress;
                let max_raw = raw_priors.iter().cloned().fold(f32::MIN, f32::max);
                let shifted: Vec<f32> = raw_priors.iter()
                    .map(|s| ((s - max_raw) / tau).exp())
                    .collect();
                let sum: f32 = shifted.iter().sum();

                // Pop 並使用正規化後的 prior
                if let Some((pi, vi, x, y)) = node.unexpanded.pop() {
                    let idx = node.unexpanded.len(); // 已被 pop，長度即此項原 index
                    let prior = if sum > 0.0 { shifted[idx] / sum } else { 1.0 };
                    let variant = &remaining_pieces[pi].variants[vi];
                    let c = tree.add_child(node_idx, pi, vi, x, y, prior);
                    sim_board.place_piece(variant, x, y, sim_player);
                    path.push((c, Some(node_idx)));
                    // 計算 NN value
                    let vn = crate::ai::value::get_value_network();
                    if vn.is_loaded() {
                        let nn_v = vn.evaluate(&sim_board, player.0 as usize, all_players.len());
                        if let Some(n) = tree.nodes.get_mut(c) { n.nn_value = nn_v; }
                        expanded_nn = nn_v;
                    }
                }
            }
        }

        // 從 path 最後節點取得 nn_value
        let leaf_nn = path.last().and_then(|&(nidx, _)| tree.nodes.get(nidx)).map(|n| n.nn_value).unwrap_or(-1.0);
        let leaf_nn = leaf_nn.max(expanded_nn);

        // Playout
        let (result, _depth) = fast_playout(&sim_board, player, all_players, remaining_pieces, leaf_nn, None);

        // Backpropagation（含 virtual_loss 清除）
        for &(nidx, p_opt) in path.iter().rev() {
            tree.nodes[nidx].visits += 1;
            tree.nodes[nidx].total_score += result;
            if let Some(pidx) = p_opt {
                if let Some(edge) = tree.nodes[pidx].children.iter_mut().find(|e| e.child_idx == nidx) {
                    edge.visits += 1;
                    edge.total_score += result;
                    edge.virtual_loss = (edge.virtual_loss - VIRTUAL_LOSS).max(0.0);
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

fn fast_playout<const N: usize>(
    board: &Board<N>,
    root_player: PlayerId,
    all_players: &[PlayerId],
    all_pieces: &[PieceShape],
    nn_value: f32,
    _profile: Option<&mut RolloutProfile>,
) -> (f32, usize) {
    let mut sim = RolloutBoard::<N>::from_board(board);
    let mut rng = rand::rng();
    let mut sim_player = root_player;
    let mut passes = 0;
    let mut used: Vec<Vec<usize>> = (0..all_players.len()).map(|_| Vec::new()).collect();
    let mut steps: [usize; 4] = [0; 4];

    let piece_order = build_piece_order(all_pieces);
    let mut total_occupied = sim.occupied_count();
    let mut prev_player = sim_player;
    let mut frontier = build_frontier_fast::<N>(&sim, sim_player);

    loop {
        let playout_steps = get_playout_depth(total_occupied, evaluate::TOTAL_PIECE_AREA);
        if steps[sim_player.0] >= playout_steps {
            break;
        }

        if sim_player != prev_player {
            frontier = build_frontier_fast::<N>(&sim, sim_player);
            prev_player = sim_player;
        }

        if frontier.is_empty() {
            passes += 1;
            if passes >= 4 * all_players.len() { break; }
            sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
            continue;
        }

        let progress = evaluate::compute_progress(total_occupied, evaluate::TOTAL_PIECE_AREA);
        let max_cand = 18;

        // 打亂 frontier 順序以增加 rollout 多樣性
        let mut shuffled_frontier = frontier.clone();
        shuffled_frontier.shuffle(&mut rng);
        let raw = collect_candidates_rollout::<N>(&sim, sim_player, all_pieces, &used[sim_player.0], &piece_order, &shuffled_frontier, max_cand);

        if raw.is_empty() {
            passes += 1;
            if passes >= 4 * all_players.len() { break; }
            sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
            continue;
        }

        // 用模擬後的棋盤做 scoring（RolloutBoard → Board）
        let eval_board = sim.to_board::<N>();
        let mut scored: Vec<(usize, usize, i32, i32, i32)> = raw.iter().map(|&(pi, vi, x, y)| {
            let variant = &all_pieces[pi].variants[vi];
            let s = evaluate::count_corner_contacts(&eval_board, variant, x, y, sim_player) * 15
                + variant.cells.len() as i32 * 10;
            (pi, vi, x, y, s)
        }).collect();
        scored.sort_by(|a, b| b.4.cmp(&a.4));

        let temp = if progress < 0.2 { 0.7 }
                   else { 1.5 - progress * 1.2 };
        let scores: Vec<i32> = scored.iter().map(|s| s.4).collect();
        let idx = evaluate::temperature_sample(&scores, temp.max(0.01), &mut rng);
        let (pi, vi, x, y, _) = scored[idx];
        let variant = &all_pieces[pi].variants[vi];
        // RolloutBoard place_piece
        for &(dx, dy) in &variant.cells {
            let ax = x + dx; let ay = y + dy;
            if ax >= 0 && ax < N as i32 && ay >= 0 && ay < N as i32 {
                sim.cells[ay as usize][ax as usize] = sim_player.0 as u8 + 1;
                sim.row_masks[ay as usize] |= 1u32 << (ax as usize);
            }
        }
        update_frontier_rollout(&sim, &mut frontier, variant, x, y);
        used[sim_player.0].push(pi);
        steps[sim_player.0] += 1;
        passes = 0;
        total_occupied = sim.occupied_count();

        sim_player = PlayerId((sim_player.0 + 1) % all_players.len());
    }

    let eval_board = sim.to_board::<N>();
    let progress = evaluate::compute_progress(total_occupied, evaluate::TOTAL_PIECE_AREA);
    let root_rem: Vec<PieceShape> = all_pieces.iter()
        .enumerate()
        .filter(|(i, _)| !used[root_player.0].contains(i))
        .map(|(_, s)| s.clone())
        .collect();
    let placed_val: f32 = used[root_player.0].iter()
        .map(|&pi| (all_pieces[pi].base.cells.len() as f32).powf(1.3))
        .sum();
    let mut result = evaluate::rollout_heuristic_evaluate(&eval_board, root_player, all_players, Some(&root_rem), progress, placed_val);
    if nn_value >= 0.0 {
        const NN_BLEND: f32 = 0.3;
        result = result * (1.0 - NN_BLEND) + nn_value * NN_BLEND;
    }
    (result, steps[root_player.0])
}
