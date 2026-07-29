use std::path::Path;

use std::sync::{Arc, OnceLock, atomic::{AtomicU64, Ordering}};

use dashmap::DashMap;
use tract_onnx::prelude::*;

use crate::game::board::{Board, CellState};
/// Candidate move for scoring network
#[derive(Debug, Clone)]
pub struct Candidate {
    pub piece: u8,
    pub variant: u8,
    pub x: i8,
    pub y: i8,
    pub piece_size: u8,
    pub heuristic_prior: f32,
}

/// Result of evaluate_with_candidates()
pub struct PolicyResult {
    pub value: f32,
    pub priors: Vec<f32>,
}

/// NN 推論結果快取（key = board hash × player_count）
static NN_CACHE: OnceLock<DashMap<u64, f32>> = OnceLock::new();
/// 快取命中統計
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

fn get_cache() -> &'static DashMap<u64, f32> {
    NN_CACHE.get_or_init(|| DashMap::with_capacity(65536))
}

/// 清除 NN 快取
pub fn clear_nn_cache() {
    get_cache().clear();
    CACHE_HITS.store(0, Ordering::Relaxed);
    CACHE_MISSES.store(0, Ordering::Relaxed);
}

/// 顯示快取命中率
pub fn print_cache_stats() {
    let hits = CACHE_HITS.load(Ordering::Relaxed);
    let misses = CACHE_MISSES.load(Ordering::Relaxed);
    let total = hits + misses;
    if total > 0 {
        eprintln!("NN cache: {hits} hits, {misses} misses, {:.1}% hit rate", hits as f64 / total as f64 * 100.0);
    }
}

/// 計算 board 的 hash（用於快取 key）
fn board_hash<const N: usize>(board: &Board<N>, player_count: usize) -> u64 {
    let mut h = 0x517cc1b727220a95u64;
    for y in 0..N.min(20) {
        for x in 0..N.min(20) {
            let cell = match board.cells[y][x] {
                CellState::Empty => 0u8,
                CellState::Occupied(p) => p.0 as u8 + 1,
            };
            h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(cell as u64);
        }
    }
    h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(player_count as u64);
    h
}

/// 全域的 ValueNetwork（僅初始化一次）
static VALUE_NET: OnceLock<ValueNetwork> = OnceLock::new();

/// 取得或初始化全域 ValueNetwork
pub fn get_value_network() -> &'static ValueNetwork {
    VALUE_NET.get_or_init(|| {
        let path = if Path::new("model/dual_unified.onnx").exists() {
            "model/dual_unified.onnx"
        } else {
            "model/value_unified.onnx"
        };
        ValueNetwork::new(path)
    })
}

/// ONNX Value Network 推論（使用 tract，純 Rust）
///
/// 當 model/value_unified.onnx 存在時自動載入並使用。
/// 不存在時回退到 0.5（中性值）。
pub struct ValueNetwork {
    model: Option<Arc<TypedRunnableModel>>,
}

impl ValueNetwork {
    pub fn new(model_path: &str) -> Self {
        let model = Path::new(model_path).exists().then(|| {
            eprintln!("ValueNetwork: loading from {model_path}...");
            let m = onnx()
                .model_for_path(model_path)
                .expect("Failed to load ONNX model")
                .into_optimized()
                .expect("Failed to optimize ONNX model")
                .into_runnable()
                .expect("Failed to compile ONNX model");
            eprintln!("ValueNetwork: loaded from {model_path}");
            m
        });
        Self { model }
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// 將 board 轉為 tensor 輸入用陣列
    fn board_to_input<const N: usize>(board: &Board<N>, _player: usize) -> [f32; 400] {
        let mut input = [0.0f32; 400];
        for y in 0..N.min(20) {
            for x in 0..N.min(20) {
                input[y * 20 + x] = match board.cells[y][x] {
                    CellState::Empty => 0.0,
                    CellState::Occupied(p) => (p.0 as u8 + 1) as f32,
                };
            }
        }
        input
    }

    /// 評估盤面 Value [0, 1]（對齊 MCTS Q 值慣例）
    pub fn evaluate<const N: usize>(
        &self, board: &Board<N>, player: usize, player_count: usize,
    ) -> f32 {
        // 先查快取
        let hash = board_hash(board, player_count);
        if let Some(cached) = get_cache().get(&hash) {
            CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return *cached;
        }

        let Some(model) = self.model.clone() else { return 0.5 };
        CACHE_MISSES.fetch_add(1, Ordering::Relaxed);

        // 輸入 1: board (1, 1, 20, 20), float32
        let input = Self::board_to_input(board, player);
        let board_tensor = tract_ndarray::Array4::from_shape_vec(
            (1, 1, 20, 20), input.to_vec(),
        ).unwrap();

        // 輸入 2: player_count index (1,), int64
        let pc_idx = (player_count.saturating_sub(2).min(2)) as i64;
        let pc_tensor = tract_ndarray::Array1::from_vec(vec![pc_idx]);

        // 推論
        let result = model.run(tvec!(
            board_tensor.into_tensor().into(),
            pc_tensor.into_tensor().into(),
        ));
        let value = match result {
            Ok(mut outputs) => {
                let tensor = outputs.swap_remove(0).into_tensor();
                let view = tensor.to_plain_array_view::<f32>();
                match view {
                    Ok(v) => *v.as_slice().unwrap_or(&[0.5]).first().unwrap_or(&0.5),
                    Err(_) => 0.5,
                }
            }
            Err(e) => {
                eprintln!("ValueNetwork inference error: {e}");
                0.5
            }
        };
        let value = value.clamp(0.0, 1.0);

        // 存入快取（限制大小防止記憶體爆炸）
        if get_cache().len() < 100_000 {
            get_cache().insert(hash, value);
        }
        value
    }

    /// 混合 evaluate（neural + heuristic）
    /// blend = 1.0 → 完全使用 neural
    /// blend = 0.0 → 完全使用 heuristic
    pub fn evaluate_blended<const N: usize>(
        &self, board: &Board<N>, player: usize, player_count: usize,
        heuristic_val: f32, blend: f32,
    ) -> f32 {
        if !self.is_loaded() {
            return heuristic_val;
        }
        let neural = self.evaluate(board, player, player_count);
        neural * blend + heuristic_val * (1.0 - blend)
    }

    /// Candidate Scoring Network evaluation
    ///
    /// Runs board encoder once, then scores all candidates in batch.
    /// Returns (value, priors) where priors are softmax-normalized and
    /// already blended with heuristic_prior using progress-based weights.
    pub fn evaluate_with_candidates<const N: usize>(
        &self, board: &Board<N>, candidates: &[Candidate],
        _player: usize, player_count: usize, progress: f32,
    ) -> Option<PolicyResult> {
        let model = self.model.clone()?;
        let max_n = 32;
        let n = candidates.len().min(max_n);

        if n == 0 { return None; }

        // Build board input (1, 1, 20, 20)
        let mut input = [0.0f32; 400];
        for y in 0..N.min(20) {
            for x in 0..N.min(20) {
                input[y * 20 + x] = match board.cells[y][x] {
                    CellState::Empty => 0.0,
                    CellState::Occupied(p) => (p.0 as u8 + 1) as f32,
                };
            }
        }
        let board_t = tract_ndarray::Array4::from_shape_vec((1, 1, 20, 20), input.to_vec()).unwrap();

        // Build candidate arrays (pad to 32)
        let mut pieces = vec![0i64; max_n];
        let mut variants = vec![0i64; max_n];
        let mut xs = vec![0.0f32; max_n];
        let mut ys = vec![0.0f32; max_n];
        let mut sizes = vec![0.0f32; max_n];
        for (i, c) in candidates.iter().enumerate().take(max_n) {
            pieces[i] = c.piece as i64;
            variants[i] = c.variant as i64;
            xs[i] = c.x as f32 / 19.0;
            ys[i] = c.y as f32 / 19.0;
            sizes[i] = c.piece_size as f32 / 5.0;
        }

        let pc_idx = (player_count.saturating_sub(2).min(2)) as i64;

        let result = model.run(tvec![
            board_t.into_tensor().into(),
            tract_ndarray::Array1::from_vec(vec![pc_idx]).into_tensor().into(),
            tract_ndarray::Array1::from_vec(vec![progress as f64]).into_tensor().into(),
            tract_ndarray::Array1::from_vec(pieces.clone()).into_tensor().into(),
            tract_ndarray::Array1::from_vec(variants).into_tensor().into(),
            tract_ndarray::Array1::from_vec(xs).into_tensor().into(),
            tract_ndarray::Array1::from_vec(ys).into_tensor().into(),
            tract_ndarray::Array1::from_vec(sizes).into_tensor().into(),
        ]).ok()?;

        if result.len() < 2 { return None; }

        // output[0] = value (scalar), output[1] = scores (N,)
        let value = result[1].clone().into_tensor()
            .to_plain_array_view::<f32>().ok()
            .and_then(|v| v.iter().copied().next())
            .unwrap_or(0.5).clamp(0.0, 1.0);

        let scores: Vec<f32> = result[0].clone().into_tensor()
            .to_plain_array_view::<f32>().ok()
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();

        if scores.len() < n { return None; }

        // Softmax
        let max_s = scores.iter().take(n).cloned().fold(f32::MIN, f32::max);
        let shifted: Vec<f32> = scores.iter().take(n).map(|s| (s - max_s).exp()).collect();
        let sum: f32 = shifted.iter().sum();
        let mut priors: Vec<f32> = if sum > 0.0 {
            shifted.iter().map(|s| s / sum).collect()
        } else {
            vec![1.0 / n as f32; n]
        };

        // Blend with heuristic prior
        let nn_blend = if progress < 0.3 { 0.5 }
                      else if progress < 0.7 { 0.7 }
                      else { 0.9 };
        for (p, c) in priors.iter_mut().zip(candidates.iter()) {
            *p = *p * nn_blend + c.heuristic_prior * (1.0 - nn_blend);
        }

        Some(PolicyResult { value, priors })
    }

    /// 評估盤面 Value + Policy（雙頭網路）
    ///
    /// ONNX 輸出順序：output[0]=policy(83200), output[1]=value(1)
    /// 若模型只有 value（舊版），policy 回傳空 Vec
    pub fn evaluate_policy<const N: usize>(
        &self, board: &Board<N>, player: usize, player_count: usize,
    ) -> (f32, Vec<f32>) {
        let Some(model) = self.model.clone() else { return (0.5, vec![]) };
        let input = Self::board_to_input(board, player);
        let board_tensor = tract_ndarray::Array4::from_shape_vec(
            (1, 1, 20, 20), input.to_vec(),
        ).unwrap();
        let pc_idx = (player_count.saturating_sub(2).min(2)) as i64;
        let pc_tensor = tract_ndarray::Array1::from_vec(vec![pc_idx]);
        let result = model.run(tvec!(
            board_tensor.into_tensor().into(),
            pc_tensor.into_tensor().into(),
        ));
        match result {
            Ok(outputs) => {
                if outputs.len() < 2 {
                    return (0.5, vec![]);
                }
                // output[0] = policy (1, 83200 logits), output[1] = value (scalar)
                let mut logits: Vec<f32> = outputs[0].clone().into_tensor()
                    .to_plain_array_view::<f32>()
                    .ok()
                    .map(|v| v.iter().copied().collect())
                    .unwrap_or_default();
                // Softmax: transform logits → probabilities [0, 1]
                if !logits.is_empty() {
                    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
                    let sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
                    if sum > 0.0 {
                        for l in &mut logits {
                            *l = (*l - max_logit).exp() / sum;
                        }
                    }
                }
                let value = outputs[1].clone().into_tensor()
                    .to_plain_array_view::<f32>()
                    .ok()
                    .and_then(|v| v.iter().copied().next())
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                (value, logits)
            }
            Err(e) => {
                eprintln!("ValueNetwork evaluate_policy error: {e}");
                (0.5, vec![])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_network_without_model() {
        let vn = ValueNetwork::new("nonexistent.onnx");
        assert!(!vn.is_loaded());
        let board: Board<20> = Board::new();
        let score = vn.evaluate(&board, 0, 2);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_board_to_input() {
        let mut board: Board<20> = Board::new();
        board.cells[0][0] = CellState::Occupied(crate::game::player::PlayerId(0));
        let input = ValueNetwork::board_to_input(&board, 0);
        assert!((input[0] - 1.0).abs() < 0.001); // player 0 → cell value 1
        assert!((input[1] - 0.0).abs() < 0.001); // empty → 0
    }

    #[test]
    fn test_evaluate_with_model() {
        let vn = ValueNetwork::new("model/value_unified.onnx");
        let board: Board<20> = Board::new();
        let score = vn.evaluate(&board, 0, 2);
        if vn.is_loaded() {
            assert!(score >= 0.0 && score <= 1.0);
        } else {
            assert!((score - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_blended_fallback() {
        let vn = ValueNetwork::new("nonexistent.onnx");
        let board: Board<20> = Board::new();
        // 無模型時 blend 無效，應回傳 heuristic
        let score = vn.evaluate_blended(&board, 0, 2, 0.7, 0.5);
        assert!((score - 0.7).abs() < 0.01);
    }
}
