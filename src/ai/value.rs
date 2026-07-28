use std::path::Path;

use std::sync::{Arc, OnceLock, atomic::{AtomicU64, Ordering}};

use dashmap::DashMap;
use tract_onnx::prelude::*;

use crate::game::board::{Board, CellState};

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
        ValueNetwork::new("model/value_unified.onnx")
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

    /// 評估盤面 Value + Policy（雙頭網路）
    ///
    /// ONNX 輸出順序：output[0]=value, output[1]=policy
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
                // output[0] = value
                let value = match outputs[0].clone().into_tensor().to_plain_array_view::<f32>() {
                    Ok(v) => v.as_slice().unwrap_or(&[0.5])[0].clamp(0.0, 1.0),
                    Err(_) => 0.5,
                };
                // output[1] = policy（若存在）
                let policy = if outputs.len() > 1 {
                    match outputs[1].clone().into_tensor().to_plain_array_view::<f32>() {
                        Ok(v) => v.as_slice().unwrap_or(&[]).to_vec(),
                        Err(_) => vec![],
                    }
                } else {
                    vec![]
                };
                (value, policy)
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
