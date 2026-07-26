use rand::Rng;

use crate::game::board::{Board, CellState};
use crate::game::player::PlayerId;

/// Zobrist Hash 表：400 cells × 5 種狀態（Empty + 4 players）
const CELLS: usize = 400;
const STATES: usize = 5;

fn cell_state_index(state: CellState) -> usize {
    match state {
        CellState::Empty => 0,
        CellState::Occupied(PlayerId(0)) => 1,
        CellState::Occupied(PlayerId(1)) => 2,
        CellState::Occupied(PlayerId(2)) => 3,
        CellState::Occupied(PlayerId(3)) => 4,
        _ => 0,
    }
}

pub struct ZobristTable {
    table: [[u64; STATES]; CELLS],
}

impl ZobristTable {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let mut table = [[0u64; STATES]; CELLS];
        for cell in table.iter_mut() {
            for state in cell.iter_mut() {
                *state = rng.random();
            }
        }
        Self { table }
    }

    /// 計算完整盤面的 hash（O(N²)）
    pub fn hash<const N: usize>(&self, board: &Board<N>) -> u64 {
        let mut h = 0;
        for y in 0..N {
            for x in 0..N {
                let state = board.cells[y][x];
                let idx = y * N + x;
                h ^= self.table[idx][cell_state_index(state)];
            }
        }
        h
    }

    /// 增量更新 hash（O(1)）
    pub fn update(&self, old_hash: u64, x: i32, y: i32, old_state: CellState, new_state: CellState) -> u64 {
        let idx = (y as usize) * 20 + (x as usize); // N=20
        let old_val = self.table[idx][cell_state_index(old_state)];
        let new_val = self.table[idx][cell_state_index(new_state)];
        old_hash ^ old_val ^ new_val
    }
}
