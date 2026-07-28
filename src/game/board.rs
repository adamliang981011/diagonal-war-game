use std::hash::{Hash, Hasher};

use crate::game::piece::PieceVariant;
use crate::game::player::PlayerId;

/// 棋盤上每個格子的狀態
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Empty,
    Occupied(PlayerId),
}

impl Hash for CellState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            CellState::Empty => 0u8.hash(state),
            CellState::Occupied(pid) => {
                1u8.hash(state);
                pid.0.hash(state);
            }
        }
    }
}

/// 放置錯誤類型
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("超出棋盤邊界")]
    OutOfBounds,
    #[error("與現有棋子重疊")]
    Overlap,
    #[error("同色棋子邊接觸")]
    EdgeContact,
    #[error("未與同色棋子角接觸（非第一步）")]
    NoCornerContact,
    #[error("第一步必須覆蓋起始角")]
    MustCoverStartingCorner,
}

// ============================================================
// GameBoard trait — 通用棋盤介面
// ============================================================

pub trait GameBoard: Clone + Send + Sync {
    fn is_in_bounds(&self, x: i32, y: i32) -> bool;
    fn get_cell(&self, x: i32, y: i32) -> CellState;
    fn board_hash(&self) -> u64;
    fn is_valid(&self, variant: &PieceVariant, x: i32, y: i32, player: PlayerId, is_first: bool, corner: Option<Corner>) -> Result<(), PlacementError>;
    fn place_piece(&mut self, variant: &PieceVariant, x: i32, y: i32, player: PlayerId);
    fn try_place(&mut self, variant: &PieceVariant, x: i32, y: i32, player: PlayerId, is_first: bool, corner: Option<Corner>) -> Result<(), PlacementError>;
}

// ============================================================
// Board<const N: usize> — 方板實作
// ============================================================

/// 泛型棋盤
#[derive(Debug, Clone)]
pub struct Board<const N: usize> {
    pub cells: [[CellState; N]; N],
}

/// 玩家的起始角位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corner {
    TopLeft,     // (0, 0)
    TopRight,    // (0, N-1)
    BottomRight, // (N-1, N-1)
    BottomLeft,  // (N-1, 0)
}

impl Corner {
    pub fn position<const N: usize>(&self) -> (i32, i32) {
        match self {
            Corner::TopLeft => (0, 0),
            Corner::TopRight => (N as i32 - 1, 0),
            Corner::BottomRight => (N as i32 - 1, N as i32 - 1),
            Corner::BottomLeft => (0, N as i32 - 1),
        }
    }
}

impl<const N: usize> Hash for Board<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for row in self.cells.iter() {
            for cell in row.iter() {
                cell.hash(state);
            }
        }
    }
}

impl<const N: usize> Board<N> {
    /// 快速盤面雜湊值（用於開局書查詢）
    pub fn board_hash(&self) -> u64 {
        use std::hash::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
    /// 建立空棋盤
    pub fn new() -> Self {
        Self {
            cells: [[CellState::Empty; N]; N],
        }
    }

    /// 檢查座標是否在棋盤內
    pub fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < N as i32 && y >= 0 && y < N as i32
    }

    /// 取得格子狀態
    pub fn get_cell(&self, x: i32, y: i32) -> CellState {
        if self.is_in_bounds(x, y) {
            self.cells[y as usize][x as usize]
        } else {
            CellState::Empty // 視為空，用於邊界檢查
        }
    }

    /// 檢查某個 PieceVariant 在給定位置是否合法放置
    pub fn is_valid(
        &self,
        variant: &PieceVariant,
        pos_x: i32,
        pos_y: i32,
        player: PlayerId,
        is_first_move: bool,
        starting_corner: Option<Corner>,
    ) -> Result<(), PlacementError> {
        let absolute_cells: Vec<(i32, i32)> = variant
            .cells
            .iter()
            .map(|&(dx, dy)| (pos_x + dx, pos_y + dy))
            .collect();

        // 1. 邊界檢查
        for &(x, y) in &absolute_cells {
            if !self.is_in_bounds(x, y) {
                return Err(PlacementError::OutOfBounds);
            }
        }

        // 2. 重疊檢查
        for &(x, y) in &absolute_cells {
            if self.cells[y as usize][x as usize] != CellState::Empty {
                return Err(PlacementError::Overlap);
            }
        }

        if is_first_move {
            // 第一步：必須覆蓋起始角
            if let Some(corner) = starting_corner {
                let (cx, cy) = corner.position::<N>();
                if !absolute_cells.contains(&(cx, cy)) {
                    return Err(PlacementError::MustCoverStartingCorner);
                }
            }
            // 第一步跳過角/邊接觸檢查
            Ok(())
        } else {
            // 第一步之後的放置規則
            self.validate_placement_rules(&absolute_cells, player)
        }
    }

    /// 驗證非第一步的放置規則：角接觸 + 邊分離
    fn validate_placement_rules(
        &self,
        cells: &[(i32, i32)],
        player: PlayerId,
    ) -> Result<(), PlacementError> {
        let mut has_corner_contact = false;

        for &(x, y) in cells {
            // 檢查 4 個正交方向（邊接觸）
            for &(nx, ny) in &[(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if self.get_cell(nx, ny) == CellState::Occupied(player) {
                    return Err(PlacementError::EdgeContact);
                }
            }

            // 檢查 4 個對角方向（角接觸）
            if !has_corner_contact {
                for &(nx, ny) in &[
                    (x + 1, y + 1),
                    (x + 1, y - 1),
                    (x - 1, y + 1),
                    (x - 1, y - 1),
                ] {
                    if self.get_cell(nx, ny) == CellState::Occupied(player) {
                        has_corner_contact = true;
                        break;
                    }
                }
            }
        }

        if !has_corner_contact {
            return Err(PlacementError::NoCornerContact);
        }

        Ok(())
    }

    /// 放置棋子（不檢查合法性，需先呼叫 is_valid）
    pub fn place_piece(
        &mut self,
        variant: &PieceVariant,
        pos_x: i32,
        pos_y: i32,
        player: PlayerId,
    ) {
        for &(dx, dy) in &variant.cells {
            let x = pos_x + dx;
            let y = pos_y + dy;
            self.cells[y as usize][x as usize] = CellState::Occupied(player);
        }
    }

    /// 嘗試放置棋子（檢查 + 放置）
    pub fn try_place(
        &mut self,
        variant: &PieceVariant,
        pos_x: i32,
        pos_y: i32,
        player: PlayerId,
        is_first_move: bool,
        starting_corner: Option<Corner>,
    ) -> Result<(), PlacementError> {
        self.is_valid(variant, pos_x, pos_y, player, is_first_move, starting_corner)?;
        self.place_piece(variant, pos_x, pos_y, player);
        Ok(())
    }
}

impl<const N: usize> Default for Board<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> GameBoard for Board<N> {
    fn is_in_bounds(&self, x: i32, y: i32) -> bool { self.is_in_bounds(x, y) }
    fn get_cell(&self, x: i32, y: i32) -> CellState { self.get_cell(x, y) }
    fn board_hash(&self) -> u64 { self.board_hash() }
    fn is_valid(&self, var: &PieceVariant, x: i32, y: i32, p: PlayerId, f: bool, c: Option<Corner>) -> Result<(), PlacementError> { self.is_valid(var, x, y, p, f, c) }
    fn place_piece(&mut self, var: &PieceVariant, x: i32, y: i32, p: PlayerId) { self.place_piece(var, x, y, p) }
    fn try_place(&mut self, var: &PieceVariant, x: i32, y: i32, p: PlayerId, f: bool, c: Option<Corner>) -> Result<(), PlacementError> { self.try_place(var, x, y, p, f, c) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board_is_empty() {
        let board: Board<20> = Board::new();
        for y in 0..20 {
            for x in 0..20 {
                assert_eq!(board.cells[y][x], CellState::Empty);
            }
        }
    }

    #[test]
    fn test_bounds_checking() {
        let board: Board<10> = Board::new();
        assert!(board.is_in_bounds(0, 0));
        assert!(board.is_in_bounds(9, 9));
        assert!(!board.is_in_bounds(-1, 0));
        assert!(!board.is_in_bounds(0, 10));
    }

    #[test]
    fn test_first_move_must_cover_corner() {
        let mut board: Board<20> = Board::new();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 放在左上角（應成功，玩家 A 起始角是左上）
        assert!(board
            .try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft))
            .is_ok());

        // 重新開始，放在其他地方（應失敗）
        let mut board2: Board<20> = Board::new();
        assert_eq!(
            board2.try_place(&monomino, 5, 5, player, true, Some(Corner::TopLeft)),
            Err(PlacementError::MustCoverStartingCorner)
        );
    }

    #[test]
    fn test_cannot_overlap() {
        let mut board: Board<20> = Board::new();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 第一步放置成功
        board.try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft)).unwrap();

        // 第二步：放另一格，合法（角接觸）
        let result = board.try_place(&monomino, 1, 1, player, false, None);
        assert!(result.is_ok());

        // 在已佔據的位置上重疊放置
        let result = board.try_place(&monomino, 0, 0, player, false, None);
        assert_eq!(result, Err(PlacementError::Overlap));
    }

    #[test]
    fn test_edge_contact_forbidden() {
        let mut board: Board<20> = Board::new();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 第一步
        board.try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft)).unwrap();

        // 第二步：在 (1,0) 邊接觸（禁止）
        let result = board.try_place(&monomino, 1, 0, player, false, None);
        assert_eq!(result, Err(PlacementError::EdgeContact));
    }

    #[test]
    fn test_corner_contact_required() {
        let mut board: Board<20> = Board::new();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 第一步
        board.try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft)).unwrap();

        // 第二步：放在 (2,2)，與 (0,0) 沒有角接觸（距離太遠）
        let result = board.try_place(&monomino, 2, 2, player, false, None);
        assert_eq!(result, Err(PlacementError::NoCornerContact));
    }

    #[test]
    fn test_corner_contact_works() {
        let mut board: Board<20> = Board::new();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 第一步
        board.try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft)).unwrap();

        // 第二步：放在 (1,1)，與 (0,0) 角接觸
        let result = board.try_place(&monomino, 1, 1, player, false, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_different_players_no_restrictions() {
        let mut board: Board<10> = Board::new();
        let player_a = PlayerId(0);
        let player_b = PlayerId(1);
        let mono = PieceVariant::new(vec![(0, 0)]);

        // A 從左上 (0,0) 開始，B 從右上 (9,0) 開始
        board.try_place(&mono, 0, 0, player_a, true, Some(Corner::TopLeft)).unwrap();
        board.try_place(&mono, 9, 0, player_b, true, Some(Corner::TopRight)).unwrap();

        // A 沿對角線擴張：(1,1),(2,2),(3,3),(4,4)
        for i in 1..=4 {
            board.try_place(&mono, i, i, player_a, false, None).unwrap();
        }

        // B 沿對角線擴張：(8,1),(7,2),(6,3)
        for i in 1..=3 {
            board.try_place(&mono, 9 - i, i, player_b, false, None).unwrap();
        }

        // B 放 (5,4)：與 B 的 (6,3) 角接觸，
        // 同時與 A 的 (4,4) 邊接觸（不同玩家 → 合法）
        let result = board.try_place(&mono, 5, 4, player_b, false, None);
        assert!(
            result.is_ok(),
            "不同玩家邊接觸應合法，但得到錯誤：{:?}",
            result
        );
    }

    #[test]
    fn test_first_move_must_cover_corner_by_player_index() {
        use crate::game::player::starting_corner_for_player;

        let mut board: Board<20> = Board::new();
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 玩家 A (index 0): 左上角 (0,0)
        let corner_a = starting_corner_for_player(0, 4);
        assert!(board.try_place(&monomino, 0, 0, PlayerId(0), true, corner_a).is_ok());

        // 玩家 B (index 1): 右上角 (19,0)
        let corner_b = starting_corner_for_player(1, 4);
        assert!(board.try_place(&monomino, 19, 0, PlayerId(1), true, corner_b).is_ok());

        // 玩家 C (index 2): 右下角 (19,19)
        let corner_c = starting_corner_for_player(2, 4);
        assert!(board.try_place(&monomino, 19, 19, PlayerId(2), true, corner_c).is_ok());

        // 玩家 D (index 3): 左下角 (0,19)
        let corner_d = starting_corner_for_player(3, 4);
        assert!(board.try_place(&monomino, 0, 19, PlayerId(3), true, corner_d).is_ok());
    }
}
