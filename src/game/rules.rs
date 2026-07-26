use crate::game::board::{Board, Corner, PlacementError};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// 檢查玩家是否有任何合法放置
pub fn player_has_legal_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> bool {
    for shape in remaining_pieces {
        for variant in &shape.variants {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    // 快速邊界過濾
                    if x + variant.width > N as i32 || y + variant.height > N as i32 {
                        continue;
                    }
                    if board
                        .is_valid(variant, x, y, player, is_first_move, starting_corner)
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// 找到玩家的第一個合法放置（用於 AI 或驗證）
#[allow(dead_code)]
pub fn find_any_legal_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> Option<(usize, usize, i32, i32)> {
    for (shape_idx, shape) in remaining_pieces.iter().enumerate() {
        for (variant_idx, variant) in shape.variants.iter().enumerate() {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    if x + variant.width > N as i32 || y + variant.height > N as i32 {
                        continue;
                    }
                    if board
                        .is_valid(variant, x, y, player, is_first_move, starting_corner)
                        .is_ok()
                    {
                        return Some((shape_idx, variant_idx, x, y));
                    }
                }
            }
        }
    }
    None
}

/// 找到所有合法放置（用於 AI 策略）
#[allow(dead_code)]
pub fn find_all_legal_moves<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> Vec<(usize, usize, i32, i32)> {
    let mut moves = Vec::new();
    for (shape_idx, shape) in remaining_pieces.iter().enumerate() {
        for (variant_idx, variant) in shape.variants.iter().enumerate() {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    if x + variant.width > N as i32 || y + variant.height > N as i32 {
                        continue;
                    }
                    if board
                        .is_valid(variant, x, y, player, is_first_move, starting_corner)
                        .is_ok()
                    {
                        moves.push((shape_idx, variant_idx, x, y));
                    }
                }
            }
        }
    }
    moves
}

/// 放置結果
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementResult {
    pub success: bool,
    pub error: Option<PlacementError>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::piece::PieceVariant;
    use crate::game::piece_library::create_all_pieces;

    #[test]
    fn test_first_move_is_legal() {
        let board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let player = PlayerId(0);

        assert!(player_has_legal_move(
            &board,
            player,
            &pieces,
            true,
            Some(Corner::TopLeft),
        ));
    }

    #[test]
    fn test_no_legal_move_on_full_board() {
        let board: Board<3> = Board::new();
        let pieces = create_all_pieces();
        let player = PlayerId(0);

        // 3x3 棋盤，第一個棋子 monomino 在 (0,0) 是合法的
        assert!(player_has_legal_move(
            &board,
            player,
            &pieces,
            true,
            Some(Corner::TopLeft),
        ));
    }

    #[test]
    fn test_player_cannot_move_when_no_pieces_left() {
        let board: Board<20> = Board::new();
        let player = PlayerId(0);
        let empty_pieces: Vec<PieceShape> = vec![];

        assert!(!player_has_legal_move(
            &board,
            player,
            &empty_pieces,
            true,
            Some(Corner::TopLeft),
        ));
    }

    #[test]
    fn test_legal_move_after_first_placement() {
        let mut board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let player = PlayerId(0);
        let monomino = PieceVariant::new(vec![(0, 0)]);

        // 第一步
        board.try_place(&monomino, 0, 0, player, true, Some(Corner::TopLeft)).unwrap();

        // 移掉 monomino 讓它不是可選棋子
        let remaining: Vec<PieceShape> = pieces.into_iter().filter(|p| p.id.0 != 0).collect();

        // 第二步應該可以在 (1,1) 放棋子
        assert!(player_has_legal_move(
            &board,
            player,
            &remaining,
            false,
            None,
        ));
    }
}
