use crate::game::board::{Board, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// 貪婪 AI 的選棋結果
#[derive(Debug, Clone)]
pub struct AiMove {
    pub piece_index: usize,
    pub variant_index: usize,
    pub x: i32,
    pub y: i32,
    pub score: i32,
}

/// 貪婪 AI：對所有合法放置評分，選出最佳者
/// 評分策略：棋子越大越好 + 角接觸越多越好
pub fn choose_move<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
) -> Option<AiMove> {
    let mut best: Option<AiMove> = None;

    for (pi, shape) in remaining_pieces.iter().enumerate() {
        for (vi, variant) in shape.variants.iter().enumerate() {
            for y in 0..N as i32 {
                for x in 0..N as i32 {
                    if x + variant.width > N as i32 || y + variant.height > N as i32 {
                        continue;
                    }
                    if board
                        .is_valid(variant, x, y, player, is_first_move, starting_corner)
                        .is_ok()
                    {
                        let contacts = count_corner_contacts(board, variant, x, y, player);
                        // 分數 = 棋子格數 × 100 + 角接觸數
                        let score = (variant.cells.len() as i32) * 100 + contacts;

                        let should_replace = match &best {
                            None => true,
                            Some(current) => {
                                score > current.score
                                    || (score == current.score && variant.cells.len() > remaining_pieces[current.piece_index].base.cells.len())
                            }
                        };

                        if should_replace {
                            best = Some(AiMove {
                                piece_index: pi,
                                variant_index: vi,
                                x,
                                y,
                                score,
                            });
                        }
                    }
                }
            }
        }
    }
    best
}

/// 計算放置後與自己棋子的角接觸數量
fn count_corner_contacts<const N: usize>(
    board: &Board<N>,
    variant: &crate::game::piece::PieceVariant,
    pos_x: i32,
    pos_y: i32,
    player: PlayerId,
) -> i32 {
    use crate::game::board::CellState;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::board::Board;
    use crate::game::piece::PieceVariant;
    use crate::game::piece_library::create_all_pieces;

    #[test]
    fn test_ai_finds_first_move() {
        let board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let result = choose_move(&board, PlayerId(0), &pieces, true, Some(Corner::TopLeft));
        assert!(result.is_some(), "AI should find a valid first move");
    }

    #[test]
    fn test_ai_chooses_largest_piece_first() {
        let board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let result = choose_move(&board, PlayerId(0), &pieces, true, Some(Corner::TopLeft));
        assert!(result.is_some());
        // 第一步應該選 pentomino（5 格），因為分數最高
        let mv = result.unwrap();
        assert_eq!(
            pieces[mv.piece_index].base.cells.len(),
            5,
            "AI should prefer the largest piece (pentomino) for first move"
        );
    }

    #[test]
    fn test_ai_returns_none_when_no_moves() {
        let board: Board<3> = Board::new();
        let pieces = create_all_pieces();
        let result = choose_move(&board, PlayerId(0), &pieces, false, None);
        // 3x3 棋盤，非第一步，沒有任何已放置的棋子 → 不可能有角接觸
        assert!(result.is_none());
    }

    #[test]
    fn test_ai_finds_second_move() {
        let mut board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let mono = PieceVariant::new(vec![(0, 0)]);

        // 先手放 (0,0)
        board.try_place(&mono, 0, 0, PlayerId(0), true, Some(Corner::TopLeft)).unwrap();

        // 移掉已用的 monomino
        let remaining: Vec<PieceShape> = pieces.into_iter().filter(|p| p.id.0 != 0).collect();

        let result = choose_move(&board, PlayerId(0), &remaining, false, None);
        assert!(result.is_some(), "AI should find a valid second move");
    }
}
