use crate::ai::evaluate;
use crate::ai::{list_legal_moves, AiMove};
use crate::game::board::{Board, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

/// 貪婪 AI：對所有合法放置評分，選出最佳者
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
                        let score = evaluate::score_placement(board, variant, x, y, player);

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

/// 貪婪 AI + Temperature：依 softmax 權重隨機選取
pub fn choose_move_with_temp<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    remaining_pieces: &[PieceShape],
    is_first_move: bool,
    starting_corner: Option<Corner>,
    temp: f32,
) -> Option<AiMove> {
    let moves = list_legal_moves(board, player, remaining_pieces, is_first_move, starting_corner);
    if moves.is_empty() {
        return None;
    }
    let scores: Vec<i32> = moves.iter().map(|(_, _, _, _, s)| *s).collect();
    let mut rng = rand::rng();
    let idx = evaluate::temperature_sample(&scores, temp, &mut rng);
    let (pi, vi, x, y, sc) = moves[idx];
    Some(AiMove { piece_index: pi, variant_index: vi, x, y, score: sc })
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
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_chooses_largest_piece_first() {
        let board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let result = choose_move(&board, PlayerId(0), &pieces, true, Some(Corner::TopLeft));
        assert!(result.is_some());
        let mv = result.unwrap();
        assert_eq!(
            pieces[mv.piece_index].base.cells.len(),
            6,
            "AI should prefer the largest piece (hexomino) for first move"
        );
    }

    #[test]
    fn test_ai_returns_none_when_no_moves() {
        let board: Board<3> = Board::new();
        let pieces = create_all_pieces();
        let result = choose_move(&board, PlayerId(0), &pieces, false, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_ai_finds_second_move() {
        let mut board: Board<20> = Board::new();
        let pieces = create_all_pieces();
        let mono = PieceVariant::new(vec![(0, 0)]);

        board.try_place(&mono, 0, 0, PlayerId(0), true, Some(Corner::TopLeft)).unwrap();
        let remaining: Vec<PieceShape> = pieces.into_iter().filter(|p| p.id.0 != 0).collect();
        let result = choose_move(&board, PlayerId(0), &remaining, false, None);
        assert!(result.is_some());
    }
}
