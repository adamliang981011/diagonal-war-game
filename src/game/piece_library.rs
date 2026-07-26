use crate::game::piece::{PieceId, PieceShape};

/// 建立所有 21 個 Blokus 風格棋子（monomino 到 pentomino）
/// 總格數：89 格
pub fn create_all_pieces() -> Vec<PieceShape> {
    vec![
        // === Monomino (1) ===
        PieceShape::new(PieceId(0), "Monomino", vec![(0, 0)]),

        // === Domino (1) ===
        PieceShape::new(PieceId(1), "Domino", vec![(0, 0), (0, 1)]),

        // === Triominoes (2) ===
        PieceShape::new(PieceId(2), "Triomino I", vec![(0, 0), (0, 1), (0, 2)]),
        PieceShape::new(PieceId(3), "Triomino L", vec![(0, 0), (1, 0), (0, 1)]),

        // === Tetrominoes (5) ===
        PieceShape::new(PieceId(4), "Tetromino O", vec![(0, 0), (1, 0), (0, 1), (1, 1)]),
        PieceShape::new(PieceId(5), "Tetromino I", vec![(0, 0), (0, 1), (0, 2), (0, 3)]),
        PieceShape::new(PieceId(6), "Tetromino T", vec![(0, 0), (0, 1), (0, 2), (1, 1)]),
        PieceShape::new(PieceId(7), "Tetromino L", vec![(0, 0), (1, 0), (2, 0), (0, 1)]),
        PieceShape::new(PieceId(8), "Tetromino S", vec![(0, 0), (1, 0), (1, 1), (2, 1)]),

        // === Pentominoes (12) ===
        // F: 3x3 不對稱形
        PieceShape::new(
            PieceId(9),
            "Pentomino F",
            vec![(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
        ),
        // I: 直線 5
        PieceShape::new(
            PieceId(10),
            "Pentomino I",
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        ),
        // L: 4 直線 + 1 尾端
        PieceShape::new(
            PieceId(11),
            "Pentomino L",
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
        ),
        // N: 2×4 鋸齒形
        PieceShape::new(
            PieceId(12),
            "Pentomino N",
            vec![(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        ),
        // P: 2x2 + 1 延伸
        PieceShape::new(
            PieceId(13),
            "Pentomino P",
            vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
        ),
        // T: T 形
        PieceShape::new(
            PieceId(14),
            "Pentomino T",
            vec![(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        ),
        // U: U 形
        PieceShape::new(
            PieceId(15),
            "Pentomino U",
            vec![(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
        ),
        // V: V 形
        PieceShape::new(
            PieceId(16),
            "Pentomino V",
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)],
        ),
        // W: 三階梯（標準）
        PieceShape::new(
            PieceId(17),
            "Pentomino W",
            vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        ),
        // X: 十字形
        PieceShape::new(
            PieceId(18),
            "Pentomino X",
            vec![(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
        ),
        // Y: 叉形（4 直線 + 中間分支）
        PieceShape::new(
            PieceId(19),
            "Pentomino Y",
            vec![(1, 0), (1, 1), (0, 2), (1, 2), (1, 3)],
        ),
        // Z: Z 形
        PieceShape::new(
            PieceId(20),
            "Pentomino Z",
            vec![(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        ),

        // === Hexominoes (5) ===
        // I6: 直線 6 — 極長進攻
        PieceShape::new(
            PieceId(21),
            "Hexomino I6",
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        ),
        // O6: 2×3 矩形 — 高效填滿防禦
        PieceShape::new(
            PieceId(22),
            "Hexomino O6",
            vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        ),
        // S6: 蛇形階梯 — 多角接觸生存
        PieceShape::new(
            PieceId(23),
            "Hexomino S6",
            vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
        ),
        // L6: 大 L — 長臂進攻 + 短臂防禦
        PieceShape::new(
            PieceId(24),
            "Hexomino L6",
            vec![(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)],
        ),
        // T6: 大 T — 多方向擴張分支
        PieceShape::new(
            PieceId(25),
            "Hexomino T6",
            vec![(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (1, 3)],
        ),
    ]
}

/// 計算所有棋子的總格數
pub fn total_piece_squares(pieces: &[PieceShape]) -> usize {
    pieces.iter().map(|p| p.base.cells.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_count() {
        let pieces = create_all_pieces();
        assert_eq!(pieces.len(), 26);
    }

    #[test]
    fn test_total_squares() {
        let pieces = create_all_pieces();
        let total = total_piece_squares(&pieces);
        assert_eq!(total, 119);
    }

    #[test]
    fn test_each_piece_has_variants() {
        let pieces = create_all_pieces();
        for piece in &pieces {
            assert!(
                !piece.variants.is_empty(),
                "Piece {} has no variants",
                piece.name
            );
        }
    }

    #[test]
    fn test_piece_id_uniqueness() {
        let pieces = create_all_pieces();
        let mut ids: Vec<PieceId> = pieces.iter().map(|p| p.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), pieces.len());
    }

    #[test]
    fn test_all_pieces_have_correct_size() {
        let pieces = create_all_pieces();
        let expected_sizes: Vec<usize> = vec![
            1, 2, 3, 3, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6,
        ];
        for (piece, expected) in pieces.iter().zip(&expected_sizes) {
            assert_eq!(
                piece.base.cells.len(),
                *expected,
                "Piece {} has wrong size",
                piece.name
            );
        }
    }

    #[test]
    fn test_no_duplicate_across_pieces() {
        use std::collections::HashSet;
        let pieces = create_all_pieces();
        let mut seen: HashSet<crate::game::piece::PieceVariant> = HashSet::new();
        for piece in &pieces {
            for variant in &piece.variants {
                assert!(
                    seen.insert(variant.clone()),
                    "重複的棋子形狀！Piece '{}' 的 variant {:?} 已在其他棋子中出現過",
                    piece.name, variant.cells
                );
            }
        }
    }
}
