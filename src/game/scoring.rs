use crate::game::player::Player;

/// 單一玩家的計分結果
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerScore {
    pub player_index: usize,
    pub remaining_squares: usize,
    pub elimination_turn: Option<usize>,
}

/// 最終排名
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameResult {
    pub rankings: Vec<PlayerScore>,
}

/// 計算排名：
/// 1. 未淘汰者（elimination_turn = None）排最前
/// 2. 淘汰者依淘汰順序（越晚淘汰排名越高）
/// 3. 同時淘汰者比剩餘格數（越少越高）
pub fn calculate_rankings(players: &[Player]) -> GameResult {
    let mut scores: Vec<PlayerScore> = players
        .iter()
        .enumerate()
        .map(|(i, p)| PlayerScore {
            player_index: i,
            remaining_squares: p.remaining_squares(),
            elimination_turn: p.elimination_turn,
        })
        .collect();

    scores.sort_by(|a, b| match (a.elimination_turn, b.elimination_turn) {
        (None, None) => a.remaining_squares.cmp(&b.remaining_squares),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some(a_t), Some(b_t)) => b_t
            .cmp(&a_t)
            .then(a.remaining_squares.cmp(&b.remaining_squares)),
    });

    GameResult { rankings: scores }
}

/// 判斷勝者
pub fn winner(result: &GameResult) -> Option<usize> {
    result.rankings.first().map(|s| s.player_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::piece::PieceShape;

    fn create_test_piece(size: usize) -> PieceShape {
        let cells: Vec<(i32, i32)> = (0..size as i32).map(|i| (0, i)).collect();
        PieceShape::new(crate::game::piece::PieceId(size), "Test", cells)
    }

    #[test]
    fn test_winner_is_last_survivor() {
        let pieces = vec![create_test_piece(2)];
        let players = vec![
            Player {
                id: crate::game::player::PlayerId(0),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(0),
            },
            Player {
                id: crate::game::player::PlayerId(1),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(1),
            },
            Player {
                id: crate::game::player::PlayerId(2),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: false,
                elimination_turn: None,
            },
        ];

        let result = calculate_rankings(&players);
        assert_eq!(result.rankings.len(), 3);
        assert_eq!(result.rankings[0].player_index, 2); // 未淘汰者第一
    }

    #[test]
    fn test_ranking_by_elimination_order() {
        let pieces = vec![create_test_piece(2)];
        let players = vec![
            Player {
                id: crate::game::player::PlayerId(0),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(0),
            },
            Player {
                id: crate::game::player::PlayerId(1),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(1),
            },
            Player {
                id: crate::game::player::PlayerId(2),
                remaining_pieces: vec![pieces[0].clone()],
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(2),
            },
        ];

        let result = calculate_rankings(&players);
        // 越晚淘汰排名越高
        assert_eq!(result.rankings[0].player_index, 2); // 最後淘汰
        assert_eq!(result.rankings[1].player_index, 1);
        assert_eq!(result.rankings[2].player_index, 0); // 最早淘汰
    }

    #[test]
    fn test_same_elimination_turn_tie_by_squares() {
        let pieces = vec![create_test_piece(5), create_test_piece(3)];
        let players = vec![
            Player {
                id: crate::game::player::PlayerId(0),
                remaining_pieces: vec![pieces[0].clone()], // 5 格
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(0),
            },
            Player {
                id: crate::game::player::PlayerId(1),
                remaining_pieces: vec![pieces[1].clone()], // 3 格
                has_placed_first_piece: true,
                passed: true,
                elimination_turn: Some(0),
            },
        ];

        let result = calculate_rankings(&players);
        assert_eq!(result.rankings[0].player_index, 1); // 格數少者排前
        assert_eq!(result.rankings[1].player_index, 0);
    }
}
