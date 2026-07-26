use crate::game::board::Corner;
use crate::game::piece::PieceShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub usize);

#[derive(Debug, Clone)]
pub struct Player {
    pub id: PlayerId,
    pub remaining_pieces: Vec<PieceShape>,
    pub has_placed_first_piece: bool,
    pub passed: bool,
    pub elimination_turn: Option<usize>,
}

impl Player {
    pub fn new(id: PlayerId, all_pieces: &[PieceShape]) -> Self {
        Self {
            id,
            remaining_pieces: all_pieces.to_vec(),
            has_placed_first_piece: false,
            passed: false,
            elimination_turn: None,
        }
    }

    pub fn remove_piece(&mut self, piece_id: crate::game::piece::PieceId) {
        self.remaining_pieces.retain(|p| p.id != piece_id);
    }

    pub fn reset_pass(&mut self) {
        self.passed = false;
    }

    pub fn remaining_squares(&self) -> usize {
        self.remaining_pieces
            .iter()
            .map(|p| p.base.cells.len())
            .sum()
    }
}

pub fn starting_corner_for_player(player_index: usize, total_players: usize) -> Option<Corner> {
    match (player_index, total_players) {
        (0, _) => Some(Corner::TopLeft),
        (1, _) => Some(Corner::TopRight),
        (2, _) => Some(Corner::BottomRight),
        (3, _) => Some(Corner::BottomLeft),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::piece_library::create_all_pieces;

    #[test]
    fn test_new_player_has_all_pieces() {
        let pieces = create_all_pieces();
        let player = Player::new(PlayerId(0), &pieces);
        assert_eq!(player.remaining_pieces.len(), 21);
        assert!(!player.has_placed_first_piece);
        assert!(!player.passed);
        assert_eq!(player.elimination_turn, None);
    }

    #[test]
    fn test_remove_piece() {
        let pieces = create_all_pieces();
        let mut player = Player::new(PlayerId(0), &pieces);
        let monomino_id = pieces[0].id;
        player.remove_piece(monomino_id);
        assert_eq!(player.remaining_pieces.len(), 20);
    }

    #[test]
    fn test_remaining_squares() {
        let pieces = create_all_pieces();
        let player = Player::new(PlayerId(0), &pieces);
        assert_eq!(player.remaining_squares(), 89);
    }

    #[test]
    fn test_starting_corners() {
        assert_eq!(starting_corner_for_player(0, 4), Some(Corner::TopLeft));
        assert_eq!(starting_corner_for_player(1, 4), Some(Corner::TopRight));
        assert_eq!(starting_corner_for_player(2, 4), Some(Corner::BottomRight));
        assert_eq!(starting_corner_for_player(3, 4), Some(Corner::BottomLeft));
    }
}
