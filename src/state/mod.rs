use bevy::prelude::*;

use crate::ai::greedy::{AiMove, choose_move};
use crate::game::board::Board;
use crate::game::piece_library;
use crate::game::player::{starting_corner_for_player, Player, PlayerId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    Menu,
    Selecting,
    Placing,
    TurnTransition,
    GameOver,
}

#[derive(Debug, Clone)]
pub struct SelectionState {
    pub piece_index: usize,
    pub variant_index: usize,
}

#[derive(Resource)]
pub struct GameResource {
    pub board: Board<20>,
    pub players: Vec<Player>,
    pub current_player: usize,
    pub phase: GamePhase,
    pub selection: Option<SelectionState>,
    pub elimination_counter: usize,
}

#[derive(Resource)]
pub struct GameConfig {
    pub player_count: usize,
    pub is_ai: Vec<bool>,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            player_count: 2,
            is_ai: vec![false, false],
        }
    }
}

#[derive(Resource)]
pub struct AiTimer {
    pub timer: Timer,
    pub active: bool,
    pub computed_move: Option<AiMove>,
}

impl Default for AiTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.6, TimerMode::Once),
            active: false,
            computed_move: None,
        }
    }
}

impl GameResource {
    pub fn new(config: &GameConfig) -> Self {
        let pieces = piece_library::create_all_pieces();
        let players: Vec<Player> = (0..config.player_count)
            .map(|i| Player::new(PlayerId(i), &pieces))
            .collect();
        Self {
            board: Board::new(),
            players,
            current_player: 0,
            phase: GamePhase::Menu,
            selection: None,
            elimination_counter: 0,
        }
    }

    pub fn start_game(&mut self) {
        self.phase = GamePhase::Selecting;
        self.current_player = 0;
        self.elimination_counter = 0;
        for player in &mut self.players {
            player.passed = false;
            player.elimination_turn = None;
        }
    }

    pub fn current_player_id(&self) -> PlayerId {
        PlayerId(self.current_player)
    }

    pub fn current_player_ref(&self) -> &Player {
        &self.players[self.current_player]
    }

    pub fn current_player_mut(&mut self) -> &mut Player {
        &mut self.players[self.current_player]
    }

    pub fn is_first_move(&self) -> bool {
        !self.players[self.current_player].has_placed_first_piece
    }

    pub fn starting_corner(&self) -> Option<crate::game::board::Corner> {
        starting_corner_for_player(self.current_player, self.players.len())
    }

    pub fn selected_variant(&self) -> Option<crate::game::piece::PieceVariant> {
        self.selection.as_ref().map(|sel| {
            let shape = &self.players[self.current_player].remaining_pieces[sel.piece_index];
            shape.variants[sel.variant_index].clone()
        })
    }

    pub fn eliminate_current_player(&mut self) {
        let turn = self.elimination_counter;
        self.elimination_counter += 1;
        let player = self.current_player_mut();
        player.passed = true;
        player.elimination_turn = Some(turn);
    }

    pub fn advance_turn(&mut self) {
        self.selection = None;
        self.phase = GamePhase::Selecting;
        let n = self.players.len();
        for _ in 0..n {
            self.current_player = (self.current_player + 1) % n;
            if !self.players[self.current_player].passed {
                return;
            }
        }
    }

    pub fn active_player_count(&self) -> usize {
        self.players.iter().filter(|p| !p.passed).count()
    }

    pub fn check_player_has_moves(&self) -> bool {
        let player = self.current_player_ref();
        if player.remaining_pieces.is_empty() {
            return false;
        }
        crate::game::rules::player_has_legal_move(
            &self.board,
            player.id,
            &player.remaining_pieces,
            self.is_first_move(),
            self.starting_corner(),
        )
    }
}

/// AI 回合處理系統
pub fn handle_ai_turn(
    mut game: ResMut<GameResource>,
    config: Res<GameConfig>,
    mut ai_timer: ResMut<AiTimer>,
    time: Res<Time>,
) {
    if game.phase != GamePhase::Selecting {
        ai_timer.active = false;
        ai_timer.computed_move = None;
        return;
    }

    let curr = game.current_player;
    if curr >= config.is_ai.len() || !config.is_ai[curr] {
        ai_timer.active = false;
        return;
    }

    if !ai_timer.active {
        ai_timer.active = true;
        ai_timer.timer.reset();

        let remaining = game.players[curr].remaining_pieces.clone();
        let pid = game.current_player_id();
        let is_first = game.is_first_move();
        let corner = game.starting_corner();
        let board_copy = game.board.clone();

        let mv = choose_move(&board_copy, pid, &remaining, is_first, corner);
        ai_timer.computed_move = mv;
    }

    ai_timer.timer.tick(time.delta());
    if !ai_timer.timer.just_finished() {
        return;
    }

    if let Some(mv) = ai_timer.computed_move.take() {
        let variant = game.players[curr].remaining_pieces[mv.piece_index].variants[mv.variant_index].clone();
        let pid = game.current_player_id();
        let shape_id = game.players[curr].remaining_pieces[mv.piece_index].id;
        game.board.place_piece(&variant, mv.x, mv.y, pid);
        game.current_player_mut().remove_piece(shape_id);
        game.current_player_mut().has_placed_first_piece = true;
        game.advance_turn();
    } else {
        game.eliminate_current_player();
        if game.active_player_count() <= 1 {
            game.phase = GamePhase::GameOver;
        } else {
            game.advance_turn();
        }
    }
    ai_timer.active = false;
}
