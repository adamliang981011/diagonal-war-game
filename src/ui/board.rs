use bevy::prelude::*;

use crate::game::board::CellState;
use crate::state::{GamePhase, GameResource};
use crate::ui::styles::*;

#[derive(Component)]
pub struct BoardCell {
    pub x: usize,
    pub y: usize,
}

#[derive(Component)]
pub struct GhostCell;

pub struct BoardPlugin;

impl Plugin for BoardPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_board)
            .add_systems(Update, toggle_board_visibility)
            .add_systems(Update, update_board_colors)
            .add_systems(Update, update_ghost_piece)
            .add_systems(Update, handle_placement_click);
    }
}

fn toggle_board_visibility(
    game: Res<GameResource>,
    mut query: Query<&mut Visibility, With<BoardCell>>,
) {
    let visible = game.phase != GamePhase::Menu;
    for mut vis in &mut query {
        *vis = if visible { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn spawn_board(mut commands: Commands) {
    const BOARD_SIZE: usize = 20;
    let total = CELL_SIZE + CELL_GAP;
    let start_x = BOARD_OFFSET_X - (BOARD_SIZE as f32 * total) / 2.0 + CELL_SIZE / 2.0;
    let start_y = BOARD_OFFSET_Y + (BOARD_SIZE as f32 * total) / 2.0 - CELL_SIZE / 2.0;

    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            let px = start_x + x as f32 * total;
            let py = start_y - y as f32 * total;
            commands.spawn((
                Sprite::from_color(EMPTY_CELL_COLOR, Vec2::new(CELL_SIZE, CELL_SIZE)),
                Transform::from_xyz(px, py, 0.0),
                BoardCell { x, y },
            ));
        }
    }
}

fn update_board_colors(
    game: Res<GameResource>,
    mut query: Query<(&BoardCell, &mut Sprite)>,
) {
    if !game.is_changed() {
        return;
    }
    for (cell, mut sprite) in &mut query {
        let cell_state = game.board.cells[cell.y][cell.x];
        sprite.color = match cell_state {
            CellState::Empty => EMPTY_CELL_COLOR,
            CellState::Occupied(pid) => player_color(pid.0),
        };
    }
}

fn update_ghost_piece(
    game: Res<GameResource>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut commands: Commands,
    ghost_query: Query<Entity, With<GhostCell>>,
    mut last_state: Local<Option<(i32, i32, usize)>>,
) {
    let new_state = if game.phase == GamePhase::Placing {
        cursor_grid_pos(&windows, &cameras).and_then(|(gx, gy)| {
            game.selection.as_ref().map(|s| (gx, gy, s.variant_index))
        })
    } else {
        None
    };

    if *last_state == new_state {
        return;
    }
    *last_state = new_state;

    for entity in &ghost_query {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }

    let Some((gx, gy)) = cursor_grid_pos(&windows, &cameras) else {
        return;
    };
    let Some(variant) = game.selected_variant() else {
        return;
    };

    let is_valid = game
        .board
        .is_valid(
            &variant,
            gx,
            gy,
            game.current_player_id(),
            game.is_first_move(),
            game.starting_corner(),
        )
        .is_ok();

    let color = if is_valid {
        GHOST_VALID_COLOR
    } else {
        GHOST_INVALID_COLOR
    };

    if let Some((start_x, start_y, total)) = board_layout() {
        for &(dx, dy) in &variant.cells {
            let px = start_x + (gx + dx) as f32 * total;
            let py = start_y - (gy + dy) as f32 * total;
            commands.spawn((
                Sprite::from_color(color, Vec2::new(CELL_SIZE, CELL_SIZE)),
                Transform::from_xyz(px, py, 1.0),
                GhostCell,
            ));
        }
    }
}

fn handle_placement_click(
    mut game: ResMut<GameResource>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
) {
    if game.phase != GamePhase::Placing {
        return;
    }
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(variant) = game.selected_variant() else {
        return;
    };
    let Some((gx, gy)) = cursor_grid_pos(&windows, &cameras) else {
        return;
    };

    if game
        .board
        .is_valid(
            &variant,
            gx,
            gy,
            game.current_player_id(),
            game.is_first_move(),
            game.starting_corner(),
        )
        .is_ok()
    {
        let pid = game.current_player_id();
        let shape_idx = game.selection.as_ref().unwrap().piece_index;
        let shape_id = game.players[game.current_player].remaining_pieces[shape_idx].id;
        game.board.place_piece(&variant, gx, gy, pid);
        game.current_player_mut().remove_piece(shape_id);
        game.current_player_mut().has_placed_first_piece = true;
        game.selection = None;
        game.advance_turn();
    }
}

fn cursor_grid_pos(
    windows: &Query<&Window>,
    cameras: &Query<(&Camera, &GlobalTransform)>,
) -> Option<(i32, i32)> {
    let Ok(window) = windows.single() else {
        return None;
    };
    let Ok((camera, cam_transform)) = cameras.single() else {
        return None;
    };
    let cursor = window.cursor_position()?;
    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, cursor) else {
        return None;
    };

    let (start_x, start_y, total) = board_layout()?;
    let half_cell = CELL_SIZE / 2.0;
    const BOARD_SIZE: f32 = 20.0;
    if world_pos.x >= start_x - half_cell
        && world_pos.x <= start_x + BOARD_SIZE * total - half_cell
        && world_pos.y <= start_y + half_cell
        && world_pos.y >= start_y - BOARD_SIZE * total + half_cell
    {
        let gx = ((world_pos.x - start_x + CELL_SIZE / 2.0) / total) as i32;
        let gy = ((start_y - world_pos.y + CELL_SIZE / 2.0) / total) as i32;
        Some((gx, gy))
    } else {
        None
    }
}

fn board_layout() -> Option<(f32, f32, f32)> {
    const BOARD_SIZE: f32 = 20.0;
    let total = CELL_SIZE + CELL_GAP;
    let start_x = BOARD_OFFSET_X - (BOARD_SIZE * total) / 2.0 + CELL_SIZE / 2.0;
    let start_y = BOARD_OFFSET_Y + (BOARD_SIZE * total) / 2.0 - CELL_SIZE / 2.0;
    Some((start_x, start_y, total))
}
