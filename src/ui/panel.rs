use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;
use bevy::ui::ScrollPosition;

use crate::state::{GamePhase, GameResource};
use crate::ui::styles::*;

#[derive(Component)]
pub struct PanelBackground;

#[derive(Component)]
pub struct PiecePanelButton {
    pub piece_index: usize,
}

#[derive(Component)]
pub struct PiecePreviewRoot;

#[derive(Component)]
pub struct PanelContent;

pub struct PanelPlugin;

impl Plugin for PanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_panel_bg)
            .add_systems(Update, toggle_panel_visibility)
            .add_systems(Update, update_piece_panel)
            .add_systems(Update, handle_panel_click)
            .add_systems(Update, handle_scroll_events.before(update_piece_panel));
    }
}

fn toggle_panel_visibility(
    game: Res<GameResource>,
    mut query: Query<&mut Visibility, With<PanelBackground>>,
) {
    let visible = game.phase != GamePhase::Menu;
    for mut vis in &mut query {
        *vis = if visible { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn spawn_panel_bg(mut commands: Commands, asset_server: Res<AssetServer>) {
    let f: Handle<Font> = asset_server.load("fonts/NotoSansTC-Variable.ttf");
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                top: Val::Px(10.0),
                width: Val::Px(PANEL_WIDTH),
                height: Val::Px(WINDOW_HEIGHT - 20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
            PanelBackground,
        ))
        .with_child((
            Text::new("Diagonal War"),
            TextFont { font: FontSource::Handle(f), font_size: FontSize::Px(24.0), ..default() },
            TextColor(Color::WHITE),
            TextLayout { justify: Justify::Center, ..default() },
        ));
}

fn update_piece_panel(
    game: Res<GameResource>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    panel_query: Query<Entity, With<PanelBackground>>,
    existing: Query<Entity, With<PanelContent>>,
    mut prev_state: Local<(usize, usize)>,
) {
    if game.phase != GamePhase::Selecting && game.phase != GamePhase::Placing {
        return;
    }
    let Ok(panel_entity) = panel_query.single() else {
        return;
    };

    let player = game.current_player_ref();
    let state = (game.current_player, player.remaining_squares());
    if *prev_state == state {
        return;
    }
    *prev_state = state;

    for entity in &existing {
        commands.entity(entity).despawn();
    }

    let player = game.current_player_ref();
    let color_idx = player.id.0;
    let f: Handle<Font> = asset_server.load("fonts/NotoSansTC-Variable.ttf");

    let mini_size = 10.0;
    let gap = 2.0;
    let cell_total = mini_size + gap;
    let grid_cells = 5;
    let grid_size = grid_cells as f32 * cell_total;
    let card_w = (PANEL_WIDTH - 16.0 - 8.0 - 4.0) / 2.0;
    let card_content_w = card_w - 8.0;
    let card_h = grid_size + 8.0;
    let grid_off_x = (card_content_w - grid_size) / 2.0;

    commands.entity(panel_entity).with_children(|parent| {
        parent.spawn((
            Text::new(format!("玩家 {} 的回合", color_idx + 1)),
            TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(18.0), ..default() },
            TextColor(player_color(color_idx)),
            TextLayout { justify: Justify::Center, ..default() },
            PanelContent,
        ));
        parent.spawn((Node { height: Val::Px(8.0), ..default() }, PanelContent));

        parent
            .spawn((
                Node {
                    width: Val::Px(PANEL_WIDTH - 16.0),
                    height: Val::Px(WINDOW_HEIGHT - 60.0),
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(4.0),
                    row_gap: Val::Px(4.0),
                    overflow: Overflow::scroll_y(),
                    align_content: AlignContent::FlexStart,
                    ..default()
                },
                PanelContent,
            ))
            .with_children(|grid| {
                for (i, shape) in player.remaining_pieces.iter().enumerate() {
                    let variant = &shape.base;
                    let ox = ((grid_cells - variant.width) / 2) as i32;
                    let oy = ((grid_cells - variant.height) / 2) as i32;
                    grid
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(card_w),
                                height: Val::Px(card_h),
                                flex_shrink: 0.0,
                                padding: UiRect::all(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            PiecePanelButton { piece_index: i },
                            PiecePreviewRoot,
                ScrollPosition::default(),
                PanelContent,
                        ))
                        .with_children(|btn| {
                            for &(dx, dy) in &variant.cells {
                                btn.spawn((
                                    Node {
                                        position_type: PositionType::Absolute,
                                        left: Val::Px(grid_off_x + (dx + ox) as f32 * cell_total),
                                        top: Val::Px((dy + oy) as f32 * cell_total),
                                        width: Val::Px(mini_size),
                                        height: Val::Px(mini_size),
                                        ..default()
                                    },
                                    BackgroundColor(player_color(color_idx)),
                                ));
                            }
                        });
                }
            });
    });
}

fn handle_panel_click(
    mut game: ResMut<GameResource>,
    interaction_query: Query<(&Interaction, &PiecePanelButton), Changed<Interaction>>,
) {
    for (interaction, btn) in &interaction_query {
        if *interaction == Interaction::Pressed {
            game.selection = Some(crate::state::SelectionState {
                piece_index: btn.piece_index,
                variant_index: 0,
            });
            game.phase = GamePhase::Placing;
        }
    }
}

fn handle_scroll_events(
    scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<&mut ScrollPosition>,
) {
    if scroll.delta.y != 0.0 {
        for mut pos in &mut query {
            pos.y = (pos.y - scroll.delta.y * 80.0).max(0.0);
        }
    }
}
