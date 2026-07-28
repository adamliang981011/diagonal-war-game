use bevy::picking::Pickable;
use bevy::prelude::*;

use crate::game::scoring;
use crate::state::{GameConfig, GamePhase, GameResource};
use crate::ui::styles::*;

#[derive(Component)]
pub struct GameOverOverlay;

#[derive(Component)]
pub struct ControlsText;

#[derive(Component)]
pub struct RestartButton;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_hud)
            .add_systems(Update, toggle_controls_visibility)
            .add_systems(Update, update_controls_text)
            .add_systems(Update, handle_rotation_input)
            .add_systems(Update, check_pass_auto)
            .add_systems(Update, show_game_over)
            .add_systems(Update, handle_restart_click);
    }
}

fn spawn_hud(mut commands: Commands, asset_server: Res<AssetServer>) {
    let f: Handle<Font> = asset_server.load("fonts/NotoSansTC-Variable.ttf");
    commands.spawn((
        Text::new(""),
        TextFont { font: FontSource::Handle(f), font_size: FontSize::Px(14.0), ..default() },
        TextColor(Color::WHITE),
        TextLayout { justify: Justify::Center, ..default() },
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(10.0),
            bottom: Val::Px(10.0),
            ..default()
        },
        ControlsText,
    ));
}

fn update_controls_text(
    game: Res<GameResource>,
    mut query: Query<&mut Text, With<ControlsText>>,
    mut last: Local<String>,
) {
    let Ok(mut text) = query.single_mut() else {
        return;
    };
    let new_text = match game.phase {
        GamePhase::Menu => "",
        GamePhase::Selecting => "選擇棋子：點擊右側面板的棋子",
        GamePhase::Placing => "[R] 旋轉  [M] 鏡像  左鍵放置  右鍵取消",
        GamePhase::TurnTransition => "",
        GamePhase::GameOver => "遊戲結束！",
    };
    if *last == new_text {
        return;
    }
    *last = new_text.to_string();
    *text = Text::new(new_text);
}

fn toggle_controls_visibility(
    game: Res<GameResource>,
    mut query: Query<&mut Visibility, With<ControlsText>>,
) {
    if let Ok(mut vis) = query.single_mut() {
        *vis = if game.phase == GamePhase::Menu {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
}

fn handle_rotation_input(
    mut game: ResMut<GameResource>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if game.phase != GamePhase::Placing {
        return;
    }
    let piece_index = match game.selection.as_ref().map(|s| s.piece_index) {
        Some(idx) => idx,
        None => return,
    };
    let variant_count = game.players[game.current_player].remaining_pieces[piece_index]
        .variants
        .len();

    if keys.just_pressed(KeyCode::KeyR) {
        if let Some(ref mut sel) = game.selection {
            sel.variant_index = (sel.variant_index + 1) % variant_count;
        }
    }
    if keys.just_pressed(KeyCode::KeyM) {
        if let Some(ref mut sel) = game.selection {
            let half = variant_count / 2;
            sel.variant_index = (sel.variant_index + half) % variant_count;
        }
    }
}

fn check_pass_auto(mut game: ResMut<GameResource>) {
    if game.phase != GamePhase::Selecting {
        return;
    }
    if !game.check_player_has_moves() {
        game.eliminate_current_player();
        if game.active_player_count() <= 1 {
            game.phase = GamePhase::GameOver;
        } else {
            game.advance_turn();
        }
    }
}

fn show_game_over(
    game: Res<GameResource>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    existing: Query<Entity, With<GameOverOverlay>>,
) {
    if !existing.is_empty() {
        return;
    }
    if game.phase != GamePhase::GameOver {
        return;
    }

    let result = scoring::calculate_rankings(&game.players);
    let f: Handle<Font> = asset_server.load("fonts/NotoSansTC-Variable.ttf");

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.9)),
            GameOverOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("遊戲結束！"),
                TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(48.0), ..default() },
                TextColor(Color::WHITE),
                TextLayout { justify: Justify::Center, ..default() },
            ));
            parent.spawn((Node { height: Val::Px(16.0), ..default() },));

            let medals = ["1st", "2nd", "3rd", "4th"];
            for (rank, score) in result.rankings.iter().enumerate() {
                let color = player_color(score.player_index);
                parent.spawn((
                    Text::new(format!(
                        "{}  玩家 {}  —  剩餘 {} 格",
                        medals.get(rank).unwrap_or(&"  "),
                        score.player_index + 1,
                        score.remaining_squares
                    )),
                    TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(28.0), ..default() },
                    TextColor(color),
                    TextLayout { justify: Justify::Center, ..default() },
                ));
            }
            parent.spawn((Node { height: Val::Px(24.0), ..default() },));
            parent
                .spawn((
                    Button,
                    Pickable::default(),
                    Interaction::default(),
                    Node {
                        width: Val::Px(220.0),
                        height: Val::Px(50.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
                    RestartButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("回到主選單"),
                        TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(22.0), ..default() },
                        TextColor(Color::WHITE),
                        TextLayout { justify: Justify::Center, ..default() },
                    ));
                });
        });
}

fn handle_restart_click(
    mut game: ResMut<GameResource>,
    config: Res<GameConfig>,
    interaction_query: Query<&Interaction, (With<RestartButton>, Changed<Interaction>)>,
    mut commands: Commands,
    overlay_query: Query<Entity, With<GameOverOverlay>>,
) {
    for interaction in &interaction_query {
        if *interaction == Interaction::Pressed {
            *game = GameResource::new(&config);
            game.phase = GamePhase::Menu;
            for entity in &overlay_query {
                if let Ok(mut ec) = commands.get_entity(entity) { ec.despawn(); }
            }
        }
    }
}