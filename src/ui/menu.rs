use bevy::picking::Pickable;
use bevy::prelude::*;

use crate::state::{GameConfig, GamePhase, GameResource};
use crate::ui::panel::PiecePreviewRoot;
use crate::ui::styles::*;

#[derive(Component)]
pub struct MenuRoot;

#[derive(Component)]
pub struct PlayerCountButton {
    pub count: usize,
}

#[derive(Component)]
pub struct AiToggleButton {
    pub player_index: usize,
}

#[derive(Component)]
pub struct StartGameButton;

#[derive(Component)]
pub struct PlayerSlotRoot {
    pub player_index: usize,
}

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, show_menu_screen)
            .add_systems(Update, button_style_system)
            .add_systems(Update, handle_menu_clicks);
    }
}

fn show_menu_screen(
    game: Res<GameResource>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    existing: Query<Entity, With<MenuRoot>>,
    config: Res<GameConfig>,
) {
    if game.phase != GamePhase::Menu {
        for entity in &existing {
            commands.entity(entity).despawn();
        }
        return;
    }
    if !existing.is_empty() {
        return;
    }

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
            BackgroundColor(Color::srgb(0.1, 0.1, 0.15)),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Diagonal War"),
                TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(56.0), ..default() },
                TextColor(Color::srgb(0.8, 0.8, 1.0)),
                TextLayout { justify: Justify::Center, ..default() },
            ));

            parent.spawn((Node { height: Val::Px(20.0), ..default() }, MenuRoot));

            parent.spawn((
                Text::new("玩家人數"),
                TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(24.0), ..default() },
                TextColor(Color::WHITE),
                TextLayout { justify: Justify::Center, ..default() },
            ));

            parent.spawn((Node { height: Val::Px(12.0), ..default() }, MenuRoot));

            let counts = [2, 3, 4];
            parent
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(12.0),
                        ..default()
                    },
                ))
                .with_children(|row| {
                    for &c in &counts {
                        row.spawn((
                            Button,
                            Pickable::default(),
                            Interaction::default(),
                            Node {
                                width: Val::Px(60.0),
                                height: Val::Px(40.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
                            PlayerCountButton { count: c },
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new(format!("{}", c)),
                                TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(20.0), ..default() },
                                TextColor(Color::WHITE),
                                TextLayout { justify: Justify::Center, ..default() },
                            ));
                        });
                    }
                });

            parent.spawn((Node { height: Val::Px(20.0), ..default() }, MenuRoot));

            for i in 0..4 {
                let visible = i < config.player_count;
                parent
                    .spawn((
                        Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(16.0),
                            padding: UiRect::vertical(Val::Px(4.0)),
                            ..default()
                        },
                        if visible { Visibility::Visible } else { Visibility::Hidden },
                        PlayerSlotRoot { player_index: i },
                    ))
                    .with_children(|row| {
                        let ai_label = if config.is_ai.get(i).copied().unwrap_or(false) { "AI" } else { "人類" };
                        row.spawn((
                            Text::new(format!("玩家 {}", i + 1)),
                            TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(22.0), ..default() },
                            TextColor(player_color(i)),
                            TextLayout { justify: Justify::Center, ..default() },
                        ));
                        row.spawn((
                            Button,
                            Pickable::default(),
                            Interaction::default(),
                            Node {
                                width: Val::Px(100.0),
                                height: Val::Px(36.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                            AiToggleButton { player_index: i },
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new(ai_label),
                                TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(16.0), ..default() },
                                TextColor(Color::WHITE),
                                TextLayout { justify: Justify::Center, ..default() },
                            ));
                        });
                    });
            }

            parent.spawn((Node { height: Val::Px(24.0), ..default() }, MenuRoot));

            parent
                .spawn((
                    Button,
                    Pickable::default(),
                    Interaction::default(),
                    Node {
                        width: Val::Px(220.0),
                        height: Val::Px(60.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.0, 0.6, 0.3)),
                    StartGameButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("開始遊戲"),
                        TextFont { font: FontSource::Handle(f.clone()), font_size: FontSize::Px(28.0), ..default() },
                        TextColor(Color::WHITE),
                        TextLayout { justify: Justify::Center, ..default() },
                    ));
                });
        });
}

fn button_style_system(
    mut query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>, Without<PiecePreviewRoot>)>,
) {
    for (interaction, mut color) in &mut query {
        match *interaction {
            Interaction::Pressed => {
                *color = BackgroundColor(Color::srgb(0.4, 0.4, 0.4));
            }
            Interaction::Hovered => {
                *color = BackgroundColor(Color::srgb(0.5, 0.5, 0.5));
            }
            Interaction::None => {
                *color = BackgroundColor(Color::srgb(0.3, 0.3, 0.3));
            }
        }
    }
}

fn handle_menu_clicks(
    mut game: ResMut<GameResource>,
    mut config: ResMut<GameConfig>,
    interaction_query: Query<(&Interaction, &PlayerCountButton), Changed<Interaction>>,
    toggle_query: Query<(&Interaction, &AiToggleButton), Changed<Interaction>>,
    start_query: Query<(&Interaction, &StartGameButton), Changed<Interaction>>,
    mut text_query: Query<&mut Text>,
    children_query: Query<&Children>,
    ai_button_query: Query<(Entity, &AiToggleButton)>,
    mut slot_query: Query<(&PlayerSlotRoot, &mut Visibility)>,
) {
    for (interaction, btn) in &interaction_query {
        if *interaction == Interaction::Pressed {
            config.player_count = btn.count;
            config.is_ai.resize(btn.count, false);
            for (slot, mut vis) in &mut slot_query {
                *vis = if slot.player_index < btn.count {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }

    for (interaction, toggle) in &toggle_query {
        if *interaction == Interaction::Pressed {
            let idx = toggle.player_index;
            if idx < config.player_count {
                config.is_ai[idx] = !config.is_ai[idx];
                for (entity, atb) in &ai_button_query {
                    if atb.player_index == idx {
                        if let Ok(children) = children_query.get(entity) {
                            if let Some(&child) = children.first() {
                                if let Ok(mut text) = text_query.get_mut(child) {
                                    *text = Text::new(if config.is_ai[idx] { "AI" } else { "人類" });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (interaction, _) in &start_query {
        if *interaction == Interaction::Pressed {
            *game = GameResource::new(&config);
            game.start_game();
            break;
        }
    }
}
