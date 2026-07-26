use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::ui::picking_backend::UiPickingCamera;
use bevy::window::WindowResolution;

use diagonal_war::state::{handle_ai_turn, AiTimer, GameConfig, GameResource};
use diagonal_war::ui::board::BoardPlugin;
use diagonal_war::ui::hud::HudPlugin;
use diagonal_war::ui::menu::MenuPlugin;
use diagonal_war::ui::panel::PanelPlugin;
use diagonal_war::ui::styles::{WINDOW_HEIGHT, WINDOW_WIDTH};

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Diagonal War".into(),
                        resolution: WindowResolution::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32),
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        backends: Some(Backends::DX12),
                        ..default()
                    })),
                    ..default()
                })
                .set(LogPlugin {
                    filter: "info,bevy_text=info,wgpu_core=warn,wgpu_hal=warn,cosmic_text=warn,icu_segmenter=error".into(),
                    ..default()
                }),
        )
        .insert_resource(GameResource::new(&GameConfig::default()))
        .insert_resource(GameConfig::default())
        .insert_resource(AiTimer::default())
        .add_plugins((BoardPlugin, PanelPlugin, HudPlugin, MenuPlugin))
        .add_systems(Startup, setup_camera)
        .add_systems(Update, handle_ai_turn)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, UiPickingCamera));
}
