use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::settings::{Backends, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::ui::picking_backend::UiPickingCamera;
use bevy::window::WindowResolution;

use diagonal_war::ai::opening_book::OpeningBook;
use diagonal_war::ai::train::train_depth_3;
use diagonal_war::game::piece_library;
use diagonal_war::state::{handle_ai_turn, AiTimer, GameConfig, GameResource, OpeningBookResource, SearchResource};
use diagonal_war::ai::SearchState;
use diagonal_war::ui::board::BoardPlugin;
use diagonal_war::ui::hud::HudPlugin;
use diagonal_war::ui::menu::MenuPlugin;
use diagonal_war::ui::panel::PanelPlugin;
use diagonal_war::ui::styles::{WINDOW_HEIGHT, WINDOW_WIDTH};

fn main() {
    // 啟動時載入或訓練開局書
    let opening_book = load_or_train_book();

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
                    filter: "info,bevy_text=info,bevy_ecs::error=error,wgpu_core=warn,wgpu_hal=warn,cosmic_text=warn,icu_segmenter=error".into(),
                    ..default()
                }),
        )
        .insert_resource(GameResource::new(&GameConfig::default()))
        .insert_resource(GameConfig::default())
        .insert_resource(AiTimer::default())
        .insert_resource(OpeningBookResource { book: opening_book })
        .insert_resource(SearchResource { state: Some(SearchState { tree: None }) })
        .add_plugins((BoardPlugin, PanelPlugin, HudPlugin, MenuPlugin))
        .add_systems(Startup, setup_camera)
        .add_systems(Update, handle_ai_turn)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, UiPickingCamera));
}

/// 嘗試載入開局書，不存在則訓練
fn load_or_train_book() -> OpeningBook {
    let path = "assets/opening_book.json";

    // 嘗試載入
    if let Ok(book) = OpeningBook::load(path) {
        println!("[OpeningBook] 已載入 {} 個 entry（{}）", book.len(), path);
        return book;
    }

    println!("[OpeningBook] 未找到已訓練的開局書，開始訓練...");

    // 確保 assets/ 目錄存在
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let pieces = piece_library::create_all_pieces();
    match train_depth_3(&pieces, path) {
        Ok(book) => {
            println!("[OpeningBook] 訓練完成：{} 個 entry", book.len());
            book
        }
        Err(e) => {
            eprintln!("[OpeningBook] 訓練失敗：{}，使用空白開局書", e);
            OpeningBook::new()
        }
    }
}
