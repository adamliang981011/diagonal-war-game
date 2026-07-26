use bevy::prelude::*;

// === 版面尺寸 ===
pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const CELL_SIZE: f32 = 32.0;
pub const CELL_GAP: f32 = 1.0;
pub const BOARD_OFFSET_X: f32 = -160.0;
pub const BOARD_OFFSET_Y: f32 = 0.0;
pub const PANEL_WIDTH: f32 = 280.0;

// === 玩家顏色（色盲友善調色盤） ===
pub const PLAYER_COLORS: [Color; 4] = [
    Color::srgb(0.00, 0.45, 0.74),  // 藍
    Color::srgb(0.85, 0.33, 0.10),  // 橙
    Color::srgb(0.47, 0.67, 0.19),  // 綠
    Color::srgb(0.90, 0.76, 0.00),  // 黃
];

// === 棋盤顏色 ===
pub const EMPTY_CELL_COLOR: Color = Color::srgb(0.90, 0.90, 0.90);
pub const GRID_LINE_COLOR: Color = Color::srgb(0.70, 0.70, 0.70);
pub const GHOST_VALID_COLOR: Color = Color::srgba(0.00, 1.00, 0.00, 0.40);
pub const GHOST_INVALID_COLOR: Color = Color::srgba(1.00, 0.00, 0.00, 0.40);

// === UI 文字 ===
pub fn player_color(index: usize) -> Color {
    PLAYER_COLORS[index % PLAYER_COLORS.len()]
}
