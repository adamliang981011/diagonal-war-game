use std::sync::LazyLock;

use crate::game::piece::PieceVariant;

const CENTER_EXPONENT: f32 = 1.4;

/// 20×20 開局熱力圖（中央高、角落低）
static HEATMAP: LazyLock<[f32; 400]> = LazyLock::new(|| {
    let mut m = [0.0f32; 400];
    let cx = 9.5f32;
    let cy = 9.5f32;
    let max_d = 14.0f32;
    for y in 0..20 {
        for x in 0..20 {
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            m[y * 20 + x] = (1.0 - (d / max_d).clamp(0.0, 1.0)).powf(CENTER_EXPONENT);
        }
    }
    m
});

/// 棋子的重心 Heat（用 centroid 取代格平均）
pub fn centroid_heat(variant: &PieceVariant, pos_x: i32, pos_y: i32) -> f32 {
    let n = variant.cells.len() as f32;
    if n == 0.0 { return 0.0; }
    let cx = variant.cells.iter().map(|&(dx, _)| dx).sum::<i32>() as f32 / n;
    let cy = variant.cells.iter().map(|&(_, dy)| dy).sum::<i32>() as f32 / n;
    let gx = (pos_x as f32 + cx).round().clamp(0.0, 19.0) as usize;
    let gy = (pos_y as f32 + cy).round().clamp(0.0, 19.0) as usize;
    HEATMAP[gy * 20 + gx]
}
