// AI 強度整合測試
// 使用方式：
//   cargo test --test ai_tournament -- --nocapture
//   cargo test test_ai_lv2_beats_lv1 -- --nocapture

use diagonal_war::ai::choose_move as ai_choose;
use diagonal_war::ai::AiDifficulty;
use diagonal_war::game::board::{Board, CellState, Corner};
use diagonal_war::game::piece_library;
use diagonal_war::game::player::PlayerId;

/// 簡化版 AI vs AI 對戰（只跑有限步數，非完整遊戲）
fn simulate_partial<const N: usize>(
    ai_a: AiDifficulty,
    ai_b: AiDifficulty,
    max_steps: usize,
) -> f32 {
    let all_pieces = piece_library::create_all_pieces();
    let mut board: Board<N> = Board::new();

    let mut step = 0;
    let mut players = [true, true]; // player 0 (ai_a), player 1 (ai_b)

    loop {
        let player_idx = step % 2;
        let remaining = if player_idx == 0 { &all_pieces } else { &all_pieces };
        let diff = if player_idx == 0 { ai_a } else { ai_b };
        let pid = PlayerId(player_idx as usize);
        let is_first = !board.cells.iter().flatten().any(|&c| CellState::Occupied(pid) == c);
        let corner = match player_idx { 0 => Corner::TopLeft, _ => Corner::TopRight };

        let mv = ai_choose(&board, pid, remaining, is_first, Some(corner), diff);
        if let Some(m) = mv {
            let variant = &remaining[m.piece_index].variants[m.variant_index];
            board.place_piece(variant, m.x, m.y, pid);
        } else {
            players[player_idx] = false;
            if !players[0] && !players[1] { break; }
        }

        step += 1;
        if step >= max_steps * 2 { break; }
    }

    // 計算玩家 0 的優勢
    let owned_a = board.cells.iter().flatten()
        .filter(|&&c| c == CellState::Occupied(PlayerId(0)))
        .count() as f32;
    let owned_b = board.cells.iter().flatten()
        .filter(|&&c| c == CellState::Occupied(PlayerId(1)))
        .count() as f32;
    let total = owned_a + owned_b;
    if total == 0.0 { 0.5 } else { owned_a / total }
}

/// 驗證 Lv2 Greedy 勝率高於 Lv1 Random
#[test]
fn test_ai_lv2_beats_lv1() {
    let mut a_wins = 0u32;
    let games = 10;
    for _ in 0..games {
        let score = simulate_partial::<20>(AiDifficulty::Greedy, AiDifficulty::Random, 15);
        if score > 0.5 { a_wins += 1; }
    }
    // Lv2 Greedy 應該要明顯勝過 Lv1 Random
    assert!(a_wins as f32 / games as f32 > 0.6,
        "Lv2 Greedy 對 Lv1 Random 勝率應 >60%，實際 {}/{}={}%",
        a_wins, games, a_wins as f32 / games as f32 * 100.0);
}

/// 驗證 AI 至少能正常放置第一步
#[test]
fn test_all_levels_can_place_first_move() {
    let levels = [
        AiDifficulty::Random,
        AiDifficulty::Greedy,
        AiDifficulty::GreedyWithTemp(0.5),
        AiDifficulty::Search1Ply,
        AiDifficulty::Search2Ply,
    ];
    let all_pieces = piece_library::create_all_pieces();
    let board: Board<20> = Board::new();

    for (i, diff) in levels.iter().enumerate() {
        let pid = PlayerId(i);
        let corner = match i { 0 => Corner::TopLeft, 1 => Corner::TopRight, 2 => Corner::BottomRight, _ => Corner::BottomLeft };
        let result = ai_choose(&board, pid, &all_pieces, true, Some(corner), *diff);
        assert!(result.is_some(), "Level {} (AI {:?}) 無法選擇第一步", i + 1, diff);
    }
}
