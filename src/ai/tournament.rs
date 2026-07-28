use crate::ai::{AiDifficulty, choose_move};
use crate::game::board::{Board, CellState};
use crate::game::piece_library;
use crate::game::player::{PlayerId, starting_corner_for_player};

/// 對戰報告
pub struct BattleReport {
    pub wins_a: usize,
    pub wins_b: usize,
    pub draws: usize,
    pub elo_a: f32,
    pub elo_b: f32,
    pub avg_turns: f32,
    pub avg_remaining_a: f32,
    pub avg_remaining_b: f32,
    pub avg_mobility_a: f32,
    pub avg_mobility_b: f32,
}

/// AI A vs AI B 多局對戰
pub fn battle(
    player_count: usize,
    a_diff: AiDifficulty,
    a_player_index: usize,
    b_diff: AiDifficulty,
    b_player_index: usize,
    games: usize,
) -> BattleReport {
    let pieces = piece_library::create_all_pieces();
    let mut wins_a = 0usize;
    let mut wins_b = 0usize;
    let mut draws = 0usize;
    let mut total_turns = 0usize;
    let mut remaining_a_sum = 0usize;
    let mut remaining_b_sum = 0usize;
    let mut mobility_a_sum = 0usize;
    let mut mobility_b_sum = 0usize;

    for _ in 0..games {
        let mut board: Board<20> = Board::new();
        let mut current_player = 0usize;
        let mut eliminated = vec![false; player_count];
        let mut elim_turn = vec![None; player_count];
        let mut placed_first = vec![false; player_count];
        let mut remaining: Vec<Vec<usize>> = (0..player_count)
            .map(|_| (0..pieces.len()).collect())
            .collect();
        let mut turn_count = 0usize;

        loop {
            // 跳過已淘汰玩家
            if eliminated[current_player] {
                current_player = (current_player + 1) % player_count;
                continue;
            }

            // 檢查是否有合法步
            let pid = PlayerId(current_player);
            let is_first = !placed_first[current_player];
            let corner = starting_corner_for_player(current_player, player_count);
            let rem_shapes: Vec<_> = remaining[current_player].iter()
                .map(|&i| pieces[i].clone()).collect();

            let has_move = crate::game::rules::player_has_legal_move(
                &board, pid, &rem_shapes, is_first, corner,
            );

            if !has_move {
                eliminated[current_player] = true;
                elim_turn[current_player] = Some(turn_count);
                let active: usize = eliminated.iter().filter(|&&e| !e).count();
                if active <= 1 {
                    // 遊戲結束
                    let winner = eliminated.iter().position(|&e| !e);
                    match winner {
                        Some(w) if w == a_player_index => wins_a += 1,
                        Some(w) if w == b_player_index => wins_b += 1,
                        _ => draws += 1,
                    }
                    total_turns += turn_count;
                    remaining_a_sum += remaining[a_player_index].len();
                    remaining_b_sum += remaining[b_player_index].len();
                    // 終局 mobility（估算）
                    mobility_a_sum += estimate_mobility_for(&board, PlayerId(a_player_index));
                    mobility_b_sum += estimate_mobility_for(&board, PlayerId(b_player_index));
                    break;
                }
                current_player = (current_player + 1) % player_count;
                continue;
            }

            // AI 選擇步
            let diff = if current_player == a_player_index { a_diff }
                       else if current_player == b_player_index { b_diff }
                       else { AiDifficulty::Greedy };

            let mv = choose_move::<20>(
                &board, pid, &rem_shapes, is_first, corner, diff, player_count,
            );

            if let Some(mv) = mv {
                // mv.piece_index 是 rem_shapes 內的索引
                let global_idx = rem_shapes[mv.piece_index].id.0;
                let variant = pieces[global_idx].variants[mv.variant_index].clone();
                board.place_piece(&variant, mv.x, mv.y, pid);
                let idx = remaining[current_player].iter()
                    .position(|&i| i == global_idx).unwrap();
                remaining[current_player].remove(idx);
                placed_first[current_player] = true;
            }

            turn_count += 1;
            current_player = (current_player + 1) % player_count;
        }
    }

    let games_f = games as f32;
    let elo_k = 32.0;
    let expected_a = 1.0 / (1.0 + 10.0_f32.powf(0.0)); // 初始 Elo 相同 → 0.5
    let elo_a = elo_k * (wins_a as f32 - expected_a * games_f);

    BattleReport {
        wins_a, wins_b, draws,
        elo_a,
        elo_b: -elo_a,
        avg_turns: total_turns as f32 / games_f,
        avg_remaining_a: remaining_a_sum as f32 / games_f,
        avg_remaining_b: remaining_b_sum as f32 / games_f,
        avg_mobility_a: mobility_a_sum as f32 / games_f,
        avg_mobility_b: mobility_b_sum as f32 / games_f,
    }
}

fn estimate_mobility_for<const N: usize>(board: &Board<N>, player: PlayerId) -> usize {
    let mut count = 0;
    for y in 0..N { for x in 0..N {
        if board.cells[y][x] == CellState::Occupied(player) {
            for (nx, ny) in &[(x as i32 + 1, y as i32), (x as i32 - 1, y as i32),
                              (x as i32, y as i32 + 1), (x as i32, y as i32 - 1)] {
                if *nx >= 0 && *nx < N as i32 && *ny >= 0 && *ny < N as i32
                    && board.cells[*ny as usize][*nx as usize] == CellState::Empty {
                    count += 1;
                }
            }
        }
    }}
    count
}

pub fn print_report(report: &BattleReport, label_a: &str, label_b: &str, n: usize) {
    println!("===== AI Tournament: {} vs {} ({} games) =====", label_a, label_b, n);
    println!("Win Rate:  {} {:.1}%  |  {} {:.1}%  |  Draw {:.1}%",
        label_a, report.wins_a as f32 / n as f32 * 100.0,
        label_b, report.wins_b as f32 / n as f32 * 100.0,
        report.draws as f32 / n as f32 * 100.0);
    println!("Elo Diff:  {} {:.0}  |  {} {:.0}",
        label_a, report.elo_a, label_b, report.elo_b);
    println!("Avg Turns: {:.1}", report.avg_turns);
    println!("Avg Remaining Pieces: {} {:.1}  |  {} {:.1}",
        label_a, report.avg_remaining_a, label_b, report.avg_remaining_b);
    println!("Avg Mobility (frontier): {} {:.1}  |  {} {:.1}",
        label_a, report.avg_mobility_a, label_b, report.avg_mobility_b);
    println!("==================================================");
}
