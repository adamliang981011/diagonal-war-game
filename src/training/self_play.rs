use crate::ai::tournament::BattleReport;
use crate::ai::{AiDifficulty, choose_move_with_stats, decode_action};
use crate::game::board::Board;
use crate::game::piece_library;
use crate::game::player::{PlayerId, starting_corner_for_player};
use crate::training::export::*;

/// AI vs AI 自我對弈，產生訓練資料
pub fn generate_self_play_games(
    player_count: usize,
    difficulties: &[AiDifficulty],
    games: usize,
    output_path: &str,
    random_seed: u64,
) -> Result<BattleReport, Box<dyn std::error::Error>> {
    let pieces = piece_library::create_all_pieces();
    let mut all_records = Vec::with_capacity(games);
    let mut wins = vec![0usize; player_count];
    let mut total_turns = 0usize;

    for game_id in 0..games {
        if game_id % 10 == 0 {
            eprintln!("  Game {}/{}...", game_id + 1, games);
        }
        let mut board: Board<20> = Board::new();
        let mut current_player = 0usize;
        let mut eliminated = vec![false; player_count];
        let mut placed_first = vec![false; player_count];
        let mut remaining: Vec<Vec<usize>> = (0..player_count)
            .map(|_| (0..pieces.len()).collect())
            .collect();
        let mut step_records: Vec<StepRecord> = Vec::new();
        let mut turn = 0u16;

        loop {
            if eliminated[current_player] {
                current_player = (current_player + 1) % player_count;
                continue;
            }

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
                let active: usize = eliminated.iter().filter(|&&e| !e).count();
                if active <= 1 {
                    let winner = eliminated.iter().position(|&e| !e);
                    let final_winner = match winner {
                        Some(w) => w as u8,
                        None => 255,
                    };
                    for rec in &mut step_records {
                        rec.winner = final_winner;
                    }
                    let game_record = GameRecord {
                        game_id: game_id as u64,
                        player_count: player_count as u8,
                        steps: step_records.clone(),
                        final_winner,
                    };
                    all_records.push(game_record);
                    if let Some(w) = winner { wins[w] += 1; }
                    total_turns += turn as usize;
                    break;
                }
                current_player = (current_player + 1) % player_count;
                continue;
            }

            let diff = difficulties[current_player.min(difficulties.len() - 1)];
            let result = choose_move_with_stats::<20>(
                &board, pid, &rem_shapes, is_first, corner, diff, player_count,
            );

            if let Some((mv, stats)) = result {
                // 輔助：將 local piece_index 轉為 global
                fn to_global(pi: usize, vi: usize, x: i32, y: i32, rem_shapes: &[crate::game::piece::PieceShape]) -> (u8, u8, i8, i8) {
                    let global_pi = if pi < rem_shapes.len() { rem_shapes[pi].id.0 } else { pi };
                    (global_pi as u8, vi as u8, x as i8, y as i8)
                }

                // chosen_move
                let (cm_p, cm_v, cm_x, cm_y) = to_global(mv.piece_index, mv.variant_index, mv.x, mv.y, &rem_shapes);

                // root_visits → PolicyRecord（probability = visits / total_visits）
                let total_v = stats.total_visits.max(1) as f32;
                let root_visits: Vec<PolicyRecord> = stats.visits.iter()
                    .map(|&(action, visits)| {
                        let (pi, vi, x, y) = decode_action(action);
                        let (gp, gv, gx, gy) = to_global(pi, vi, x, y, &rem_shapes);
                        PolicyRecord {
                            piece: gp, variant: gv, x: gx, y: gy,
                            probability: visits as f32 / total_v,
                        }
                    })
                    .collect();

                let record = StepRecord {
                    board: board_to_array(&board),
                    current_player: current_player as u8,
                    player_count: player_count as u8,
                    remaining_mask: remaining_to_masks(&remaining, player_count),
                    turn,
                    winner: 255,
                    mcts_value: stats.value,
                    total_visits: stats.total_visits,
                    chosen_move: VisitRecord {
                        piece: cm_p, variant: cm_v, x: cm_x, y: cm_y,
                        visits: stats.visits.iter()
                            .find(|&&(a, _)| {
                                let (pi, vi, x, y) = decode_action(a);
                                pi == mv.piece_index && vi == mv.variant_index && x == mv.x && y == mv.y
                            })
                            .map(|&(_, v)| v)
                            .unwrap_or(0),
                    },
                    root_visits,
                    game_rule_version: GAME_RULE_VERSION,
                    ai_version: AI_VERSION,
                    random_seed,
                };
                step_records.push(record);

                let global_idx = rem_shapes[mv.piece_index].id.0;
                let variant = pieces[global_idx].variants[mv.variant_index].clone();
                board.place_piece(&variant, mv.x, mv.y, pid);
                let idx = remaining[current_player].iter()
                    .position(|&i| i == global_idx).unwrap();
                remaining[current_player].remove(idx);
                placed_first[current_player] = true;
            }

            turn += 1;
            current_player = (current_player + 1) % player_count;
        }
    }

    if !output_path.is_empty() {
        write_games(output_path, &all_records)?;
    }

    let games_f = games as f32;
    Ok(BattleReport {
        wins_a: wins[0],
        wins_b: if player_count > 1 { wins[1] } else { 0 },
        draws: games - wins.iter().sum::<usize>(),
        elo_a: 0.0, elo_b: 0.0,
        avg_turns: total_turns as f32 / games_f,
        avg_remaining_a: 0.0, avg_remaining_b: 0.0,
        avg_mobility_a: 0.0, avg_mobility_b: 0.0,
    })
}
