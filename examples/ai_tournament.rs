// AI 強度分級錦標賽
// 執行方式：cargo run --example ai_tournament

use std::collections::HashMap;
use std::time::Instant;

use diagonal_war::ai::choose_move as ai_choose;
use diagonal_war::ai::elo::EloRating;
use diagonal_war::ai::AiDifficulty;
use diagonal_war::game::board::{Board, CellState, Corner};
use diagonal_war::game::piece_library;
use diagonal_war::game::player::PlayerId;

/// 進行一場完整雙人遊戲，回傳勝者 player_index (0/1)，None = 平手
fn play_game<const N: usize>(ai_a: AiDifficulty, ai_b: AiDifficulty) -> Option<usize> {
    let all_pieces = piece_library::create_all_pieces();
    let mut board: Board<N> = Board::new();
    let mut active = [true, true];
    let mut used_a: Vec<usize> = Vec::new();
    let mut used_b: Vec<usize> = Vec::new();

    loop {
        if active[0] {
            let remaining: Vec<_> = all_pieces.iter().enumerate()
                .filter(|(i, _)| !used_a.contains(i)).map(|(_, p)| p.clone()).collect();
            let mv = ai_choose(&board, PlayerId(0), &remaining,
                !board.cells.iter().flatten().any(|&c| CellState::Occupied(PlayerId(0)) == c),
                Some(Corner::TopLeft), ai_a, 2);
            if let Some(m) = mv {
                let v = &remaining[m.piece_index].variants[m.variant_index];
                board.place_piece(v, m.x, m.y, PlayerId(0));
                used_a.push(all_pieces.iter().position(|p| p.id == remaining[m.piece_index].id).unwrap());
            } else { active[0] = false; }
        }
        if active[1] {
            let remaining: Vec<_> = all_pieces.iter().enumerate()
                .filter(|(i, _)| !used_b.contains(i)).map(|(_, p)| p.clone()).collect();
            let mv = ai_choose(&board, PlayerId(1), &remaining,
                !board.cells.iter().flatten().any(|&c| CellState::Occupied(PlayerId(1)) == c),
                Some(Corner::TopRight), ai_b, 2);
            if let Some(m) = mv {
                let v = &remaining[m.piece_index].variants[m.variant_index];
                board.place_piece(v, m.x, m.y, PlayerId(1));
                used_b.push(all_pieces.iter().position(|p| p.id == remaining[m.piece_index].id).unwrap());
            } else { active[1] = false; }
        }
        if !active[0] && !active[1] { break; }
    }

    // 依剩餘格數判定勝負
    let remain_a: usize = all_pieces.iter().enumerate()
        .filter(|(i, _)| !used_a.contains(i)).map(|(_, p)| p.base.cells.len()).sum();
    let remain_b: usize = all_pieces.iter().enumerate()
        .filter(|(i, _)| !used_b.contains(i)).map(|(_, p)| p.base.cells.len()).sum();

    if remain_a < remain_b { Some(0) }
    else if remain_b < remain_a { Some(1) }
    else { None }
}

fn run_matchup(name_a: &str, diff_a: AiDifficulty, name_b: &str, diff_b: AiDifficulty, games: usize) -> (u32, u32, u32, f32) {
    print!("  {} vs {} ({} 場)...", name_a, name_b, games);
    let start = Instant::now();
    let mut a_wins = 0u32; let mut b_wins = 0u32; let mut draws = 0u32;
    for _ in 0..games {
        match play_game::<20>(diff_a, diff_b) {
            Some(0) => a_wins += 1, Some(1) => b_wins += 1, _ => draws += 1,
        }
    }
    let elapsed = start.elapsed();
    println!(" 完成! (A: {}勝/B: {}勝/平: {} 耗時{:.0}s)", a_wins, b_wins, draws, elapsed.as_secs_f32());
    (a_wins, b_wins, draws, elapsed.as_secs_f32())
}

fn main() {
    println!("===== Diagonal War AI 強度錦標賽 =====");
    println!("分級說明:");
    println!("  Tier 1 (快): Lv1 隨機 / Lv2 貪婪 / Lv3 多變  (每組 30 場)");
    println!("  Tier 2 (中): Lv4 前瞻1 / Lv5 前瞻2           (每組 5 場)");
    println!("  Tier 3 (慢): Lv6 MCTS                        (每組 10 場)");
    println!();

    let tiers: Vec<(&str, AiDifficulty, usize, usize)> = vec![
        ("Lv1 隨機",  AiDifficulty::Random,            0, 30),
        ("Lv2 貪婪",  AiDifficulty::Greedy,            0, 30),
        ("Lv3 多變",  AiDifficulty::GreedyWithTemp(0.3), 0, 30),
        ("Lv4 前瞻1", AiDifficulty::Search1Ply,        1,  5),
        ("Lv5 前瞻2", AiDifficulty::Search2Ply,        1,  5),
        ("Lv6 MCTS",  AiDifficulty::Mcts { iterations: 50 }, 2, 10),
    ];

    let mut elo: HashMap<&str, EloRating> = HashMap::new();
    for (name, _, _, _) in &tiers { elo.insert(name, EloRating::new()); }

    let total_start = Instant::now();

    // Tier 1 內部對戰（快，30 場）
    println!("=== Tier 1 內部對戰 ===");
    for i in 0..3 {
        for j in (i+1)..3 {
            let (na, da, _, n) = tiers[i]; let (nb, db, _, _) = tiers[j];
            let (aw, bw, dr, _t) = run_matchup(na, da, nb, db, n);
            let total = aw + bw + dr;
            let mut elo_a = elo.get(na).unwrap().clone(); let mut elo_b = elo.get(nb).unwrap().clone();
            for _ in 0..aw { EloRating::update(&mut elo_a, &mut elo_b); }
            for _ in 0..bw { EloRating::update(&mut elo_b, &mut elo_a); }
            for _ in 0..dr { EloRating::update(&mut elo_a, &mut elo_b); EloRating::update(&mut elo_b, &mut elo_a); }
            elo.insert(na, elo_a); elo.insert(nb, elo_b);
            println!("  結果: {} {:.0}% / {} {:.0}%", na, aw as f32/total as f32*100.0, nb, bw as f32/total as f32*100.0);
        }
    }

    // Tier 2+3 挑戰 Tier 1（各 5-10 場）
    println!("\n=== Tier 2+3 挑戰 Tier 1 ===");
    for &(nb, db, _tb, ng) in &tiers[3..] {
        for &(na, da, _, _) in &tiers[0..3] {
            let (aw, bw, dr, _t) = run_matchup(na, da, nb, db, ng);
            let total = aw + bw + dr;
            let mut elo_a = elo.get(na).unwrap().clone(); let mut elo_b = elo.get(nb).unwrap().clone();
            for _ in 0..aw { EloRating::update(&mut elo_a, &mut elo_b); }
            for _ in 0..bw { EloRating::update(&mut elo_b, &mut elo_a); }
            for _ in 0..dr { EloRating::update(&mut elo_a, &mut elo_b); EloRating::update(&mut elo_b, &mut elo_a); }
            elo.insert(na, elo_a); elo.insert(nb, elo_b);
            println!("  結果: {} {:.0}% / {} {:.0}%", na, aw as f32/total as f32*100.0, nb, bw as f32/total as f32*100.0);
        }
    }

    // Tier 2 vs Tier 3（各 3 場）
    println!("\n=== Tier 2 vs Tier 3 ===");
    for &(na, da, _, _) in &tiers[3..] {
        for &(nb, db, _, _) in &tiers[3..] {
            if na == nb { continue; }
            let (aw, bw, dr, _t) = run_matchup(na, da, nb, db, 3);
            let total = aw + bw + dr;
            let mut elo_a = elo.get(na).unwrap().clone(); let mut elo_b = elo.get(nb).unwrap().clone();
            for _ in 0..aw { EloRating::update(&mut elo_a, &mut elo_b); }
            for _ in 0..bw { EloRating::update(&mut elo_b, &mut elo_a); }
            for _ in 0..dr { EloRating::update(&mut elo_a, &mut elo_b); EloRating::update(&mut elo_b, &mut elo_a); }
            elo.insert(na, elo_a); elo.insert(nb, elo_b);
            println!("  結果: {} {:.0}% / {} {:.0}%", na, aw as f32/total as f32*100.0, nb, bw as f32/total as f32*100.0);
        }
    }

    // 最終 ELO 排名
    println!("\n===== 最終 ELO 排名 =====");
    let mut ranked: Vec<(&&str, &EloRating)> = elo.iter().collect();
    ranked.sort_by(|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap());
    for (rank, (name, e)) in ranked.iter().enumerate() {
        println!("  {}. {}  ELO: {:.0}  ({}W/{}L/{}D)", rank+1, name, e.rating, e.wins, e.losses, e.draws);
    }
    println!("\n總耗時: {:.1}s", total_start.elapsed().as_secs_f32());
}
