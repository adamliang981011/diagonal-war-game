/// MCTS PUCT_C Benchmark
/// 量測不同 PUCT_C 值對搜尋分布的影響
///
/// 使用方法：
///   cargo run --release --example thread_bench
///
/// 輸出 CSV：
///   puct_c,time_sec,top1_ratio,top5_ratio,children
///   0.50,0.281,0.592,0.844,18

use std::time::Instant;

use diagonal_war::ai::config::official_config;
use diagonal_war::game::board::Board;
use diagonal_war::game::piece_library;
use diagonal_war::game::player::{starting_corner_for_player, PlayerId};

fn main() {
    let pieces = piece_library::create_all_pieces();
    let mut board: Board<20> = Board::new();
    let player = PlayerId(0);

    // 模擬 midgame 盤面：放置多顆棋子製造複雜局面
    // Player 0 佔據左下角區域
    for (pi, x, y, vi) in &[(0, 0, 0, 0), (1, 2, 0, 0), (2, 0, 3, 0), (4, 3, 0, 0), (9, 5, 4, 0)] {
        let v = pieces[*pi].variants[*vi].clone();
        board.place_piece(&v, *x, *y, player);
    }
    // Player 1 佔據右上角（只放單格棋子避免越界）
    let p1 = PlayerId(1);
    let mono1 = pieces[0].variants[0].clone();
    board.place_piece(&mono1, 19, 19, p1);
    let mono2 = pieces[0].variants[0].clone();
    board.place_piece(&mono2, 18, 18, p1);

    let all_pieces: Vec<_> = pieces.iter().take(13).map(|s| s.clone()).collect();

    let puct_values = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00];
    let total_iters = 500;
    let threads = 2;  // 固定 2 threads

    println!("puct_c,time_sec,top1_ratio,top5_ratio,children,top1_id");
    for &puct_c in &puct_values {
        let mut cfg = official_config();
        cfg.ucb_c = puct_c;
        cfg.iterations = total_iters;
        cfg.parallel_threads = threads;
        cfg.print_profile = false;

        let mut children_visits: Vec<u32> = Vec::new();
        let elapsed = {
            let start = Instant::now();
            diagonal_war::ai::with_search_state(|search_state| {
                let _ = diagonal_war::ai::mcts::choose_move::<20>(
                    &board, player, &all_pieces, false,
                    starting_corner_for_player(player.0, 2),
                    &cfg, 2, &mut None, search_state,
                );
                // 在 choose_move 之後、search_state 被覆寫前讀取 tree
                if let Some(ref state) = *search_state {
                    if let Some(ref tree) = state.tree {
                        for child in &tree.nodes[0].children {
                            children_visits.push(child.visits);
                        }
                    }
                }
            });
            start.elapsed()
        };

        if !children_visits.is_empty() {
            children_visits.sort_by(|a, b| b.cmp(a));
            let total: u32 = children_visits.iter().sum();
            let top1 = children_visits[0];
            let top5: u32 = children_visits.iter().take(5).sum();
            let top1_ratio = top1 as f64 / total as f64;
            let top5_ratio = top5 as f64 / total as f64;
                println!("{:.2},{:.4},{:.4},{:.4},{}", puct_c, elapsed.as_secs_f64(), top1_ratio, top5_ratio, children_visits.len());
        } else {
            println!("{:.2},{:.4},0.0,0.0,0", puct_c, elapsed.as_secs_f64());
        }
    }
}
