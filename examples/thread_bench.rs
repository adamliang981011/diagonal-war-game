/// MCTS Thread Benchmark
/// 量測不同 thread 數下的搜尋效率
///
/// 使用方法：
///   cargo run --release --example thread_bench
///
/// 輸出 CSV 格式，可直接貼到試算表：
///   threads,iterations,time_sec,iters_per_sec
///   1,1000,0.82,1219
///   2,1000,0.56,1785
///   4,1000,0.53,1886

use std::time::Instant;

use diagonal_war::ai::config::official_config;
use diagonal_war::game::board::Board;
use diagonal_war::game::piece_library;
use diagonal_war::game::player::{starting_corner_for_player, PlayerId};

fn main() {
    let pieces = piece_library::create_all_pieces();
    let mut board: Board<20> = Board::new();
    let player = PlayerId(0);

    // 設定初始盤面：在角落放一個 1g 棋子
    let mono = pieces[0].variants[0].clone();
    board.place_piece(&mono, 0, 0, player);

    // 模擬 midgame 盤面：2P 各分一半棋子
    let all_pieces: Vec<_> = pieces.iter().take(13).map(|s| s.clone()).collect();

    let thread_configs = [1usize, 2, 4];
    let total_iters = 500;

    println!("threads,iterations,time_sec,iters_per_sec");
    for &threads in &thread_configs {
        let mut cfg = official_config();
        cfg.iterations = total_iters;
        cfg.parallel_threads = threads;
        cfg.print_profile = false;

        let start = Instant::now();
        let result = diagonal_war::ai::mcts::choose_move::<20>(
            &board, player, &all_pieces, false,
            starting_corner_for_player(player.0, 2),
            &cfg, 2, &mut None, &mut None,
        );
        let elapsed = start.elapsed();

        let iters_per_sec = total_iters as f64 / elapsed.as_secs_f64();
        println!("{},{},{:.4},{:.1}", threads, total_iters, elapsed.as_secs_f64(), iters_per_sec);

        if let Some(ref mv) = result {
            eprintln!("  {threads} threads: best={}g({},{}) Q={:.3}",
                all_pieces[mv.piece_index].base.cells.len(), mv.x, mv.y,
                mv.score as f32 / 1000.0);
        }
    }
}
