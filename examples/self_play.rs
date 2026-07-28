// Self-play 訓練資料產生器
// 執行方式：
//   cargo run --example self_play --release
//   cargo run --example self_play --release -- --players 4 --games 500 --threads 4
// 輸出：training_data/selfplay_{p}p.bin

use std::env;
use std::time::Instant;

use diagonal_war::ai::AiDifficulty;
use diagonal_war::training::self_play::generate_self_play_games;
use diagonal_war::training::export::read_games;

fn parse_args() -> (usize, usize, usize, u64) {
    let args: Vec<String> = env::args().collect();
    let mut players = 2usize;
    let mut games = 200usize;
    let mut threads = 4usize;
    let mut seed = 0u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--players" | "-p" => { i += 1; if i < args.len() { players = args[i].parse().unwrap_or(2); } }
            "--games" | "-g" => { i += 1; if i < args.len() { games = args[i].parse().unwrap_or(200); } }
            "--threads" | "-t" => { i += 1; if i < args.len() { threads = args[i].parse().unwrap_or(4); } }
            "--seed" | "-s" => { i += 1; if i < args.len() { seed = args[i].parse().unwrap_or(0); } }
            _ => {}
        }
        i += 1;
    }
    (players, games, threads, seed)
}

fn main() {
    let (players, num_games, _threads, seed) = parse_args();

    let output_path = format!("training_data/selfplay_{}p.bin", players);
    std::fs::create_dir_all("training_data").ok();

    println!("=== Self-play 訓練資料產生器 ===");
    println!("設定: {} 人, {} 局, seed={}", players, num_games, seed);

    let start = Instant::now();

    let difficulties: Vec<AiDifficulty> = (0..players)
        .map(|_| AiDifficulty::Mcts { iterations: 100 })
        .collect();

    let report = generate_self_play_games(
        players,
        &difficulties,
        num_games,
        &output_path,
        seed,
    ).expect("self-play failed");

    let elapsed = start.elapsed();

    println!("\n=== 完成 ===");
    println!("耗時: {:.1}s", elapsed.as_secs_f64());
    println!("平均每局: {:.1}s", elapsed.as_secs_f64() / num_games as f64);
    println!("平均回合數: {:.1}", report.avg_turns);
    println!("勝率 P0: {:.1}%", report.wins_a as f64 / num_games as f64 * 100.0);
    if players == 2 {
        println!("勝率 P1: {:.1}%", report.wins_b as f64 / num_games as f64 * 100.0);
    }
    println!("平手: {:.1}%", report.draws as f64 / num_games as f64 * 100.0);

    // 驗證輸出檔案
    if let Ok(games) = read_games(&output_path) {
        let total_steps: usize = games.iter().map(|g| g.steps.len()).sum();
        let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
        println!("\n輸出檔案: {}", output_path);
        println!("檔案大小: {:.1} MB", file_size as f64 / 1_000_000.0);
        println!("對局數: {}", games.len());
        println!("總步數: {}", total_steps);
        if total_steps > 0 {
            println!("訓練樣本數: {} (每步一筆)", total_steps);
        }
    }

    println!("\n下一步: python python/train_value_network.py --data \"training_data/selfplay_*p.bin\" --model-path model/value.pt --epochs 30");
}
