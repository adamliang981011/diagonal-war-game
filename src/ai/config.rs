/// MCTS 設定（統一管理，避免常數散落各處）
#[derive(Debug, Clone)]
pub struct MctsConfig {
    pub iterations: usize,
    pub ucb_c: f32,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub max_tt_size: usize,
    pub parallel_threads: usize,
    pub virtual_loss: f32,
    pub prior_weight: f32,
    // 三階段 playout 深度 (opening, midgame, endgame)
    pub playout_depth_op: usize,
    pub playout_depth_mid: usize,
    pub playout_depth_end: usize,
    /// 開局進度閾值（低於此視為開局）
    pub progress_op: f32,
    /// 中局進度閾值（低於此視為中局）
    pub progress_mid: f32,
    /// 是否輸出 profiling 資訊（self-play 時關閉）
    pub print_profile: bool,
    /// Root Dirichlet Noise（self-play 時啟用）
    pub dirichlet_noise: bool,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        official_config()
    }
}

/// 正式遊戲（預設強度）
pub fn official_config() -> MctsConfig {
    MctsConfig {
        iterations: 500,
        ucb_c: 1.8,
        temperature_start: 1.5,
        temperature_end: 0.3,
        max_tt_size: 50_000,
        parallel_threads: 4,
        virtual_loss: 0.05,
        prior_weight: 0.5,
        playout_depth_op: 16,
        playout_depth_mid: 12,
        playout_depth_end: 20,
        progress_op: 0.3,
        progress_mid: 0.7,
        print_profile: true,
        dirichlet_noise: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
    }
}

/// 自我對弈（收集訓練資料，較低 iteration 換取更多局數）
pub fn self_play_config() -> MctsConfig {
    let mut cfg = official_config();
    cfg.iterations = 100;
    cfg.temperature_start = 1.8;
    cfg.temperature_end = 0.5;
    cfg.print_profile = false;
    cfg.dirichlet_noise = true;
    cfg
}

/// 速度基準測試（低 iteration + 固定 temperature）
pub fn bench_config() -> MctsConfig {
    MctsConfig {
        iterations: 50,
        ucb_c: 1.8,
        temperature_start: 0.5,
        temperature_end: 0.5,
        max_tt_size: 10_000,
        parallel_threads: 2,
        virtual_loss: 0.05,
        prior_weight: 0.5,
        playout_depth_op: 16,
        playout_depth_mid: 12,
        playout_depth_end: 20,
        progress_op: 0.3,
        progress_mid: 0.7,
        print_profile: true,
        dirichlet_noise: false,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
    }
}
