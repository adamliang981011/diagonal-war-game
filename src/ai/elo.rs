/// 簡易 ELO 計算系統
const K: f32 = 32.0; // ELO K-factor

#[derive(Debug, Clone)]
pub struct EloRating {
    pub rating: f32,
    pub games_played: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl EloRating {
    pub fn new() -> Self {
        Self { rating: 1500.0, games_played: 0, wins: 0, losses: 0, draws: 0 }
    }

    /// 預期勝率（標準 ELO 公式）
    fn expected_score(rating_a: f32, rating_b: f32) -> f32 {
        1.0 / (1.0 + 10.0_f32.powf((rating_b - rating_a) / 400.0))
    }

    /// 更新兩名玩家的 ELO（result: 1.0 = A 勝，0.5 = 平手，0.0 = B 勝）
    pub fn update(winner: &mut EloRating, loser: &mut EloRating) {
        let expected_a = Self::expected_score(winner.rating, loser.rating);
        let expected_b = 1.0 - expected_a;

        winner.rating += K * (1.0 - expected_a);
        loser.rating += K * (0.0 - expected_b);

        winner.games_played += 1;
        loser.games_played += 1;
        winner.wins += 1;
        loser.losses += 1;
    }

    pub fn draw(a: &mut EloRating, b: &mut EloRating) {
        let expected_a = Self::expected_score(a.rating, b.rating);
        let expected_b = 1.0 - expected_a;

        a.rating += K * (0.5 - expected_a);
        b.rating += K * (0.5 - expected_b);

        a.games_played += 1;
        b.games_played += 1;
        a.draws += 1;
        b.draws += 1;
    }
}
