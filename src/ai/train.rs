use std::time::Instant;

use crate::ai::opening_book::{OpeningBook, OpeningEntry};
use crate::game::board::{Board, CellState, Corner};
use crate::game::piece::PieceShape;
use crate::game::player::PlayerId;

use super::list_legal_moves;

/// 訓練前 3 步的開局書
/// 限制每層分支數，只在最深層執行 playout 評估
pub fn train_depth_3(
    all_pieces: &[PieceShape],
    book_path: &str,
) -> Result<OpeningBook, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut book = OpeningBook::new();
    let board: Board<20> = Board::new();
    let all_players: Vec<PlayerId> = (0..4).map(PlayerId).collect();

    println!(
        "[OpeningBook] 開始訓練深度 ≤ 3，使用 {} 個棋子",
        all_pieces.len()
    );

    train_at_depth(&board, PlayerId(0), all_pieces, &all_players, 0, 3, &mut book);

    let elapsed = start.elapsed();
    println!(
        "[OpeningBook] 訓練完成！共 {} 個 entry，耗時 {:.1}s",
        book.len(),
        elapsed.as_secs_f32()
    );

    book.save(book_path)?;
    println!("[OpeningBook] 已儲存至 {}", book_path);
    Ok(book)
}

/// 遞迴訓練，每層只取前 MAX_BRANCH 個候選步
const MAX_BRANCH: usize = 20;

fn train_at_depth<const N: usize>(
    board: &Board<N>,
    player: PlayerId,
    all_pieces: &[PieceShape],
    all_players: &[PlayerId],
    depth: usize,
    max_depth: usize,
    book: &mut OpeningBook,
) {
    let is_first = !board.cells.iter().flatten().any(|&c| c == CellState::Occupied(player));

    let mut moves = list_legal_moves(board, player, all_pieces, is_first, Some(match player.0 {
        0 => Corner::TopLeft,
        1 => Corner::TopRight,
        2 => Corner::BottomRight,
        _ => Corner::BottomLeft,
    }));

    if moves.is_empty() {
        return;
    }

    // 只取前 MAX_BRANCH 個高分步
    moves.sort_by(|a, b| b.4.cmp(&a.4));
    moves.truncate(MAX_BRANCH);

    let mut entries: Vec<(u64, OpeningEntry)> = Vec::with_capacity(moves.len());

    for &(pi, vi, x, y, _base) in &moves {
        let variant = &all_pieces[pi].variants[vi];
        let mut sim_board = board.clone();
        sim_board.place_piece(variant, x, y, player);
        let sim_hash = sim_board.board_hash();

        if depth + 1 < max_depth {
            // 遞迴下一層（不評估，只在最深層評估）
            let next_player = PlayerId((player.0 + 1) % 4);
            train_at_depth(&sim_board, next_player, all_pieces, all_players, depth + 1, max_depth, book);
        }

        // 只在最深層或底層執行 playout
        let score = if depth + 1 >= max_depth {
            fast_playout(&sim_board, player, all_players, all_pieces)
        } else {
            0.5 // 中間層暫用 0.5，後續由子層的統計取代
        };

        entries.push((sim_hash, OpeningEntry {
            visits: 10,
            score,
            best_piece: pi,
            best_variant: vi,
            best_x: x,
            best_y: y,
        }));
    }

    // 找出此盤面最佳步
    let board_hash = board.board_hash();
    if let Some(best) = entries.iter().max_by(|a, b| {
        a.1.score.partial_cmp(&b.1.score).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        book.insert(board_hash, OpeningEntry {
            visits: 10,
            score: best.1.score,
            best_piece: best.1.best_piece,
            best_variant: best.1.best_variant,
            best_x: best.1.best_x,
            best_y: best.1.best_y,
        });
    }
}

/// 快速 playout：使用 find_any_legal_move + 亂序掃描（不需 list_legal_moves）
fn fast_playout<const N: usize>(
    board: &Board<N>,
    root_player: PlayerId,
    all_players: &[PlayerId],
    all_pieces: &[PieceShape],
) -> f32 {
    use crate::game::rules;
    let mut sim_board = board.clone();
    let mut sim_player = root_player;
    let mut consecutive_passes = 0;
    let mut used: Vec<usize> = Vec::new();

    loop {
        let is_first = !sim_board.cells.iter().flatten().any(|&c| c == CellState::Occupied(sim_player));
        let corner = match sim_player.0 {
            0 => Corner::TopLeft,
            1 => Corner::TopRight,
            2 => Corner::BottomRight,
            _ => Corner::BottomLeft,
        };

        // 使用 find_any_legal_move（O(數百) vs list_legal_moves O(6萬)）
        let mv = rules::find_any_legal_move(&sim_board, sim_player, all_pieces, is_first, Some(corner));

        if let Some((pi, vi, x, y)) = mv {
            if !used.contains(&pi) {
                consecutive_passes = 0;
                used.push(pi);
                let variant = &all_pieces[pi].variants[vi];
                sim_board.place_piece(variant, x, y, sim_player);
            } else {
                consecutive_passes += 1;
            }
        } else {
            consecutive_passes += 1;
        }

        if consecutive_passes >= 4 * all_players.len() {
            break;
        }
        sim_player = PlayerId((sim_player.0 + 1) % 4);
    }

    // 使用啟發式評估而非單純算佔領比例
    let occupied = sim_board.cells.iter().flatten().filter(|&&c| c != CellState::Empty).count() as f32;
    let weights = crate::ai::evaluate::compute_phase_weights(occupied, crate::ai::evaluate::TOTAL_PIECE_AREA);
    crate::ai::evaluate::heuristic_evaluate_with_weights(&sim_board, root_player, all_players, &weights)
}
