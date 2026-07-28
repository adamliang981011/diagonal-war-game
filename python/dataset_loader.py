"""
Diagonal War — 訓練資料載入器

從 Rust bincode 格式的 .bin 檔載入訓練資料，
轉為 PyTorch Dataset。

Usage:
    from dataset_loader import TrainingDataset
    ds = TrainingDataset(["games_001.bin", "games_002.bin"])
    board, value = ds[0]  # board: (1, 20, 20), value: float
"""

import struct
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_bincode_games(filepath: str) -> list[dict]:
    """讀取 Rust bincode 序列化的 Vec<GameRecord>"""
    with open(filepath, "rb") as f:
        data = f.read()

    offset = 0
    # u64: number of games
    num_games = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    games = []
    for _ in range(num_games):
        game = _parse_game_record(data, offset)
        offset = game["_end"]
        games.append(game)

    return games


def _parse_game_record(data: bytes, offset: int) -> dict:
    game = {}
    # u64: game_id
    game["game_id"] = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    # u8: player_count
    game["player_count"] = data[offset]
    offset += 1

    # Vec<StepRecord>: u64 length + steps
    num_steps = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    steps = []
    for _ in range(num_steps):
        step, offset = _parse_step_record(data, offset)
        steps.append(step)
    game["steps"] = steps

    # u8: final_winner
    game["final_winner"] = data[offset]
    offset += 1

    game["_end"] = offset
    return game


def _parse_step_record(data: bytes, offset: int) -> tuple:
    step = {}
    # [u8; 20][20]: board (400 bytes)
    board = np.frombuffer(data[offset:offset + 400], dtype=np.uint8).reshape(20, 20)
    step["board"] = board
    offset += 400

    # u8: current_player
    step["current_player"] = data[offset]
    offset += 1

    # u8: player_count
    step["player_count"] = data[offset]
    offset += 1

    # [u64; 4]: remaining_mask (32 bytes)
    masks = struct.unpack_from("<4Q", data, offset)
    step["remaining_mask"] = masks
    offset += 32

    # u16: turn
    step["turn"] = struct.unpack_from("<H", data, offset)[0]
    offset += 2

    # u8: winner
    step["winner"] = data[offset]
    offset += 1

    # f32: mcts_value
    step["mcts_value"] = struct.unpack_from("<f", data, offset)[0]
    offset += 4

    # u32: total_visits
    step["total_visits"] = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # VisitRecord chosen_move: (u8 piece, u8 variant, i8 x, i8 y, u32 visits)
    cm = {}
    cm["piece"] = data[offset]; offset += 1
    cm["variant"] = data[offset]; offset += 1
    cm["x"] = struct.unpack_from("<b", data, offset)[0]; offset += 1
    cm["y"] = struct.unpack_from("<b", data, offset)[0]; offset += 1
    cm["visits"] = struct.unpack_from("<I", data, offset)[0]; offset += 4
    step["chosen_move"] = cm

    # Vec<PolicyRecord>: u64 length + N * (u8 piece, u8 variant, i8 x, i8 y, f32 probability)
    num_visits = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    visits = []
    for _ in range(num_visits):
        rec = {}
        rec["piece"] = data[offset]; offset += 1
        rec["variant"] = data[offset]; offset += 1
        rec["x"] = struct.unpack_from("<b", data, offset)[0]; offset += 1
        rec["y"] = struct.unpack_from("<b", data, offset)[0]; offset += 1
        rec["probability"] = struct.unpack_from("<f", data, offset)[0]; offset += 4
        visits.append(rec)
    step["root_visits"] = visits

    # u16: game_rule_version
    step["game_rule_version"] = struct.unpack_from("<H", data, offset)[0]
    offset += 2

    # u32: ai_version
    step["ai_version"] = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    # u64: random_seed
    step["random_seed"] = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    return step, offset


def compute_value(step: dict) -> float:
    """根據當前玩家與最終贏家計算 Value (-1~1)"""
    w = step["winner"]
    if w == 255:
        return 0.0  # draw / unknown
    if w == step["current_player"]:
        return 1.0  # 當前玩家獲勝
    return -1.0  # 當前玩家落敗


class TrainingDataset(Dataset):
    """
    PyTorch Dataset for Value Network training.

    Args:
        filepaths: list of .bin file paths
        augment: if True, apply rotations/mirrors augmentation
        target: "winner" (default, ±1 from game outcome) or "mcts" (0~1 from MCTS root Q)
    """

    def __init__(self, filepaths: list[str], augment: bool = True, target: str = "mcts"):
        self.boards = []
        self.values = []
        self.pc_indices = []  # 0=2P, 1=3P, 2=4P
        self.augment = augment

        for fp in filepaths:
            games = parse_bincode_games(fp)
            for game in games:
                for step in game["steps"]:
                    self.boards.append(step["board"])
                    if target == "mcts":
                        self.values.append(step.get("mcts_value", 0.5))
                    else:
                        self.values.append(compute_value(step))
                    pc = step.get("player_count", 2)
                    self.pc_indices.append(max(0, min(pc - 2, 2)))

        # 轉為 torch tensor
        self.boards = np.stack(self.boards, axis=0).astype(np.float32)  # (N, 20, 20)
        self.values = np.array(self.values, dtype=np.float32)  # (N,)
        self.pc_indices = np.array(self.pc_indices, dtype=np.int64)  # (N,)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple:
        board = self.boards[idx]
        value = self.values[idx]
        pc_idx = self.pc_indices[idx]

        if self.augment:
            board = self._augment(board)

        # board: (1, 20, 20) 單通道
        board_t = torch.from_numpy(board.copy()).unsqueeze(0)
        value_t = torch.tensor(value, dtype=torch.float32)
        pc_idx_t = torch.tensor(pc_idx, dtype=torch.long)
        return board_t, value_t, pc_idx_t

    def _augment(self, board: np.ndarray) -> np.ndarray:
        """資料增強：翻轉 + 旋轉"""
        k = np.random.randint(0, 4)
        board = np.rot90(board, k=k)
        if np.random.rand() > 0.5:
            board = np.fliplr(board)
        return board


def export_flat(path_in: str, path_out_prefix: str):
    """
    將 .bin 轉為簡易二進位格式供其他語言使用：
    - {prefix}_boards.bin:  (N, 400) uint8
    - {prefix}_values.bin: (N,) float32
    """
    ds = TrainingDataset([path_in], augment=False)
    boards = ds.boards.astype(np.uint8)
    values = ds.values.astype(np.float32)

    # Header: 8 bytes = N
    with open(f"{path_out_prefix}_boards.bin", "wb") as f:
        f.write(np.uint64(len(boards)).tobytes())
        f.write(boards.tobytes())

    with open(f"{path_out_prefix}_values.bin", "wb") as f:
        f.write(np.uint64(len(values)).tobytes())
        f.write(values.tobytes())

    print(f"Exported {len(boards)} samples to {path_out_prefix}_*")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        export_flat(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python dataset_loader.py <input.bin> <output_prefix>")
