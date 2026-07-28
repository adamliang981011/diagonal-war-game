"""
Diagonal War — Value Network 訓練 + ONNX 匯出

可在 Colab 或本機執行。

Usage:
    python train_value_network.py --data training_data/games_*.bin --epochs 30

    在 Colab 上建議：
    !pip install torch onnx onnxruntime tqdm
    !python train_value_network.py --data drive/MyDrive/games_*.bin --epochs 50
"""

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import TrainingDataset


# 超參數
BOARD_SIZE = 20
N_CHANNELS = 64


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class ValueNetwork(nn.Module):
    """
    棋盤 20×20 + PlayerCount → Value (0~1)

    Architecture:
        Conv 3x3 1→64 + BN + ReLU
        6 × ResidualBlock(64)
        Conv 1x1 64→1 + Global Avg Pooling
        concat( pool(1), Embedding(3→16) )
        FC 17→1 + Tanh
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, N_CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(N_CHANNELS) for _ in range(6)])
        self.head_conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pc_embed = nn.Embedding(3, 16)  # 0=2P, 1=3P, 2=4P
        self.head_value = nn.Sequential(
            nn.Linear(1 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor, pc_idx: torch.Tensor) -> torch.Tensor:
        x = self.input(board)               # (B, 64, 20, 20)
        x = self.blocks(x)                  # (B, 64, 20, 20)
        x = self.head_conv(x)               # (B, 1, 20, 20)
        x = self.pool(x)                    # (B, 1, 1, 1)
        x = x.flatten(1)                    # (B, 1)
        embed = self.pc_embed(pc_idx)        # (B, 16)
        x = torch.cat([x, embed], dim=1)     # (B, 17)
        x = self.head_value(x)              # (B, 1)
        return x.squeeze(-1)                 # (B,)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 載入資料
    filepaths = sorted(glob.glob(args.data))
    print(f"Loading {len(filepaths)} files: {filepaths}")
    dataset = TrainingDataset(filepaths, augment=True)
    print(f"Total samples: {len(dataset)}")

    # 切分 train/val
    val_size = min(int(len(dataset) * 0.1), 5000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 建立模型
    model = ValueNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for board, value, pc_idx in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            board, value, pc_idx = board.to(device), value.to(device), pc_idx.to(device)
            optimizer.zero_grad()
            pred = model(board, pc_idx)
            loss = criterion(pred, value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * board.size(0)

        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for board, value, pc_idx in val_loader:
                board, value, pc_idx = board.to(device), value.to(device), pc_idx.to(device)
                pred = model(board, pc_idx)
                loss = criterion(pred, value)
                val_loss += loss.item() * board.size(0)
        val_loss /= val_size

        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"  → Saved best model to {args.model_path}")

    # 最終儲存
    torch.save(model.state_dict(), args.model_path.replace(".pt", "_final.pt"))
    print(f"\nDone! Best val_loss: {best_val_loss:.4f}")

    # ONNX 匯出
    export_onnx(model, args.model_path.replace(".pt", ".onnx"), device)


def export_onnx(model: nn.Module, onnx_path: str, device: torch.device):
    model.eval()
    dummy_board = torch.randn(1, 1, BOARD_SIZE, BOARD_SIZE, device=device)
    dummy_pc = torch.zeros(1, dtype=torch.long, device=device)
    torch.onnx.export(
        model,
        (dummy_board, dummy_pc),
        onnx_path,
        input_names=["board", "player_count"],
        output_names=["value"],
        dynamic_axes={"board": {0: "batch"}, "player_count": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="training_data/selfplay_*.bin",
                        help="Glob pattern for training .bin files")
    parser.add_argument("--model-path", type=str, default="model/value.pt",
                        help="Output model path (.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_path or "."), exist_ok=True)

    train(args)
