"""
Diagonal War — DualHead Network 訓練 (Value + Policy)

Usage:
    python python/train_dual_network.py \
        --data "training_data/selfplay_*.bin" \
        --model-path model/dual_unified.pt \
        --epochs 50
"""

import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import PolicyDataset, MAX_ACTIONS

BOARD_SIZE = 20
N_CHANNELS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class DualHeadNetwork(nn.Module):
    """
    棋盤 20×20 + PlayerCount → Value + Policy(70400 logits)

    Architecture:
        Conv 1→64 + 6×ResBlock
        ├── Policy: Conv 64→32 → Flatten → FC 12800→512→70400
        └── Value : Conv 64→1 → Pool → concat(embed) → FC → Tanh
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, N_CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(N_CHANNELS) for _ in range(6)])
        # Policy head (with bottleneck: 12800 → 512 → 70400)
        self.policy_head = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, MAX_ACTIONS),
        )
        # Value head
        self.head_conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pc_embed = nn.Embedding(3, 16)
        self.value_head = nn.Sequential(
            nn.Linear(1 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor, pc_idx: torch.Tensor) -> tuple:
        x = self.input(board)                  # (B, 64, 20, 20)
        x = self.blocks(x)                     # (B, 64, 20, 20)
        policy_logits = self.policy_head(x)    # (B, 70400)
        v = self.head_conv(x)                  # (B, 1, 20, 20)
        v = self.pool(v).flatten(1)            # (B, 1)
        embed = self.pc_embed(pc_idx)          # (B, 16)
        value = self.value_head(torch.cat([v, embed], dim=1)).squeeze(-1)  # (B,)
        return policy_logits, value


def soft_cross_entropy(logits: torch.Tensor, target_indices: list[torch.Tensor],
                       target_probs: list[torch.Tensor]) -> torch.Tensor:
    """Soft Cross Entropy: -Σ π(a) log P(a), supports sparse targets per sample."""
    B = logits.size(0)
    log_probs = F.log_softmax(logits, dim=1)  # (B, 70400)
    total_loss = 0.0
    for i in range(B):
        if target_indices[i].numel() == 0:
            continue
        log_p = log_probs[i][target_indices[i]]  # (N_i,)
        total_loss = total_loss - (target_probs[i] * log_p).sum()
    return total_loss / B


def dual_collate(batch):
    """Collate function for variable-length sparse policy targets."""
    boards = torch.stack([b[0] for b in batch], dim=0)
    values = torch.stack([b[1] for b in batch], dim=0)
    pc_idxs = torch.stack([b[2] for b in batch], dim=0)
    pol_idxs = [b[3] for b in batch]
    pol_probs = [b[4] for b in batch]
    return boards, values, pc_idxs, pol_idxs, pol_probs


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="training_data/selfplay_*.bin")
    parser.add_argument("--model-path", type=str, default="model/dual_unified.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0, help="λ for policy loss")
    args = parser.parse_args()

    filepaths = sorted(glob.glob(args.data))
    if not filepaths:
        print(f"No files matching {args.data}")
        return
    print(f"Loading {len(filepaths)} files: {filepaths}")

    ds = PolicyDataset(filepaths, augment=True, target="mcts")
    print(f"Dataset: {len(ds)} samples")

    # Train/val split
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=dual_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=dual_collate)

    model = DualHeadNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_v, train_loss_p, train_count = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            board, value, pc_idx, pol_idx, pol_prob = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            policy_logits, pred_value = model(board, pc_idx)

            loss_v = F.mse_loss(pred_value, value)
            loss_p = soft_cross_entropy(policy_logits,
                                        [p for p in pol_idx], [p for p in pol_prob])
            loss = loss_v + args.policy_weight * loss_p
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_v += loss_v.item() * board.size(0)
            train_loss_p += loss_p.item() * board.size(0)
            train_count += board.size(0)

        train_loss_v /= train_count
        train_loss_p /= train_count

        # Validation
        model.eval()
        val_loss_v, val_loss_p, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                board, value, pc_idx, pol_idx, pol_prob = [x.to(DEVICE) for x in batch]
                policy_logits, pred_value = model(board, pc_idx)
                loss_v = F.mse_loss(pred_value, value)
                loss_p = soft_cross_entropy(policy_logits,
                                           [p for p in pol_idx], [p for p in pol_prob])
                val_loss_v += loss_v.item() * board.size(0)
                val_loss_p += loss_p.item() * board.size(0)
                val_count += board.size(0)

        val_loss_v /= val_count
        val_loss_p /= val_count
        total_val = val_loss_v + args.policy_weight * val_loss_p
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: V={train_loss_v:.4f}/{val_loss_v:.4f}  "
              f"P={train_loss_p:.4f}/{val_loss_p:.4f}  lr={lr_now:.2e}")

        if total_val < best_loss:
            best_loss = total_val
            torch.save(model.state_dict(), args.model_path)
            print(f"  → Saved best (total={total_val:.4f})")

    torch.save(model.state_dict(), args.model_path.replace('.pt', '_final.pt'))
    print(f"\nDone! Best total loss: {best_loss:.4f}")

    # Export ONNX
    export_onnx(model, args.model_path.replace('.pt', '.onnx'))


def export_onnx(model: nn.Module, onnx_path: str):
    model.eval()
    dummy_board = torch.randn(1, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    dummy_pc = torch.zeros(1, dtype=torch.long, device=DEVICE)
    torch.onnx.export(
        model,
        (dummy_board, dummy_pc),
        onnx_path,
        input_names=["board", "player_count"],
        output_names=["value", "policy"],
        dynamic_axes={
            "board": {0: "batch"}, "player_count": {0: "batch"},
            "value": {0: "batch"}, "policy": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX exported to {onnx_path}")

    # Quick validation
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    out = sess.run(None, {
        "board": dummy_board.cpu().numpy(),
        "player_count": dummy_pc.cpu().numpy(),
    })
    print(f"ONNX test: value={out[0][0]:.4f}, policy shape={out[1].shape}")


if __name__ == "__main__":
    train()
