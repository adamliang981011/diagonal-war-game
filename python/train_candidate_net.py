"""
Diagonal War — Candidate Scoring Network 訓練

Usage:
    python python/train_candidate_net.py \
        --data "training_data/selfplay_*.bin" \
        --model-path model/candidate.pt \
        --epochs 50
"""

import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import CandidateDataset, BOARD_SIZE

N_CHANNELS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CANDIDATES = 32


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


class CandidateNetwork(nn.Module):
    """
    Candidate Scoring Network
    Board CNN → board_feat + value
    Move Encoder: piece_embed + variant_embed + size_norm + x_norm + y_norm → MLP → 64
    Score: board_feat + move_embed → MLP → scalar
    """

    def __init__(self):
        super().__init__()
        # Board Encoder
        self.board_conv = nn.Sequential(
            nn.Conv2d(1, N_CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(N_CHANNELS),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(N_CHANNELS) for _ in range(6)])
        self.head_conv = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pc_embed = nn.Embedding(3, 16)

        # Board feature projector
        self.board_proj = nn.Linear(64, 64)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(1 + 16, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh(),
        )

        # Move Encoder
        self.piece_embed = nn.Embedding(26, 16)
        self.variant_embed = nn.Embedding(8, 8)
        self.move_mlp = nn.Sequential(
            nn.Linear(16 + 8 + 1 + 1 + 1, 64), nn.ReLU(),  # piece + var + size + x + y
            nn.Linear(64, 64), nn.ReLU(),
        )

        # Score MLP (board_feat + move_feat)
        self.score_mlp = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, board, player_count, progress,
                cand_piece, cand_variant, cand_x, cand_y, cand_size):
        # Board Encoder
        x = self.board_conv(board)                           # (B, 64, 20, 20)
        x = self.blocks(x)                                   # (B, 64, 20, 20)

        # Value head
        v = self.head_conv(x)                                # (B, 1, 20, 20)
        v = self.pool(v).flatten(1)                          # (B, 1)
        embed = self.pc_embed(player_count)                  # (B, 16)
        value = self.value_head(torch.cat([v, embed], dim=1)).squeeze(-1)  # (B,)

        # Board feature (shared across candidates)
        board_feat = self.pool(x).flatten(1)                 # (B, 64)
        board_feat = self.board_proj(board_feat)             # (B, 64)

        # Move Encoder + Scoring (batch over candidates)
        N = cand_piece.size(1)  # max_candidates
        bf = board_feat.unsqueeze(1).expand(-1, N, -1)       # (B, N, 64)

        piece_e = self.piece_embed(cand_piece)               # (B, N, 16)
        var_e = self.variant_embed(cand_variant)             # (B, N, 8)
        move_feat = torch.cat([
            piece_e, var_e,
            cand_size.unsqueeze(-1),   # (B, N, 1)
            cand_x.unsqueeze(-1),      # (B, N, 1)
            cand_y.unsqueeze(-1),      # (B, N, 1)
        ], dim=-1)                                           # (B, N, 27)
        move_feat = self.move_mlp(move_feat)                 # (B, N, 64)

        scores = self.score_mlp(torch.cat([bf, move_feat], dim=-1)).squeeze(-1)  # (B, N)
        return value, scores


def candidate_loss(scores, target_probs):
    """Cross Entropy loss for variable-length candidate sets."""
    # scores: (B, N), target_probs: (B, N)
    log_probs = F.log_softmax(scores, dim=1)
    # Sum over valid candidates, mean over batch
    return - (target_probs * log_probs).sum(dim=1).mean()


def dual_collate(batch):
    boards = torch.stack([b[0] for b in batch])
    values = torch.stack([b[1] for b in batch])
    pc_idxs = torch.stack([b[2] for b in batch])
    progress = torch.stack([b[3] for b in batch])
    
    # Candidates: pad/truncate to MAX_CANDIDATES
    N = MAX_CANDIDATES
    keys = ["piece", "variant", "x", "y", "size", "prob"]
    cand = {k: [] for k in keys}
    for b in batch:
        c = b[4]
        for k in keys:
            arr = c[k].numpy()
            if len(arr) >= N:
                cand[k].append(arr[:N])
            else:
                padded = np.zeros(N, dtype=arr.dtype)
                padded[:len(arr)] = arr
                cand[k].append(padded)
    
    return (boards, values, pc_idxs, progress,
            {k: torch.from_numpy(np.stack(v)) for k, v in cand.items()})


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="training_data/selfplay_*.bin")
    parser.add_argument("--model-path", type=str, default="model/candidate.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    args = parser.parse_args()

    filepaths = sorted(glob.glob(args.data))
    if not filepaths:
        print(f"No files matching {args.data}")
        return

    ds = CandidateDataset(filepaths, augment=True)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=dual_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=dual_collate)

    model = CandidateNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, cooldown=1, min_lr=1e-6,
    )

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best_val = float('inf')
    patience_count = 0

    best_val = float('inf')
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tv, tp, cnt = 0.0, 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            boards, values, pc_idxs, progress, cand = batch
            boards = boards.to(DEVICE)
            values = values.to(DEVICE)
            pc_idxs = pc_idxs.to(DEVICE)
            progress = progress.to(DEVICE)
            cand = {k: v.to(DEVICE) for k, v in cand.items()}

            optimizer.zero_grad()
            pred_v, scores = model(boards, pc_idxs, progress,
                                   cand["piece"], cand["variant"],
                                   cand["x"], cand["y"], cand["size"])
            loss_v = F.mse_loss(pred_v, values)
            loss_p = candidate_loss(scores, cand["prob"])
            loss = loss_v + args.policy_weight * loss_p
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tv += loss_v.item() * boards.size(0)
            tp += loss_p.item() * boards.size(0)
            cnt += boards.size(0)

        tv /= cnt; tp /= cnt

        # Validation
        model.eval()
        vv, vp, vcnt = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                boards, values, pc_idxs, progress, cand = batch
                boards = boards.to(DEVICE); values = values.to(DEVICE)
                pc_idxs = pc_idxs.to(DEVICE); progress = progress.to(DEVICE)
                cand = {k: v.to(DEVICE) for k, v in cand.items()}
                pred_v, scores = model(boards, pc_idxs, progress,
                                       cand["piece"], cand["variant"],
                                       cand["x"], cand["y"], cand["size"])
                vv += F.mse_loss(pred_v, values).item() * boards.size(0)
                vp += candidate_loss(scores, cand["prob"]).item() * boards.size(0)
                vcnt += boards.size(0)

        vv /= vcnt; vp /= vcnt
        total_val = vv + args.policy_weight * vp
        scheduler.step(total_val)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: V={tv:.4f}/{vv:.4f}  P={tp:.4f}/{vp:.4f}  lr={lr_now:.2e}")

        if total_val < best_val:
            best_val = total_val
            patience_count = 0
            torch.save(model.state_dict(), args.model_path)
            print(f"  → Saved best")
        else:
            patience_count += 1
            if patience_count >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(args.model_path))
    print(f"\nDone! Best total val: {best_val:.4f}")
    export_onnx(model, args.model_path.replace('.pt', '.onnx'))


def export_onnx(model: nn.Module, path: str):
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        raise RuntimeError("Please install onnxscript: pip install onnxscript")

    model.eval()
    N = 16  # fixed candidate count for export
    dummy = (
        torch.randn(1, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE),
        torch.zeros(1, dtype=torch.long, device=DEVICE),
        torch.zeros(1, device=DEVICE),
        torch.zeros(1, N, dtype=torch.long, device=DEVICE),
        torch.zeros(1, N, dtype=torch.long, device=DEVICE),
        torch.zeros(1, N, device=DEVICE),
        torch.zeros(1, N, device=DEVICE),
        torch.zeros(1, N, device=DEVICE),
    )
    torch.onnx.export(
        model, dummy, path,
        input_names=["board", "player_count", "progress",
                     "cand_piece", "cand_variant", "cand_x", "cand_y", "cand_size"],
        output_names=["value", "scores"],
        dynamic_axes={
            "cand_piece": {1: "num_candidates"},
            "cand_variant": {1: "num_candidates"},
            "cand_x": {1: "num_candidates"},
            "cand_y": {1: "num_candidates"},
            "cand_size": {1: "num_candidates"},
            "scores": {1: "num_candidates"},
        },
        opset_version=17,
    )
    print(f"ONNX exported to {path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    out = sess.run(None, {
        "board": dummy[0].cpu().numpy(),
        "player_count": dummy[1].cpu().numpy(),
        "progress": dummy[2].cpu().numpy(),
        "cand_piece": dummy[3].cpu().numpy(),
        "cand_variant": dummy[4].cpu().numpy(),
        "cand_x": dummy[5].cpu().numpy(),
        "cand_y": dummy[6].cpu().numpy(),
        "cand_size": dummy[7].cpu().numpy(),
    })
    print(f"ONNX test: value={out[0][0]:.4f}, scores shape={out[1].shape}")


if __name__ == "__main__":
    train()
