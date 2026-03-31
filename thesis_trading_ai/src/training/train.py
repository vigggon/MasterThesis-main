"""
Training on 2019–2023 only. Validation on subset of those years.
Models never see backtest (2024-01-01 to 2024-08-09) or live data.

Parameters are updated iteratively each batch via backprop; training runs for
multiple epochs. Best checkpoint is selected by validation profit (reward),
so profitability drives model choice.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json

from utils import PROCESSED_DIR, EXPERIMENTS_ROOT, TP_ATR_MULT, SL_ATR_MULT, TRADE_THRESHOLD, NUM_CLASSES, LSTM_HIDDEN, LSTM_LAYERS, TX_DMODEL, TX_NHEAD, TX_LAYERS, MIN_ATR
from models import get_lstm, get_transformer


class FocalLoss(nn.Module):
    """Down-weights easy examples (e.g. majority 'hold') so model focuses on hard/minority 'trade' class."""
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


def load_train_data(min_atr: float = 0.0, noise_level: float = 0.0):
    data = np.load(PROCESSED_DIR / "train.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).long()
    feature_cols = list(data["feature_cols"])
    
    # Filter by Min ATR if applicable
    if min_atr > 0.0:
        try:
            atr_idx = feature_cols.index("atr_14")
            # Filter samples where ATR (at decision time) < min_atr
            # X shape: (N, Window, Features)
            atr_values = X[:, -1, atr_idx]
            mask = atr_values >= min_atr
            
            print(f"Filtering dataset by Min ATR >= {min_atr}: {mask.sum()} / {len(mask)} samples kept ({mask.float().mean():.1%})")
            X = X[mask]
            y = y[mask]
        except ValueError:
            print("Warning: 'atr_14' not found in features. Cannot filter by Min ATR.")
            
    # Noise Injection (Test B)
    if noise_level > 0.0:
        print(f"Injecting noise (std * {noise_level:.3f}) into features...")
        # Add N(0, noise_level * std) to each feature
        # Calculate std per feature across dataset
        # X shape: (N, Window, Features) -> std over (N*Window) or just N per feature?
        # Standard approach: std per feature
        # Flatten N*Window for std calc
        X_flat = X.view(-1, X.size(2))
        stds = X_flat.std(dim=0)
        
        # Noise shape must match X
        noise = torch.randn_like(X) * stds.view(1, 1, -1) * noise_level
        X = X + noise

    return X, y, feature_cols


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm: float = 1.0):
    """One epoch: iterate over batches, update parameters (weights) via gradient descent."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()  # iterative parameter update
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += yb.size(0)
    return total_loss / len(loader), correct / n


def _pred_3class(probs: torch.Tensor, th: float) -> torch.Tensor:
    """3-class: 0=hold, 1=long, 2=short. Trade when max(P(long), P(short)) >= th; pick argmax of long vs short."""
    p_long = probs[:, 1]
    p_short = probs[:, 2]
    trade = (p_long >= th) | (p_short >= th)
    # pred = 0 if ~trade else (1 if p_long >= p_short else 2)
    pred = trade.long() * (1 + (p_short > p_long).long())
    return pred


def validation_reward_and_trades(model, loader, device, tp_reward: float = None, sl_penalty: float = None, threshold: float = None, transaction_cost: float = 0.0):
    """
    Validation profit proxy and trade count. 3-class: pred 1=long, 2=short, 0=hold.
    TP=+tp_reward, SL=-sl_penalty, no trade=0. Optionally deducts transaction_cost per trade.
    Returns (total_reward, n_trades).
    """
    tp_reward = tp_reward if tp_reward is not None else TP_ATR_MULT  # 1.2
    sl_penalty = sl_penalty if sl_penalty is not None else SL_ATR_MULT  # 1.0
    th = threshold if threshold is not None else TRADE_THRESHOLD
    model.eval()
    total_reward = 0.0
    n_trades = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = _pred_3class(probs, th)
            for i in range(pred.size(0)):
                p, t = pred[i].item(), yb[i].item()
                if p == 0:
                    continue
                n_trades += 1
                if p == 1:
                    reward = (tp_reward if t == 1 else -sl_penalty) - transaction_cost
                    total_reward += reward
                else:
                    reward = (tp_reward if t == 2 else -sl_penalty) - transaction_cost
                    total_reward += reward
    return total_reward, n_trades


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += yb.size(0)
    return total_loss / len(loader), correct / n if n else 0.0


def run_train(
    model_name: str = "lstm",
    epochs: int = 250,
    batch_size: int = 64,
    lr: float = 2e-3,
    val_ratio: float = 0.15,
    seed: int = 42,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 40,
    early_stop_on_reward: bool = False,
    tune_threshold: bool = False,
    hidden_size: int = LSTM_HIDDEN,
    num_layers: int = LSTM_LAYERS,
    d_model: int = TX_DMODEL,
    nhead: int = TX_NHEAD,
    tx_layers: int = TX_LAYERS,
    transaction_cost: float = 0.1,
    min_atr: float = 0.0,
    noise_level: float = 0.0,
    exp_dir: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, feature_cols = load_train_data(min_atr=min_atr, noise_level=noise_level)
    n, _, n_features = X.shape

    # Time-based val: last val_ratio of train period
    n_val = int(n * val_ratio)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    # Class weights + Focal loss: upweight minority and focus on hard examples to avoid collapse to hold
    num_classes = NUM_CLASSES
    uniq, counts = torch.unique(y_train, return_counts=True)
    count_tensor = torch.ones(num_classes, dtype=torch.float)
    for c, cnt in zip(uniq.tolist(), counts.tolist()):
        count_tensor[c] = float(cnt)
    class_weights = (y_train.size(0) / (num_classes * count_tensor)).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=3.0, label_smoothing=0.02)  # Harder mining, slight smoothing

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    # Recency weighting: exponential decay favoring recent data
    # weight[i] = exp(alpha * (i / n)) where higher i = more recent
    n_train = len(X_train)
    recency_alpha = 2.0  # Controls recency bias (higher = stronger bias toward recent)
    sample_weights = torch.exp(recency_alpha * torch.arange(n_train, dtype=torch.float) / n_train)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=n_train, replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    if model_name == "lstm":
        model = get_lstm(n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, num_classes=num_classes)
    else:
        model = get_transformer(n_features, d_model=d_model, nhead=nhead, num_layers=tx_layers, dropout=dropout, num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Schedule LR by validation loss (stable; best checkpoint still chosen by reward)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30
    )

    out_dir = Path(exp_dir) if exp_dir else EXPERIMENTS_ROOT / f"{model_name}_open"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select best checkpoint by validation profit (reward), only among checkpoints that trade (n_trades >= 1)
    best_val_reward = float("-inf")
    best_fallback_state = None  # best by reward even with 0 trades (fallback if model never trades)
    best_fallback_reward = float("-inf")
    # Early stopping: by default on val LOSS (avoids overfitting); optional on reward
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_reward": [], "val_trades": [],
    }

    for ep in range(epochs):
        # Epoch: one full pass over train data, parameters updated every batch
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = validate(model, val_loader, criterion, device)
        vr, n_val_trades = validation_reward_and_trades(model, val_loader, device, transaction_cost=transaction_cost)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_reward"].append(vr)
        history["val_trades"].append(n_val_trades)
        # Save only if this checkpoint actually trades (so backtest won't show 0 trades)
        # Increased min trades to 5 to avoid "lucky 1 trade" checkpoints
        if n_val_trades >= 5 and vr > best_val_reward:
            best_val_reward = vr
            epochs_no_improve = 0  # reset when we save a better checkpoint (for early stop on reward)
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "model_name": model_name,
            }, out_dir / "best.pt")
        if vr > best_fallback_reward:
            best_fallback_reward = vr
            best_fallback_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(vl)  # Reduce LR on val loss (stable; best ckpt still by reward)
        # Early stopping: on val loss (default) or on val reward (optional; patience reset when we save)
        if not early_stop_on_reward:
            if vl < best_val_loss:
                best_val_loss = vl
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            if n_val_trades >= 5 and vr > best_val_reward:
                pass  # already reset epochs_no_improve when we saved
            else:
                epochs_no_improve += 1
        print_every = 10
        if (ep + 1) % print_every == 0 or ep == 0:
            print(f"Epoch {ep+1} train loss={tl:.4f} acc={ta:.4f} val loss={vl:.4f} acc={va:.4f} reward={vr:.2f} val_trades={n_val_trades}")
        if early_stopping and epochs_no_improve >= patience:
            criterion_str = "val reward" if early_stop_on_reward else "val loss"
            print(f"Early stopping at epoch {ep+1} ({criterion_str} no improvement for {patience} epochs).")
            break
    # If we never saved (model never predicted trade on val), save best by reward so we have some checkpoint
    if best_val_reward == float("-inf") and best_fallback_state is not None:
        torch.save({
            "model_state_dict": best_fallback_state,
            "feature_cols": feature_cols,
            "model_name": model_name,
        }, out_dir / "best.pt")

    # Threshold tuning: pick threshold that maximizes validation reward (used by backtester/forward_test)
    if tune_threshold and (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        best_th, best_vr = TRADE_THRESHOLD, float("-inf")
        for th in thresholds:
            vr, nt = validation_reward_and_trades(model, val_loader, device, threshold=th, transaction_cost=transaction_cost)
            if nt >= 1 and vr > best_vr:
                best_vr = vr
                best_th = th
        with open(out_dir / "trade_threshold.json", "w") as f:
            json.dump({"threshold": best_th, "val_reward": best_vr}, f, indent=2)
        print(f"Threshold tuning: best threshold={best_th} (val_reward={best_vr:.2f})")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    return history, out_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM or Transformer on open-session data.")
    parser.add_argument("model", nargs="?", default="lstm", choices=["lstm", "transformer"], help="Model to train (default: lstm)")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs (default: 250)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay / L2 (default: 1e-4)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience (default: 40)")
    parser.add_argument("--early-stop-on-reward", action="store_true", help="Early stop on val reward instead of val loss")
    parser.add_argument("--tune-threshold", action="store_true", default=True, help="Post-train threshold tuning (default: True)")
    parser.add_argument("--hidden-size", type=int, default=LSTM_HIDDEN, help=f"LSTM Hidden Size (default: {LSTM_HIDDEN})")
    parser.add_argument("--num-layers", type=int, default=LSTM_LAYERS, help=f"LSTM Layers (default: {LSTM_LAYERS})")
    parser.add_argument("--d-model", type=int, default=TX_DMODEL, help=f"Transformer d_model (default: {TX_DMODEL})")
    parser.add_argument("--nhead", type=int, default=TX_NHEAD, help=f"Transformer attention heads (default: {TX_NHEAD})")
    parser.add_argument("--tx-layers", type=int, default=TX_LAYERS, help=f"Transformer layers (default: {TX_LAYERS})")
    parser.add_argument("--exp-dir", type=str, default=None, help="Custom experiment directory (for sweep)")
    parser.add_argument("--transaction-cost", type=float, default=0.1, help="Transaction cost per trade in R units (default: 0.1 for modest friction)")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help=f"Minimum ATR to filter training data (default: {MIN_ATR})")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Inject noise into features for robustness testing (e.g. 0.01)")
    args = parser.parse_args()
    run_train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        early_stop_on_reward=args.early_stop_on_reward,
        tune_threshold=args.tune_threshold,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        tx_layers=args.tx_layers,
        exp_dir=args.exp_dir,
        transaction_cost=args.transaction_cost,
        min_atr=args.min_atr,
        noise_level=args.noise_level,
    )
