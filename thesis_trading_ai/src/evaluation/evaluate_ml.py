"""
ML metrics on backtest (2024-01-01 to 2024-08-09). Forward test = live only (no file).
Uses trained model checkpoint; no retraining.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import json
from utils import (
    PROCESSED_DIR, RESULTS_ROOT, TRADE_THRESHOLD, NUM_CLASSES,
    BEST_TRANSFORMER_DIR, BEST_LSTM_DIR
)
from models import get_lstm, get_transformer


def _get_threshold(model_name: str) -> float:
    exp_dir = BEST_TRANSFORMER_DIR if model_name == "transformer" else BEST_LSTM_DIR
    path = exp_dir / "trade_threshold.json"
    if path.exists():
        with open(path) as f:
            return float(json.load(f)["threshold"])
    return TRADE_THRESHOLD


def _pred_3class_np(probs: np.ndarray, th: float) -> np.ndarray:
    p_long, p_short = probs[:, 1], probs[:, 2]
    trade = (p_long >= th) | (p_short >= th)
    pred = np.where(~trade, 0, np.where(p_long >= p_short, 1, 2)).astype(np.int64)
    return pred


def load_model_and_data(model_name: str, split: str = "backtest"):
    data = np.load(PROCESSED_DIR / f"{split}.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = data["y"]
    times = data["times"]
    feature_cols = list(data["feature_cols"])

    exp_dir = BEST_TRANSFORMER_DIR if model_name == "transformer" else BEST_LSTM_DIR
    ckpt = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
    n_features = X.shape[2]
    from utils import (
        LSTM_HIDDEN, LSTM_LAYERS, TX_DMODEL, TX_NHEAD, TX_LAYERS, NUM_CLASSES
    )
    if model_name == "lstm":
        model = get_lstm(n_features, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    else:
        model = get_transformer(n_features, d_model=TX_DMODEL, nhead=TX_NHEAD, num_layers=TX_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()
        pred = _pred_3class_np(probs, _get_threshold(model_name))
    return pred, y, times, feature_cols


def compute_metrics(y_true, y_pred):
    # 3-class: macro avg for precision/recall/f1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_evaluate_ml(model_name: str = "lstm", split: str = "backtest") -> dict:
    pred, y, _, _ = load_model_and_data(model_name, split)
    metrics = compute_metrics(y, pred)
    out_dir = RESULTS_ROOT / "ml_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{split}_metrics.json"
    import json
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    run_evaluate_ml(model_name=name, split="backtest")
