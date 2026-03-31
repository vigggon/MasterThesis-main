"""
Forward testing = **live only**. Model is NOT retrained.
Not related to training (2019-08-12–2023) or backtest (2024-01-01 to 2024-08-09). Live deployment: load model,
build features from past only, predict sequentially on each new bar.
Both models can run at the same time on the same new data—route LSTM to one account,
Transformer to the other. Use run_live_forward.py to run live; this module provides the API.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from typing import Tuple, Union

import json

from utils import (
    PROCESSED_DIR, EXPERIMENTS_ROOT, TRADE_THRESHOLD, NUM_CLASSES,
    BEST_TRANSFORMER_DIR, BEST_LSTM_DIR,
    LSTM_HIDDEN, LSTM_LAYERS, TX_DMODEL, TX_NHEAD, TX_LAYERS
)
from models import get_lstm, get_transformer


def get_threshold(model_name: str) -> float:
    """Use threshold from trade_threshold.json if present (tuned in train), else utils default."""
    exp_dir = BEST_TRANSFORMER_DIR if model_name == "transformer" else BEST_LSTM_DIR
    path = exp_dir / "trade_threshold.json"
    if path.exists():
        with open(path) as f:
            return float(json.load(f)["threshold"])
    return TRADE_THRESHOLD


def load_both_models(device=None):
    """
    Load sweep-winning LSTM and Transformer.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(PROCESSED_DIR / "train.npz", allow_pickle=True)
    feature_cols = list(data["feature_cols"])
    n_features = data["X"].shape[2]

    # Best LSTM
    lstm = get_lstm(n_features, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    ckpt_lstm = torch.load(BEST_LSTM_DIR / "best.pt", map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt_lstm["model_state_dict"])
    lstm = lstm.to(device).eval()

    # Best Transformer
    transformer = get_transformer(n_features, d_model=TX_DMODEL, nhead=TX_NHEAD, num_layers=TX_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    ckpt_tx = torch.load(BEST_TRANSFORMER_DIR / "best.pt", map_location=device, weights_only=False)
    transformer.load_state_dict(ckpt_tx["model_state_dict"])
    transformer = transformer.to(device).eval()

    return lstm, transformer, feature_cols, device


def predict_on_new_data(
    model: torch.nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    model_name: str | None = None,
) -> Union[int, np.ndarray]:
    """
    Single model: predict on new bar(s). Returns 0=hold, 1=long, 2=short (one sample or batch).
    If model_name is given (e.g. 'lstm' or 'transformer'), uses tuned threshold from train when present.
    """
    th = get_threshold(model_name) if model_name else TRADE_THRESHOLD
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    X = X.to(device)
    if X.dim() == 2:
        X = X.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    p_long, p_short = probs[:, 1], probs[:, 2]
    trade = (p_long >= th) | (p_short >= th)
    pred = np.where(~trade, 0, np.where(p_long >= p_short, 1, 2)).astype(np.int64)
    return int(pred[0]) if pred.shape[0] == 1 else pred


def predict_both_models_on_new_data(
    lstm_model: torch.nn.Module,
    transformer_model: torch.nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    Same new data for both models. Returns (pred_lstm, pred_transformer).
    Uses tuned threshold per model when trade_threshold.json exists.
    """
    pred_lstm = predict_on_new_data(lstm_model, X, device, model_name="lstm")
    pred_transformer = predict_on_new_data(transformer_model, X, device, model_name="transformer")
    return pred_lstm, pred_transformer


if __name__ == "__main__":
    print("Forward testing is live only. Run:  python run_live_forward.py")
    print("Or use:  from forward_test import load_both_models, predict_both_models_on_new_data")
