"""Compute classification metrics and parameter counts for thesis results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from utils import (
    PROCESSED_DIR, BEST_TRANSFORMER_DIR, BEST_LSTM_DIR,
    TRADE_THRESHOLD, NUM_CLASSES,
    LSTM_HIDDEN, LSTM_LAYERS, TX_DMODEL, TX_NHEAD, TX_LAYERS
)
from models import get_lstm, get_transformer


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_backtest_data():
    data = np.load(PROCESSED_DIR / "backtest.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = torch.from_numpy(data["y"]).long()
    feature_cols = list(data["feature_cols"])
    return X, y, feature_cols


def load_model(model_name, n_features, device):
    if model_name == "lstm":
        ckpt = torch.load(BEST_LSTM_DIR / "best.pt", map_location=device, weights_only=False)
        model = get_lstm(n_features, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, num_classes=NUM_CLASSES)
    else:
        ckpt = torch.load(BEST_TRANSFORMER_DIR / "best.pt", map_location=device, weights_only=False)
        model = get_transformer(n_features, d_model=TX_DMODEL, nhead=TX_NHEAD, num_layers=TX_LAYERS, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def pred_3class(probs, th):
    p_long = probs[:, 1]
    p_short = probs[:, 2]
    trade = (p_long >= th) | (p_short >= th)
    pred = np.where(~trade, 0, np.where(p_long >= p_short, 1, 2))
    return pred.astype(np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y_true, feature_cols = load_backtest_data()
    n_features = X.shape[2]
    
    print("=" * 60)
    print("PARAMETER COUNTS")
    print("=" * 60)
    
    for name in ["lstm", "transformer"]:
        model = load_model(name, n_features, device)
        params = count_params(model)
        print(f"{name.upper()}: {params:,} trainable parameters")
    
    print()
    print("=" * 60)
    print("CLASSIFICATION METRICS (Backtest Set)")
    print("=" * 60)
    
    class_names = ["Hold", "Long", "Short"]
    
    for name in ["transformer", "lstm"]:
        model = load_model(name, n_features, device)
        
        with torch.no_grad():
            logits = model(X.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        y_pred = pred_3class(probs, TRADE_THRESHOLD)
        y_np = y_true.numpy()
        
        print(f"\n--- {name.upper()} ---")
        print(f"Threshold: {TRADE_THRESHOLD}")
        print(f"Total samples: {len(y_np)}")
        print(f"Predicted distribution: Hold={np.sum(y_pred==0)}, Long={np.sum(y_pred==1)}, Short={np.sum(y_pred==2)}")
        print(f"Actual distribution: Hold={np.sum(y_np==0)}, Long={np.sum(y_np==1)}, Short={np.sum(y_np==2)}")
        print()
        print(classification_report(y_np, y_pred, target_names=class_names, digits=3, zero_division=0))
        print("Confusion Matrix:")
        cm = confusion_matrix(y_np, y_pred, labels=[0, 1, 2])
        print(f"         Pred Hold  Pred Long  Pred Short")
        for i, row_name in enumerate(class_names):
            print(f"  {row_name:6s}  {cm[i,0]:8d}  {cm[i,1]:9d}  {cm[i,2]:10d}")


if __name__ == "__main__":
    main()
