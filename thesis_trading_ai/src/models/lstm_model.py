"""Parameter-matched LSTM with Bidirectional Attention for sequence classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention over LSTM hidden states to focus on relevant timesteps."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (B, H)
        return context


class LSTMModel(nn.Module):
    """Bidirectional LSTM with temporal attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        # Bidirectional doubles hidden size
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = TemporalAttention(hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)  # (B, T, H*2)
        # Project to original hidden size
        out = self.proj(out)  # (B, T, H)
        # Use attention-weighted context
        context = self.attention(out)  # (B, H)
        context = self.ln(context)
        context = self.drop(context)
        logits = self.fc(context)
        return logits


def get_lstm(input_size: int, **kwargs) -> LSTMModel:
    return LSTMModel(input_size=input_size, **kwargs)


