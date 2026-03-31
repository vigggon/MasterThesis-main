"""Parameter-matched Transformer for sequence classification."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer encoder: d_model=64, nhead=4, num_layers=2 to match LSTM capacity."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True, # Pre-LN for better stability
        )
        # Pre-LN (norm_first=True) disables nested tensor; set False to avoid the warning
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.proj(x)
        x = self.pos(x)
        x = self.transformer(x)
        # Use last token pooling (specific to current moment) instead of mean
        x = x[:, -1, :]
        x = self.drop(x)
        logits = self.fc(x)
        return logits


def get_transformer(input_size: int, **kwargs) -> TransformerModel:
    return TransformerModel(input_size=input_size, **kwargs)
