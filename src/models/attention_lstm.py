import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention: Query = Key = Value = x
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class AttentionLSTM(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 32, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        # Unidirectional LSTM encoder to preserve strict temporal causality
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.attention = MultiHeadAttentionBlock(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input: [Batch, Seq_Len=20, Input_Dim]
        lstm_out, _ = self.lstm(x)  # [Batch, Seq_Len, Hidden_Dim]
        attn_out = self.attention(lstm_out)  # [Batch, Seq_Len, Hidden_Dim]
        last_representation = attn_out[:, -1, :]  # Final temporal state at t
        return self.head(last_representation).squeeze(-1)