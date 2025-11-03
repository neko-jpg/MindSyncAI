import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """Feed-forward block with residual connection and LayerNorm."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class CNNBranch(nn.Module):
    """Convolutional feature extractor operating on log-mel inputs."""

    def __init__(self, n_mels: int, channels, dropout: float):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = channels[-1] * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)
        feats = self.net(x)
        mean_pool = feats.mean(dim=(-1, -2))
        max_pool = feats.amax(dim=(-1, -2))
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.dropout(pooled)


class RNNBranch(nn.Module):
    """Bidirectional GRU branch for temporal modelling."""

    def __init__(self, n_mels: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_mels, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_size * 4

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq = features.transpose(1, 2)
        seq = self.proj(seq)
        self.gru.flatten_parameters()
        out, _ = self.gru(seq)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        merged = torch.cat([mean_pool, max_pool], dim=-1)
        merged = self.norm(merged)
        return self.dropout(merged)


class AttentionBranch(nn.Module):
    """Self-attention branch with token-level weighting."""

    def __init__(self, n_mels: int, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_mels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = ResidualMLP(embed_dim, embed_dim * 2, dropout)
        self.context = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = embed_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq = self.proj(features.transpose(1, 2))
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = seq + attn_out
        seq = self.ff(seq)
        weights = torch.softmax(self.context(seq).squeeze(-1), dim=-1)
        context = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.dropout(context)


class HybridSERNet(nn.Module):
    """
    CNN + BiGRU + Multi-Head Attention マルチブランチ構造。
    Residual/LayerNorm/Dropout を組み込み、深層でも安定して学習できるようにしたモデル。
    """

    def __init__(self, cfg):
        super().__init__()
        n_mels = cfg.features.n_mels
        dropout = cfg.model.dropout

        self.cnn_branch = CNNBranch(n_mels, cfg.model.cnn_channels, dropout)
        self.rnn_branch = RNNBranch(n_mels, cfg.model.rnn_hidden_size, cfg.model.rnn_layers, dropout)
        self.attn_branch = AttentionBranch(n_mels, cfg.model.attention_embed_dim, cfg.model.attention_heads, dropout)

        total_dim = self.cnn_branch.out_dim + self.rnn_branch.out_dim + self.attn_branch.out_dim
        self.merge_norm = nn.LayerNorm(total_dim)
        self.merge_dropout = nn.Dropout(dropout)
        self.projector = nn.Linear(total_dim, cfg.model.classifier_hidden)
        self.residual_head = ResidualMLP(cfg.model.classifier_hidden, cfg.model.classifier_hidden * 2, dropout)
        self.head_norm = nn.LayerNorm(cfg.model.classifier_hidden)
        self.classifier = nn.Linear(cfg.model.classifier_hidden, cfg.dataset.num_classes)

    def forward(self, features: torch.Tensor, lengths=None, waveforms=None, **kwargs) -> torch.Tensor:
        cnn_feat = self.cnn_branch(features)
        rnn_feat = self.rnn_branch(features)
        attn_feat = self.attn_branch(features)

        merged = torch.cat([cnn_feat, rnn_feat, attn_feat], dim=-1)
        merged = self.merge_norm(merged)
        merged = self.merge_dropout(merged)
        merged = self.projector(merged)
        merged = self.residual_head(merged)
        merged = self.head_norm(merged)
        logits = self.classifier(merged)
        return logits
