from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SCALE_NAMES = ("second", "short", "medium", "long")


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.pad(values, (self.left_padding, 0))
        hidden = self.temporal(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        hidden = values + hidden
        return self.norm(hidden.transpose(1, 2)).transpose(1, 2)


class ScaleTCN(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(channels, kernel_size, dilation, dropout)
            for dilation in dilations
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.transpose(1, 2)


class DynamicHistoryModel(nn.Module):
    """Four-scale TCN + Transformer dynamic history encoder."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        history = config["history"]
        target = config["target"]
        vector_dim = int(model["vector_dim"])
        channels = int(model["tcn_channels"])
        if channels != vector_dim:
            raise ValueError("首版要求 tcn_channels 与 vector_dim 相等")
        scale_config = {str(item["name"]): item for item in history["scales"]}
        if set(scale_config) != set(SCALE_NAMES):
            raise ValueError(f"历史尺度必须为 {SCALE_NAMES}")
        self.scale_lengths = {
            name: int(scale_config[name]["bucket_count"]) for name in SCALE_NAMES
        }
        dropout = float(model["dropout"])
        dilations = [int(item) for item in model["tcn_dilations"]]
        kernel_size = int(model["tcn_kernel_size"])
        self.scale_tcns = nn.ModuleDict(
            {
                name: ScaleTCN(channels, kernel_size, dilations, dropout)
                for name in SCALE_NAMES
            }
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, vector_dim))
        self.scale_embeddings = nn.Parameter(torch.empty(len(SCALE_NAMES), vector_dim))
        self.position_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(length, vector_dim))
                for name, length in self.scale_lengths.items()
            }
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vector_dim,
            nhead=int(model["transformer_heads"]),
            dim_feedforward=int(model["transformer_ffn_dim"]),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(model["transformer_layers"]),
        )
        self.transformer_norm = nn.LayerNorm(vector_dim)
        state_hidden = int(model["state_hidden_dim"])
        self.state_encoder = nn.Sequential(
            nn.Linear(int(model["state_input_dim"]), state_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_hidden, state_hidden),
            nn.LayerNorm(state_hidden),
        )
        fusion_hidden = int(model["fusion_hidden_dim"])
        self.fusion = nn.Sequential(
            nn.LayerNorm(vector_dim + state_hidden),
            nn.Linear(vector_dim + state_hidden, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, vector_dim),
        )
        self.access_head = nn.Linear(vector_dim, 1)
        self.time_bin_count = len(target["time_boundaries_seconds"]) - 1
        if self.time_bin_count != 9:
            raise ValueError("04/05/06 接口要求 9 个访问时间档和 1 个无访问档")
        self.time_head = nn.Linear(vector_dim, self.time_bin_count)
        count_hidden = int(model["count_hidden_dim"])
        self.count_head = nn.Sequential(
            nn.Linear(vector_dim, count_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(count_hidden, 1),
            nn.Softplus(),
        )
        self._initialize_tokens()

    def _initialize_tokens(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)
        for parameter in self.position_embeddings.values():
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(
        self,
        second_counts: torch.Tensor,
        short_counts: torch.Tensor,
        medium_counts: torch.Tensor,
        long_counts: torch.Tensor,
        history_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "second": second_counts,
            "short": short_counts,
            "medium": medium_counts,
            "long": long_counts,
        }
        batch_size = second_counts.shape[0]
        tokens: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]
        for scale_index, name in enumerate(SCALE_NAMES):
            values = inputs[name]
            expected_length = self.scale_lengths[name]
            if values.ndim != 3 or values.shape[1:] != (expected_length, 1):
                raise ValueError(
                    f"{name}_counts 形状必须为 [B,{expected_length},1]，实际 {tuple(values.shape)}"
                )
            encoded = self.scale_tcns[name](values)
            encoded = encoded + self.scale_embeddings[scale_index].view(1, 1, -1)
            encoded = encoded + self.position_embeddings[name].unsqueeze(0)
            tokens.append(encoded)
        combined = torch.cat(tokens, dim=1)
        temporal = self.transformer_norm(self.transformer(combined)[:, 0])
        state = self.state_encoder(history_state)
        hidden = self.fusion(torch.cat((temporal, state), dim=-1))
        vectors = F.normalize(hidden, p=2, dim=-1, eps=1e-12)
        access_logits = self.access_head(hidden).squeeze(-1)
        time_logits = self.time_head(hidden)
        predicted_log_counts = self.count_head(hidden).squeeze(-1)
        access_probability = torch.sigmoid(access_logits)
        conditional_time = torch.softmax(time_logits, dim=-1)
        next_access_probs = torch.cat(
            (
                access_probability.unsqueeze(-1) * conditional_time,
                (1.0 - access_probability).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "hidden": hidden,
            "vectors": vectors,
            "access_logits": access_logits,
            "time_logits": time_logits,
            "predicted_log_counts": predicted_log_counts,
            "next_access_probs": next_access_probs,
            "expected_access_counts": torch.expm1(predicted_log_counts),
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
