from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .errors import ArtifactCompatibilityError
from .model import ActorConfig, FolderCacheActor


class SupervisedActor:
    def __init__(self, model: FolderCacheActor, metadata: Mapping[str, Any], device: torch.device) -> None:
        self.model = model.eval()
        self.metadata = dict(metadata)
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "auto") -> "SupervisedActor":
        resolved = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        checkpoint = torch.load(Path(checkpoint_path), map_location=resolved, weights_only=False)
        if checkpoint.get("schema_version") != "folder-cache-actor-checkpoint/v2":
            raise ArtifactCompatibilityError("不支持的Actor检查点")
        config = ActorConfig.from_mapping(checkpoint["model_config"])
        model = FolderCacheActor(config).to(resolved)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(model, checkpoint, resolved)

    @torch.inference_mode()
    def predict(
        self,
        context_features: np.ndarray | torch.Tensor,
        context_valid_mask: np.ndarray | torch.Tensor,
        time_features: np.ndarray | torch.Tensor,
    ) -> dict[str, np.ndarray]:
        features = torch.as_tensor(context_features, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(context_valid_mask, dtype=torch.bool, device=self.device)
        times = torch.as_tensor(time_features, dtype=torch.float32, device=self.device)
        result = self.model(features, valid, times)
        return {name: value.float().cpu().numpy() for name, value in result.items()}
