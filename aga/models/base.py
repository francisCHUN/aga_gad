
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGADModel(nn.Module, ABC):

    name: str = "base"
    supervised: bool = False

    @abstractmethod
    def loss(
        self,
        g,
        x: torch.Tensor,
        mask: torch.Tensor,
        anomaly: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    @torch.no_grad()
    def score(self, g, x: torch.Tensor) -> torch.Tensor:
        ...