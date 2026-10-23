
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):

    name: str = "agent"

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError