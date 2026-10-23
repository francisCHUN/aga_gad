
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import AnomalyGenerator
from .context_attack import ContextAttack
from .semantic_camouflage import SemanticCamouflage
from .bridge_attack import BridgeAttack
from .hybrid_attack import HybridAttack
from .hub_attack import HubAttack
from .spec_builder import SpecBuilder

FIXED_CORE: List[str] = [
    "context_attack",
    "semantic_camouflage",
    "bridge_attack",
    "hybrid_attack",
    "hub_attack",
]

_REGISTRY: Dict[str, AnomalyGenerator] = {}


def register(name: str, builder: AnomalyGenerator) -> None:
    _REGISTRY[name] = builder


def get(name: str) -> Optional[AnomalyGenerator]:
    if name not in _REGISTRY:
        return None
    return _REGISTRY[name]


def known_patterns() -> List[str]:
    return list(_REGISTRY.keys())


def _bootstrap() -> None:
    for b in (ContextAttack(), SemanticCamouflage(), BridgeAttack(),
              HybridAttack(), HubAttack(), SpecBuilder()):
        _REGISTRY[b.name] = b


_bootstrap()