
from __future__ import annotations

from typing import Dict, Type

from .base import BaseGADModel
from .unsupervised import DominantModel, CoLAModel
from .supervised import GCNModel, GATModel, MLPModel, GraphSAGEModel, BWGNNModel, BWGNNHeteroModel, AMNetModel, GDNModel, CAREGNNModel, PCGNNModel, SAGADModel, GHRNModel, SpaceGNNModel, GAGAModel, PMPModel, PAGEModel, DiGInGNNModel, GADAMModel
from .supervised.h2fdetector import H2FDetectorModel


MODELS: Dict[str, Type[BaseGADModel]] = {
    "dominant": DominantModel,
    "cola": CoLAModel,
    "gcn": GCNModel,
    "gat": GATModel,
    "mlp": MLPModel,
    "graphsage": GraphSAGEModel,
    "bwgnn": BWGNNModel,
    "bwgnn_hetero": BWGNNHeteroModel,
    "amnet": AMNetModel,
    "h2fdetector": H2FDetectorModel,
    "gdn": GDNModel,
    "care_gnn": CAREGNNModel,
    "pcgnn": PCGNNModel,
    "sagad": SAGADModel,
    "ghrn": GHRNModel,
    "spacegnn": SpaceGNNModel,
    "gaga": GAGAModel,
    "pmp": PMPModel,
    "page": PAGEModel,
    "dig_in_gnn": DiGInGNNModel,
    "gadam": GADAMModel,
}


def build_model(name: str, **kwargs) -> BaseGADModel:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name](**kwargs)