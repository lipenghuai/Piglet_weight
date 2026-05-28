"""
Mixture-of-Experts (MoE) modules, routing layers, and compatibility shims.

This module provides several MoE variants and routers optimized for inference efficiency,
plus backward-compatibility aliases so legacy checkpoints can be loaded without changes.
"""

from .modules import (
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    A2C2fMoE,
    ABlockMoE,
)

from .experts import (
    OptimizedSimpleExpert,
    FusedGhostExpert,
    SimpleExpert,
    GhostExpert,
    InvertedResidualExpert,
    EfficientExpertGroup,
    DepthwiseSeparableConv
)

from .routers import (
    UltraEfficientRouter,
    BaseRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    AdvancedRoutingLayer,
    DynamicRoutingLayer
)

from .utils import (
    FlopsUtils,
    get_safe_groups,
    BatchedExpertComputation
)

import torch


MOE_LAYER_NAMES = (
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "MOE",
    "ES_MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
    "A2C2fMoE",
    "ABlockMoE",
)

YOLO_REPEAT_LAYER_NAMES = ("A2C2fMoE",)


def iter_moe_aux_losses(model):
    """Yield auxiliary MoE losses exposed by modules through an aux_loss property."""
    for module in model.modules():
        aux_loss = getattr(module, "aux_loss", None)
        if isinstance(aux_loss, torch.Tensor):
            yield aux_loss


def collect_moe_aux_loss(model, device=None):
    """Return the summed auxiliary MoE loss for a model."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    total = torch.zeros((), device=device)
    for aux_loss in iter_moe_aux_losses(model):
        total = total + aux_loss.to(device=total.device, dtype=total.dtype)
    return total

__all__ = [
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
    "A2C2fMoE",
    "ABlockMoE",
    "OptimizedSimpleExpert",
    "FusedGhostExpert",
    "SimpleExpert",
    "GhostExpert",
    "InvertedResidualExpert",
    "EfficientExpertGroup",
    "DepthwiseSeparableConv",
    "UltraEfficientRouter",
    "BaseRouter",
    "EfficientSpatialRouter",
    "AdaptiveRoutingLayer",
    "LocalRoutingLayer",
    "AdvancedRoutingLayer",
    "DynamicRoutingLayer",
    "FlopsUtils",
    "get_safe_groups",
    "BatchedExpertComputation",
    "MOE_LAYER_NAMES",
    "YOLO_REPEAT_LAYER_NAMES",
    "collect_moe_aux_loss",
    "iter_moe_aux_losses",
]
