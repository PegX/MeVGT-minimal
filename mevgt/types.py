from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class CircuitID:
    """Unified circuit identifier (extendable)."""
    module: str           # e.g., 'attn', 'gnn', 'edge_gate', 'mlp'
    layer: int
    head: Optional[int] = None
    tag: Optional[str] = None  # edge-type/pathway/etc.

@dataclass
class Batch:
    graph: Any
    labels: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class MetricBundle:
    primary: float
    metrics: Dict[str, float]
