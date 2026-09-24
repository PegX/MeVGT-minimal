from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional
import torch
from mevgt.types import CircuitID, MetricBundle

@dataclass
class CircuitScore:
    circuit: CircuitID
    score: float
    detail: Dict[str, Any] = None


def discover_circuits(model: torch.nn.Module,
                      hook_specs: Dict[CircuitID, Dict[str, Any]],
                      eval_fn,
                      circuits: List[CircuitID],
                      ablate_fn: Optional[Callable[[CircuitID], Any]] = None,
                      topk: int = 20) -> List[CircuitScore]:
    """
    Score each circuit in ``circuits`` by the drop in the primary metric when
    that circuit is ablated.

    Ablation strategy (in order of preference):
    1. ``ablate_fn(circuit)`` context-manager from the adapter — supports any
       circuit type including head-level circuits.
    2. ``_mevgt_cache_layers`` / ``_mevgt_ablate_layer`` model flags — legacy
       LightGCN-style fallback for ``type=="layer"`` layer-only specs.
    3. Skip (score 0) if neither is available.

    Head-level circuits (``circuit.head is not None``) are handled transparently
    as long as the adapter's ``ablate_circuit()`` supports them.
    """
    clean = eval_fn()
    # Use primary as the canonical scalar; NDCG@10 if present for backward compat.
    clean_base = float(clean.metrics.get("NDCG@10", clean.primary))

    out: List[CircuitScore] = []

    for c in circuits:
        spec = hook_specs.get(c, {})
        circuit_type = spec.get("type", "")
        is_layer = circuit_type in {"layer", "layer_subset", "edge_row_subset", "layer_group"}
        is_head = c.head is not None

        # ----------------------------------------------------------------
        # Build ablation context
        # ----------------------------------------------------------------
        if ablate_fn is not None:
            # Adapter-provided ablation: handles all circuit types including heads.
            ctx = ablate_fn(c)
            ablate_idx = None
        elif is_layer and not is_head:
            # Legacy flag-based ablation (LightGCN / CoLaKG style).
            layer_idx = int(spec.get("idx", c.layer))
            ablate_idx = layer_idx + 1
            prev_cache = getattr(model, "_mevgt_cache_layers", False)
            prev_abl = getattr(model, "_mevgt_ablate_layer", None)
            model._mevgt_cache_layers = True
            model._mevgt_ablate_layer = ablate_idx
            ctx = contextlib.nullcontext()
        else:
            # No ablation method available — record as skipped.
            out.append(CircuitScore(circuit=c, score=0.0, detail={"skipped": True, "type": circuit_type}))
            continue

        # ----------------------------------------------------------------
        # Run ablated evaluation
        # ----------------------------------------------------------------
        try:
            with ctx:
                ab = eval_fn()
        except Exception as e:
            out.append(CircuitScore(circuit=c, score=0.0, detail={"error": str(e)}))
            continue
        finally:
            if ablate_fn is None and not is_head:
                model._mevgt_cache_layers = prev_cache
                model._mevgt_ablate_layer = prev_abl

        ab_base = float(ab.metrics.get("NDCG@10", ab.primary))
        drop = clean_base - ab_base

        detail: Dict[str, Any] = {
            "clean_primary": clean_base,
            "ablated_primary": ab_base,
            "type": circuit_type,
        }
        if ablate_idx is not None:
            detail["ablate_idx"] = ablate_idx
        if is_head:
            detail["head"] = c.head

        out.append(CircuitScore(circuit=c, score=float(drop), detail=detail))

    out.sort(key=lambda x: x.score, reverse=True)
    return out[: min(topk, len(out))]
