from contextlib import contextmanager
import torch
from mevgt.types import CircuitID, MetricBundle
from mevgt.algorithms.alg1_circuit_discovery import discover_circuits

def test_ablation_ranks_and_restores():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad(): model.weight.copy_(torch.tensor([[3., 1.]]))
    circuits = [CircuitID('linear', 0, head=i) for i in range(2)]
    @contextmanager
    def ablate(circuit):
        old = model.weight.detach().clone()
        with torch.no_grad(): model.weight[0, circuit.head] = 0
        try: yield
        finally:
            with torch.no_grad(): model.weight.copy_(old)
    def evaluate():
        return MetricBundle(float(model(torch.ones(1, 2)).item()), {})
    results = discover_circuits(model, {}, evaluate, circuits, ablate_fn=ablate, topk=2)
    assert [r.score for r in results] == [3., 1.]
    assert evaluate().primary == 4
