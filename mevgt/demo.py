"""Discover influential branches in a synthetic graph regression model."""
import argparse
import json
from pathlib import Path
from contextlib import contextmanager
import torch
from .types import CircuitID, MetricBundle
from .algorithms.alg1_circuit_discovery import discover_circuits

def run():
    # Fixed three-node graph and two feature channels; no external task adapter.
    adjacency=torch.tensor([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]])
    adjacency=adjacency/adjacency.sum(1,keepdim=True)
    features=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    aggregate=adjacency @ features
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.copy_(torch.tensor([[3.,1.]]))
    target=3*aggregate[:,0]+aggregate[:,1]
    def evaluate():
        with torch.no_grad(): value=-float(torch.mean((model(aggregate).squeeze(-1)-target)**2))
        return MetricBundle(value,{})
    @contextmanager
    def ablate(circuit):
        saved=model.weight.detach().clone()
        with torch.no_grad():model.weight[0,circuit.head]=0
        try:yield
        finally:
            with torch.no_grad():model.weight.copy_(saved)
    circuits=[CircuitID("graph_feature_branch",0,head=i) for i in range(2)]
    rows=discover_circuits(model,{},evaluate,circuits,ablate_fn=ablate,topk=2)
    return {"project":"MeVGT", "scope":"synthetic graph aggregation, not a trained graph transformer", "metric":"negative MSE (higher is better)", "baseline":evaluate().primary,"circuits":[{"branch":r.circuit.head,"score_drop":r.score,"detail":r.detail} for r in rows]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path);a=p.parse_args();text=json.dumps(run(),indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+"\n")
    else:print(text)
if __name__ == "__main__":main()
