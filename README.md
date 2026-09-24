# MeVGT

Graph-model circuit analysis. This minimal release provides an executable, offline synthetic example of one part of the research system. It is **not a complete paper artifact or benchmark reproduction**.

## What runs

Three-node graph → normalized neighborhood aggregation → two feature branches → negative-MSE ablation scores.

Included: Circuit types + original discovery routine + synthetic graph adapter. The command prints a structured JSON report; `--output` writes the same report to a file. Inputs are built into the demonstration, so no datasets, credentials, GPU, API requests or model downloads are needed at runtime.

## Quick start

Python 3.10+; run from this repository directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
mevgt-demo
mevgt-demo --output reports/demo.json
python -m pytest -q
```

Dependency installation requires access to a Python package index. A representative output is checked in at `examples/demo-output.json`. Floating-point results may vary slightly by PyTorch/platform version. The example is reproducible within the tested environment; dependencies are declared but not locked.

## Layout

- `mevgt/`: extracted components and the new `demo.py` entry point.
- `tests/`: component checks and repeatability checks.
- `examples/demo-output.json`: synthetic output from the installed command.
- `PROVENANCE.md` and `SOURCE-MANIFEST.json`: source mapping and modifications.
- `pyproject.toml`: dependencies, package discovery and CLI installation.

## Scope and extension

The demo is a small graph regression model, not a graph Transformer. The extracted discover_circuits API accepts a model, CircuitID list, zero-argument evaluator returning MetricBundle, and ablate_fn(circuit) context manager. Metrics must be higher-is-better. NDCG@10 overrides primary when provided for legacy compatibility. Inspect detail.skipped and detail.error: the legacy implementation records unsupported/failed interventions with score zero.

No private corpora, weights, checkpoints, experiment outputs, original Git history, third-party repository copies, credentials, attack-generation runners or APK modification/deployment pipelines are included. These examples analyze synthetic model or program components only. They do not establish the security of real systems.

## License and attribution

No open-source license has been selected. Confirm ownership, collaborators' publication rights and third-party obligations before making a public release; then add the appropriate LICENSE and required notices. This preparation does not claim license clearance. See `PROVENANCE.md` for details.
