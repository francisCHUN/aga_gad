# AGA: Agent-Guided Adaptive Anomaly Generation

**Agent-Guided Adaptive Anomaly Generation (AGA)** is a multi-agent framework that combines pattern discovery, evidence-based review, and flexible construction to augment training supervision with under-represented anomaly patterns for Graph Anomaly Detection (GAD).

```
Dataset → Inspector (deterministic stats) → Gap Analyzer (missing patterns, LLM; each gap = a hypothesis)
        → [review flow] Gap Reviewer (evidence-grounded verification: human-in-the-loop / automatic refinement loop)
        → Strategy Planner (generation plan, LLM; approved gaps only)
        → Generator (DGL builders + AnomalySpec interpreter, deterministic) → augmented graph (train only)
```

## Quick start

DGL requires Python 3.12. Verified environment:

```bash
/opt/miniconda3/envs/python312/bin/python   # dgl 2.2.1, torch 2.3, pydantic 2.10 ...
pip install -r requirements.txt              # in that env

PY=/opt/miniconda3/envs/python312/bin/python
```

```bash
# 1. Plan only (no graph mutation, keyless — dummy backend):
$PY -m aga.main --backend dummy --report-only --data-dir data/

# 2. Full generation + injection (centroid-blend features, keyless):
$PY -m aga.main --backend dummy --data-dir data/

# 3. Full pipeline + baseline-vs-augmented paired training comparison
#    (frozen test set, AUC/AP/F1, 5 repeated runs by default):
$PY -m aga.main --backend dummy --train-compare --model gcn --data-dir data/

# 4. Text-generation path: the LLM WRITES each synthetic node's text
#    (context_attack & semantic_camouflage), then sentence-transformers embeds it:
$PY -m aga.main --backend sohoyo --text-gen --train-compare --data-dir data/
```

For real backends prefer driving everything from `configs/default.yaml`:  
`--config configs/default.yaml --train-compare`.

## Three review loops

```bash
# A. Human review loop (recommended)
$PY -m aga.main --review-only --review-dir review_run --backend sohoyo --llm-precheck
#   → open review_run/review_report.html, curate the LLM-prefilled verdicts,
#     export both feedback files back into review_run/
$PY -m aga.main --resume-review review_run --backend sohoyo --train-compare --model h2fdetector

# B. Fully-automatic gap refinement loop (LLM Reviewer ↔ Gap Refiner rounds,
#    no pause, CI-safe): stops on convergence or the round budget; the refined
#    gap set is handed to the Planner UNCONSTRAINED. Audit bundle in review_run/loop/
$PY -m aga.main --gap-loop --review-dir review_run --backend sohoyo --train-compare

# C. Hybrid: run the loop, then PAUSE for a human curation of the refined gap
#    set (confirmed=keep / rejected=drop / revised_confirmed=edit), export
#    loop/gap_feedback.json
$PY -m aga.main --review-gap-loop --review-dir review_run --backend sohoyo
$PY -m aga.main --resume-gap-loop review_run --backend sohoyo --train-compare --model h2fdetector

# Without review flags the legacy one-shot flow runs (Gap → Strategy directly),
# unchanged from the original behaviour.
```

Related CLI knobs: `--trust-llm-review` (accept LLM-prefilled verdicts as-is),  
`--accept-insufficient` (explicit degraded approval for evidence-dead gaps —  
every verdict and the cache digest record it, never silently),  
`--coverage-criterion` (inject a coverage requirement, e.g. "every unmeasured  
family must be present").

## Layout

```
aga/
  data/           loader.py, analyze.py, review_sampling.py, review_evidence.py
  llm/            client.py, registry.py (dummy/antchat/sohoyo/stub backends), prompts.py
  agents/         inspector, gap_analyzer, gap_reviewer, gap_refiner,
                  llm_reviewer, strategy_planner, generator
  patterns/       base.py, registry.py, spec_builder.py (AnomalySpec DSL),
                  text_embedder.py, context/semantic/bridge/hybrid/hub attacks
  augmentation/   inject.py (train-only)
  orchestration/  schema.py (pydantic contracts), pipeline.py, gap_loop.py, cache.py
  models/         base.py, registry.py, split.py, train.py
    unsupervised/ dominant.py, cola.py
    supervised/   gcn, gat, mlp, graphsage, bwgnn, amnet, h2fdetector, gdn,
                  care_gnn, pcgnn, sagad, ghrn, spacegnn, gaga, pmp, page,
                  dig_in_gnn, gadam
  experiments/    attack_simulation, pattern_shift_simulation,
                  random_augment / dagad / flag / gadam experiments
  tests/          21 test files (masks, pipeline, review, cache, per-model, ...)
configs/default.yaml   # full configuration (dataset, split, LLM, generation budget,
                       # model hyperparameters, review, cache)
aga/main.py           # CLI entry
run.sh                 # ready-made commands for the three review loops
paper/LARS-GAD.md      # full paper draft (methods faithful to the code; numbers are placeholders)
```

