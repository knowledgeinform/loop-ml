# loop-ml — synthesizability screening for high-entropy oxides

Predict whether a high-entropy oxide composition synthesizes as a
single phase, from the chemical formula alone. Labels come from
experimental synthesis outcomes. Structure information enters as
computed descriptors: AFLOW POCC samples atomic arrangements on common
oxide lattices, a MACE interatomic potential relaxes them (cell and
positions), and the relaxed-energy spread becomes the EFA feature.

## Pipeline

```
composition ──> prototypes ──> POCC arrangements ──> MACE relax ──> EFA ──> classifier
 (formula)     (5 lattices,     (24 sampled per      (batched,       1/σ     (comp vs
               site split)      lattice)              resumable)              comp+EFA)
```

1. `1-collect-data.py` — synthesis records → labeled rows jsonl with
   deterministic grouped splits.
2. `2-build-prototypes.py` — each composition onto each lattice:
   PARTCAR with cations split over sublattices, `aflow --pocc` samples
   24 orderings (deterministic seed).
3. `3-relax-prototypes.py` — batched TorchSim BFGS with Frechet cell
   filter, one worker per GPU, work-stealing chunks, resumable.
4. `4-score-efa.py` — join results to ensembles, compute EFA per
   composition per lattice (inverse dg-weighted stdev of relaxed
   energy/atom, AFLOW `pocc::getEFA`) + candidate table.
5. `5-train-classifier.py` — composition features (+ optional EFA) →
   grouped-CV metrics on held-out splits.
6. `6-generate-compositions.py` — enumerate the screening space
   (equimolar cation mixtures from a pool) in rows format.
7. `7-relax-compositions.py` — step 3 over the screening compositions
   (shared corpus, skips already-relaxed structures).
8. `8-evaluate-recommendations.py` — final model trained on all labeled
   rows → ranked recommendations for the screening compositions.

Charge-feasibility filtering (`loopml/filters.py`) and the
degeneracy-weighted Boltzmann pick of the per-composition lattice
(298.15 K) happen at scoring/training time. Relaxation is exhaustive:
policy changes never require re-relaxing.

## Running

`uv venv && uv sync --extra mlip`, then `uv run`. Every script takes
`--help`; all paths below are the defaults.

Labeled training/validation:

```bash
uv run python 1-collect-data.py               # records/ -> data/phase_rows.v3.jsonl
uv run python 2-build-prototypes.py           # -> data/pocc/<chem>__<proto>/
uv run python 3-relax-prototypes.py           # -> data/mlip_work/   (GPU)
uv run python 4-score-efa.py                  # -> data/efa.jsonl
uv run python 5-train-classifier.py           # -> data/classifier_results.json
```

Screening:

```bash
uv run python 6-generate-compositions.py      # -> data/compositions.jsonl
uv run python 2-build-prototypes.py --rows data/compositions.jsonl
uv run python 7-relax-compositions.py         # -> data/mlip_work/
uv run python 4-score-efa.py
uv run python 8-evaluate-recommendations.py   # -> data/recommendations.jsonl
```

Steps 1/2/4/5/6/8 are CPU-only; 3/7 want a GPU.

## Default file locations

| file | produced by | consumed by |
|---|---|---|
| `records/` (external, `$LOOPML_RECORDS`) | the LOOP archive | 1, 5 |
| `data/phase_rows.v3.jsonl` | 1 | 2, 5, 8 |
| `data/compositions.jsonl` | 6 | 2, 8 |
| `data/pocc/<chem>__<proto>/` | 2 | 3/7, 4 |
| `data/mlip_work/bshard*.jsonl` | 3/7 | 4 |
| `data/efa.jsonl` (+ coverage json) | 4 | 5, 8 |
| `data/classifier_results.json` | 5 | — |
| `data/recommendations.jsonl` (+ report) | 8 | — |

Step 5 uses `data/efa.jsonl` when present (`--no-efa` to skip).
Relax tasks and results are keyed by task id, `<ensemble dir>/<POSCAR
file>`; a result joins back to its structure by that key alone.

## Repository layout

```
loopml/
  protogen.py   ANRL templates, PARTCAR writer, POCC driver, canonical_key
  mlip.py       MACE/TorchSim relaxation, EFA math, results ledger
  filters.py    oxidation-state charge feasibility
  config.py     env-var paths ($LOOPML_RECORDS, $LOOPML_AFLOW_BIN, ...)
tests/          unit tests (uv run pytest tests -q)
```

