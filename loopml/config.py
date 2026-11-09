"""Central paths and settings. Nothing else in the package hardcodes a path.

Every location can be overridden with an environment variable; the defaults
assume the repo sits next to its external inputs (records archive, ChemScreen,
AFLOW sources) with data/ and models/ generated locally and never committed.

Environment overrides
---------------------
LOOPML_ROOT           repo root (default: parent of the loopml package)
LOOPML_DATA_DIR       generated datasets/artifacts   (default: $ROOT/data)
LOOPML_MODELS_DIR     trained models                 (default: $ROOT/models)
LOOPML_EXTERNAL_DIR   sibling project checkout root  (default: $ROOT/..)
LOOPML_RECORDS        LOOP records archive           (default: $EXTERNAL/records)
LOOPML_CHEMSCREEN     ChemScreen checkout            (default: $EXTERNAL/ChemScreen)
LOOPML_AFLOW_BIN      AFLOW binary (POCC fork)       (default: $EXTERNAL/AFLOW-src_dev/src/aflow)
LOOPML_MACE_MODEL     MACE model file                (default: $MODELS/MACE-matpes-pbe-omat-ft.model)
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


REPO_ROOT = _env_path("LOOPML_ROOT", Path(__file__).resolve().parent.parent)
DATA_DIR = _env_path("LOOPML_DATA_DIR", REPO_ROOT / "data")
MODELS_DIR = _env_path("LOOPML_MODELS_DIR", REPO_ROOT / "models")
EXTERNAL_DIR = _env_path("LOOPML_EXTERNAL_DIR", REPO_ROOT.parent)

RECORDS_DIR = _env_path("LOOPML_RECORDS", EXTERNAL_DIR / "records")
CHEMSCREEN_DIR = _env_path("LOOPML_CHEMSCREEN", EXTERNAL_DIR / "ChemScreen")
CHEMSCREEN_DATA = CHEMSCREEN_DIR / "Data"
AFLOW_BIN = _env_path(
    "LOOPML_AFLOW_BIN", EXTERNAL_DIR / "AFLOW-src_dev" / "src" / "aflow"
)
MACE_MODEL = _env_path(
    "LOOPML_MACE_MODEL", MODELS_DIR / "MACE-matpes-pbe-omat-ft.model"
)
MACE_MODEL_NAME = "mace-matpes-pbe-0"

# --- dataset artifacts (regenerable; gitignored) ---------------------------
PHASE_ROWS_PATH = DATA_DIR / "phase_rows.v3.jsonl"
PHASE_FEATURES_PATH = DATA_DIR / "phase_features.v3.jsonl"
LABELED_DFT_MATCH_PATH = DATA_DIR / "labeled_dft_match.v1.json"
BASELINE_RESULTS_PATH = DATA_DIR / "phase_baselines_results.v2.json"

# --- MLIP layer -------------------------------------------------------------
MLIP_REFS_PATH = DATA_DIR / "mlip_refs.v1.json"
POCC_ROOT = DATA_DIR / "pocc"
POCC_WORK_ROOT = DATA_DIR / "pocc_work"
PROTO_TEMPLATE_CACHE = DATA_DIR / "proto_templates.json"

# --- trained models (gitignored) --------------------------------------------
COMP_MODEL_PATH = MODELS_DIR / "comp-v1.pkl"
BASELINE_REGISTRY_PATH = DATA_DIR / "baselines.v1.json"
