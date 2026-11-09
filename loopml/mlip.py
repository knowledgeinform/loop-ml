"""MACE MLIP energy layer (ad hoc today; service-izable later).

MACE-MATPES-PBE-0 (mace-foundations, ASL license) turns a structure into a
relaxed energy. EFA (AFLOW "entropy forming ability") follows the POCC
protocol exactly (aflow_pocc.cpp getEFA/setEFA): for an ensemble of configs
of one composition, with x_i the relaxed enthalpy per atom (H_atom_relax)
and dg_i the degeneracy weight,

    Hmix    = sum_i dg_i x_i / sum_i dg_i
    sigma   = sqrt( sum_i dg_i (x_i - Hmix)^2 / (sum_i dg_i - 1) )
    EFA     = 1 / sigma
    DEED    = sqrt(EFA / d2h)

Note EFA depends only on the SPREAD of per-atom energies across the ensemble,
so any per-atom constant baseline (element references, etc.) cancels in the
sigma: EFA is computed directly from raw relaxed per-atom energies, never
from formation energies. Formation energies (eform_i = x_i - sum_x E_atom(ref),
using the SAME potential for the refs) are a separate quantity used only for
per-structure error analysis, not for EFA.

Hard rules (see README roadmap): reference energies come from the SAME
potential (never mix MLIP and DFT on one scale); MLIP energies are noisy
features, never targets; the potential name is recorded with every number.

Execution model (deliberately database-free): a task JSONL (one line per
structure, stable task id <ensemble-dir>/<POSCAR-file>) is split round-robin
into shard files under a work
dir; one dumb worker process per shard appends resumable JSONL results
{tid, energy_atom, ...} that the reporting step joins back by tid.

Two relaxers, one JSONL contract: `relax-shard` runs ASE BFGS serially per
structure; `relax-batch` runs TorchSim BFGS with a Frechet cell filter over
many structures per device step (autograd stress, float32 like the ASE
path). torch-sim <=0.6.2 mis-slices the padded BFGS eigensolve for
mixed-size batches (smaller systems' cell DOFs freeze), so batched groups
are bucketed to equal natoms; POCC ensembles are uniform per prototype.

Element references: data/mlip_refs.v1.json (relaxed polymorph minimum +
gas molecules), `python -m loopml.mlip refs --elements Hf Nb O ...`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from . import config

REFS_PATH = config.MLIP_REFS_PATH
MODEL_PATH = config.MACE_MODEL

MODEL_NAME = config.MACE_MODEL_NAME
# relaxed per element; min wins. Lattice guesses only — the cell relaxes.
POLYMORPH_LATTICES = {
    "fcc": (None, 3.6, 4.2),
    "bcc": (None, 3.1, 3.6),
    "hcp": (None, 3.0),
    "diamond": (None, 3.6, 6.0),
}
# Gas-phase references, as in DFT formation-energy protocols.
REFERENCE_MOLECULES = {"O": "O2"}

_CALC = None


def get_calculator():
    global _CALC
    if _CALC is None:
        import torch

        from mace.calculators import MACECalculator

        torch.set_num_threads(2)
        _CALC = MACECalculator(
            model_path=str(MODEL_PATH), device=_device(), default_dtype="float32"
        )
    return _CALC


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_TSMODEL = None

import time as _time

_T0 = _time.perf_counter()


def _log(msg: str) -> None:
    import os
    import sys
    import time

    print(f"[loopml {time.perf_counter() - _T0:.1f}s] {msg}", file=sys.stderr, flush=True)


def _instrument(model) -> None:
    """Wrap model.forward to log a heartbeat every 100 model calls."""
    import time

    if getattr(model, "_loopml_logged", False):
        return
    orig = model.forward
    ctr = {"n": 0, "t": time.perf_counter()}

    def forward(state, *a, **k):
        ctr["n"] += 1
        if ctr["n"] % 100 == 1:
            _log(f"forward #{ctr['n']} atoms={state.positions.shape[0]} "
                 f"since_chunk={time.perf_counter() - ctr['t']:.1f}s")
        return orig(state, *a, **k)

    model.forward = forward
    model._loopml_logged = True
    model._loopml_ctr = ctr


def get_batch_model():
    """TorchSim model wrapper over the same MACE checkpoint (float32)."""
    global _TSMODEL
    if _TSMODEL is None:
        import os

        import torch
        from torch_sim.models.mace import MaceModel
        from torch_sim.neighbors import vesin_nl_ts

        torch.set_num_threads(2)
        kwargs = {}
        if not torch.cuda.is_available() or os.environ.get("LOOPML_FORCE_VESIN_NL") == "1":
            kwargs["neighbor_list_fn"] = vesin_nl_ts
        _TSMODEL = MaceModel(
            model=str(MODEL_PATH),
            device=torch.device(_device()),
            dtype=torch.float32,
            **kwargs,
        )
    return _TSMODEL


def atoms_from_dict(struct: dict):
    """{'Lattice Matrix', 'Elements', 'Atomic Positions', 'Direct...'} -> ASE."""
    from ase import Atoms

    return Atoms(
        symbols=struct["Elements"],
        scaled_positions=struct["Atomic Positions"],
        cell=struct["Lattice Matrix"],
        pbc=True,
    )


def atoms_from_task(task: dict):
    """Task rows carry either a POSCAR path or an inline structure dict."""
    if task.get("poscar"):
        from ase.io import read

        return read(task["poscar"], format="vasp", index=":")[-1]
    return atoms_from_dict(task["structure"])


def relax_energy(atoms, fmax: float = 0.02, steps: int = 500) -> dict:
    """Relax cell+positions (BFGS on a Frechet cell filter); E/atom + diagnostics."""
    import warnings

    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    warnings.filterwarnings("ignore")
    atoms = atoms.copy()
    atoms.calc = get_calculator()
    natoms = len(atoms)
    static = atoms.get_potential_energy() / natoms
    try:
        opt = BFGS(FrechetCellFilter(atoms), logfile=None)
        converged = opt.run(fmax=fmax, steps=steps)
    except Exception:
        converged = False
    energy = atoms.get_potential_energy() / natoms
    return {
        "atoms": atoms,
        "energy_atom": energy,
        "converged": bool(converged),
        "max_force": float(abs(atoms.get_forces()).max()) if converged else None,
        "volume_per_atom": atoms.get_volume() / natoms,
        "static_energy_atom": static,
    }


def _relax_stage(atoms_list: list, fmax: float, steps: int,
                 memory_scaler: float | None = None, optimizer: str = "bfgs",
                 steps_between_swaps: int = 5) -> tuple[list, object]:
    """One batched optimization stage over equal-size systems; (aligned dicts, state).

    memory_scaler: per-batch admit budget in TorchSim's n_atoms_x_density metric
    (= n_atoms^2 / volume_nm3; ~512 per 8-atom POCC cell), as an explicit
    InFlightAutoBatcher cap on CUDA (skips torch-sim's OOM-probing estimation).
    """
    import torch
    import torch_sim as ts

    if any(not bool(atoms.pbc.all()) for atoms in atoms_list):
        raise ValueError("relax-batch supports fully periodic cells only")
    model = get_batch_model()
    _instrument(model)
    model._loopml_ctr["t"] = __import__("time").perf_counter()
    model._loopml_ctr["n"] = 0
    _log(f"stage start: {len(atoms_list)} systems x {len(atoms_list[0])} atoms "
         f"optimizer={optimizer} steps={steps} swap_every={steps_between_swaps} "
         f"scaler={memory_scaler}")
    conv_fn = ts.generate_force_convergence_fn(force_tol=fmax, include_cell_forces=True)
    static = None
    try:
        k = max(1, 1500 // max(1, len(atoms_list[0])))
        parts = []
        for s in range(0, len(atoms_list), k):
            sub = ts.initialize_state(atoms_list[s : s + k], model.device, model.dtype)
            parts.append(model(sub)["energy"].detach())
        static = torch.cat(parts)
    except Exception:
        torch.cuda.empty_cache()
        static = None
    autobatcher: object = torch.cuda.is_available()
    if memory_scaler and torch.cuda.is_available():
        from torch_sim.autobatching import InFlightAutoBatcher

        autobatcher = InFlightAutoBatcher(
            model=model,
            max_memory_scaler=memory_scaler,
            memory_scales_with=model.memory_scales_with,
            max_memory_padding=0.9,
        )
    _t0 = __import__("time").perf_counter()
    state = ts.optimize(
        system=atoms_list,
        model=model,
        optimizer=ts.Optimizer(optimizer),
        convergence_fn=conv_fn,
        max_steps=steps,
        steps_between_swaps=steps_between_swaps,
        autobatcher=autobatcher,
        init_kwargs={"cell_filter": ts.CellFilter.frechet},
    )
    _log(f"stage done: nsteps={getattr(state, 'nsteps', '?')} "
             f"wall={__import__('time').perf_counter() - _t0:.1f}s")
    converged = conv_fn(state, None)
    max_force = torch.zeros(
        state.n_systems, device=state.device, dtype=state.dtype
    ).scatter_reduce(
        0, state.system_idx, state.forces.abs().amax(dim=1), reduce="amax"
    )
    out = []
    for i, final in enumerate(state.to_atoms()):
        natoms = len(final)
        out.append(
            {
                "atoms": final,
                "energy_atom": float(state.energy[i]) / natoms,
                "converged": bool(converged[i]),
                "max_force": float(max_force[i]) if bool(converged[i]) else None,
                "volume_per_atom": final.get_volume() / natoms,
                "static_energy_atom": (float(static[i]) / natoms) if static is not None else None,
            }
        )
    return out, state


def _relax_chunk(atoms_list: list, fmax: float, steps: int,
                 memory_scaler: float | None = None, optimizer: str = "bfgs",
                 steps_between_swaps: int = 5, pre_relax: int = 0) -> list[dict]:
    """Batched relax; pre_relax>0 runs that many FIRE steps first, then polish.

    FIRE (cheap per-step, fully batched) burns down the easy near-equilibrium
    cells while the batch is wide; survivors get a fresh equal-size batch under
    `optimizer` (BFGS by default, matches the ASE pipeline closely).
    """
    if not pre_relax:
        out, _ = _relax_stage(atoms_list, fmax, steps, memory_scaler, optimizer,
                              steps_between_swaps)
        return out
    dicts, _ = _relax_stage(atoms_list, fmax, pre_relax, memory_scaler, "fire",
                            steps_between_swaps)
    need = [i for i, d in enumerate(dicts) if not d["converged"]]
    if need:
        _log(f"pre-relax fire: {len(need)}/{len(dicts)} survivors -> {optimizer}")
        d2, _ = _relax_stage([dicts[i]["atoms"] for i in need], fmax, steps,
                             memory_scaler, optimizer, steps_between_swaps)
        for i, d in zip(need, d2):
            d["static_energy_atom"] = dicts[i]["static_energy_atom"]
            dicts[i] = d
    return dicts


def relax_batch(
    atoms_list: list, fmax: float = 0.02, steps: int = 500, batch_size: int | None = None,
    memory_scaler: float | None = None, optimizer: str = "bfgs",
    steps_between_swaps: int = 5, pre_relax: int = 0,
) -> list[dict]:
    """Batched BFGS relaxation (TorchSim) of many structures per device step.

    Outputs match relax_energy (minus the relaxed atoms side effects); systems
    are bucketed to equal natoms so the padded-batch bug in torch-sim <=0.6.2
    never bites. Within one call, converged structures are swapped out of the
    batch as they converge (every 5 steps), so a straggler only ever costs its
    own evaluations; on CUDA the TorchSim autobatcher keeps the whole bucket
    resident under a memory budget instead of chunking (memory_scaler sets an
    explicit n_atoms_x_density budget and skips torch-sim's slow OOM probe).
    A chunk that crashes is retried system-by-system.
    """
    import warnings

    warnings.filterwarnings("ignore")
    results: dict[int, dict] = {}
    groups: dict[int, list[int]] = {}
    for i, atoms in enumerate(atoms_list):
        groups.setdefault(len(atoms), []).append(i)
    for natoms, idxs in sorted(groups.items()):
        size = batch_size or len(idxs)
        for start in range(0, len(idxs), size):
            chunk_idx = idxs[start : start + size]
            try:
                chunk_out = _relax_chunk(
                    [atoms_list[i] for i in chunk_idx], fmax=fmax, steps=steps,
                    memory_scaler=memory_scaler, optimizer=optimizer,
                    steps_between_swaps=steps_between_swaps, pre_relax=pre_relax,
                )
            except Exception:
                chunk_out = None
            if chunk_out is None:
                chunk_out = []
                for i in chunk_idx:
                    try:
                        chunk_out.append(
                            _relax_chunk(
                                [atoms_list[i]], fmax=fmax, steps=steps,
                                memory_scaler=memory_scaler, optimizer=optimizer,
                                steps_between_swaps=steps_between_swaps,
                            )[0]
                        )
                    except Exception:
                        chunk_out.append(
                            {"atoms": None, "energy_atom": None, "converged": False,
                             "max_force": None, "volume_per_atom": None,
                             "static_energy_atom": None}
                        )
            results.update(dict(zip(chunk_idx, chunk_out)))
    return [results[i] for i in range(len(atoms_list))]


def task_key(poscar: Path) -> str:
    """Stable identity of a relax task: '<ensemble dir>/<POSCAR file>'.

    Derived from the corpus layout (step 2 names structures uniquely), so
    it survives transfer between machines — unlike a content hash, which
    drifts with every re-serialization of a POSCAR.  A result row is
    joined back to its structure by this key alone."""
    p = Path(poscar)
    return f"{p.parent.name}/{p.name}"


def _save_refs(refs: dict) -> None:
    REFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFS_PATH.write_text(json.dumps(refs, indent=2, sort_keys=True) + "\n")


def element_reference(symbol: str, refs: dict, fmax: float, steps: int) -> float:
    """MLIP ground-state E/atom: relaxed polymorph minimum (or gas molecule)."""
    if symbol in refs:
        return refs[symbol]["energy_atom"]
    if symbol in REFERENCE_MOLECULES:
        from ase.build import molecule

        atoms = molecule(REFERENCE_MOLECULES[symbol])
        atoms.center(vacuum=6.0)
        atoms.calc = get_calculator()
        from ase.optimize import BFGS

        converged = BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        if not converged:
            raise RuntimeError(f"molecule reference {symbol} did not converge")
        best = {
            "poly": f"gas:{REFERENCE_MOLECULES[symbol]}",
            "energy_atom": atoms.get_potential_energy() / len(atoms),
            "converged": True,
        }
    else:
        from ase.build import bulk

        best = None
        for crystal, a_guesses in POLYMORPH_LATTICES.items():
            for a in a_guesses:
                try:
                    atoms = bulk(symbol, crystalstructure=crystal, a=a)
                    result = relax_energy(atoms, fmax=fmax, steps=steps)
                except Exception:
                    continue
                if not result["converged"]:
                    continue
                candidate = {"poly": crystal, "energy_atom": result["energy_atom"], "converged": True}
                if best is None or candidate["energy_atom"] < best["energy_atom"]:
                    best = candidate
                break
        if best is None:
            raise RuntimeError(f"no converged polymorph for element {symbol}")
    refs[symbol] = best
    _save_refs(refs)
    return best["energy_atom"]


def weighted_stdev(values: list[float], weights: list[int]) -> float:
    total = sum(weights)
    if len(values) < 2 or total <= 1:
        raise ValueError("need at least two weighted values")
    mean = sum(v * w for v, w in zip(values, weights)) / total
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / (total - 1)
    return math.sqrt(var)


def hmix_from_energies(energies: list[float], dgs: list[int]) -> float:
    """AFLOW getHmix: dg-weighted mean relaxed energy per atom."""
    total = sum(dgs)
    if total <= 0:
        raise ValueError("degeneracies sum to zero")
    return sum(e * g for e, g in zip(energies, dgs)) / total


K_BOLTZMANN_EV_K = 8.617333262e-5


def boltzmann_mean_energy(energies: list[float], dgs: list[int],
                          temperature: float) -> float:
    """dg-weighted Boltzmann average of ensemble energies at T (eV/atom).

    p_i proportional to dg_i exp(-E_i/kT); energies shifted by min(E) so
    the exponentials stay finite (the mean is shift-invariant)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    emin = min(energies)
    weights = [g * math.exp(-(e - emin) / (K_BOLTZMANN_EV_K * temperature))
               for e, g in zip(energies, dgs)]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Boltzmann weights underflowed")
    return sum(w * e for w, e in zip(weights, energies)) / total


def efa_pick_lowest_energy(ensembles: list[dict], temperature: float = 298.15) -> dict:
    """Resolve one EFA across prototypes of the SAME composition.

    Policy (decided 2026-10-28): take the EFA of the ensemble with the
    lowest BOLTZMANN-WT average energy, p_i proportional to
    dg_i exp(-E_i/kT at `temperature`); energies are per-atom under one
    potential, so prototypes are directly comparable.
    Input: [{name, energies, dgs}]; returns the winner plus every
    candidate's {name, mean_energy, efa} for auditability.
    """
    scored = [{"name": e["name"],
               "mean_energy": boltzmann_mean_energy(e["energies"], e["dgs"], temperature),
               "efa": efa_from_energies(e["energies"], e["dgs"])} for e in ensembles]
    if not scored:
        raise ValueError("no ensembles")
    best = min(scored, key=lambda s: s["mean_energy"])
    return {"selected": best["name"], "efa": best["efa"], "temperature": temperature,
            "mean_energy": best["mean_energy"], "candidates": scored}


def efa_from_energies(energies: list[float], dgs: list[int]) -> float:
    """AFLOW getEFA: 1/sigma of per-atom energies, dg-weighted (aflow_pocc.cpp:911)."""
    return 1.0 / weighted_stdev(energies, dgs)


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out

    return pearson(ranks(x), ranks(y))


def composition_key(elements: dict[str, float]) -> str:
    total = sum(elements.values())
    return "|".join(
        f"{s}={round(v / total, 6)}" for s, v in sorted(elements.items())
    )


def read_results(work_dir: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if work_dir.is_dir():
        for path in sorted(work_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("tid") and row.get("energy_atom") is not None:
                        done[row["tid"]] = row
    return done


def relax_shard(in_path: Path, out_path: Path) -> None:
    """Worker entry: sequential, no orchestration, append + flush per line."""
    done = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("tid"):
                    done.add(row["tid"])
    tasks = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    with out_path.open("a") as fh:
        for task in tasks:
            if task["tid"] in done:
                continue
            try:
                atoms = atoms_from_task(task)
                result = relax_energy(atoms, fmax=task["fmax"], steps=task["steps"])
            except Exception as exc:
                result = {"energy_atom": None, "converged": False, "error": str(exc)[:160]}
            result.pop("atoms", None)
            fh.write(json.dumps({"tid": task["tid"], "model": MODEL_NAME, **result}) + "\n")
            fh.flush()


def _all_done_tids(work_dir: Path) -> set:
    done: set = set()
    for f in work_dir.glob("*.jsonl"):
        if ".in." in f.name:
            continue
        try:
            for line in f.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("tid"):
                        done.add(row["tid"])
        except OSError:
            pass
    return done


def _write_rows(fh, chunk, relaxed) -> None:
    for task, result in zip(chunk, relaxed):
        result = dict(result)
        result.pop("atoms", None)
        fh.write(json.dumps({"tid": task["tid"], "model": MODEL_NAME, **result}) + "\n")
    fh.flush()


def _process_chunk(chunk_tasks: list[dict], fh, done: set, *, batch_size, memory_scaler,
                   optimizer, steps_between_swaps, pre_relax, chunk_atoms) -> None:
    """Relax one pre-built chunk (skipping tids already in done) and flush."""
    groups: dict[tuple[int, float, int], list] = {}
    for task in chunk_tasks:
        if task["tid"] in done:
            continue
        try:
            atoms = atoms_from_task(task)
            key = (len(atoms), task["fmax"], task["steps"])
        except Exception:
            atoms, key = None, (-1, task["fmax"], task["steps"])
        groups.setdefault(key, []).append((task, atoms))
    for (natoms, fmax, steps), group in sorted(groups.items()):
        ok = [(task, atoms) for task, atoms in group if atoms is not None]
        for task, _ in group:
            if _ is None:
                _write_rows(fh, [task], [{"energy_atom": None, "converged": False,
                                          "error": "unreadable structure"}])
        sub: list[list] = []
        cur: list = []
        for ta in ok:
            if cur and ((batch_size is not None and len(cur) >= batch_size)
                        or natoms * (len(cur) + 1) > chunk_atoms):
                sub.append(cur)
                cur = []
            cur.append(ta)
        if cur:
            sub.append(cur)
        for c in sub:
            try:
                relaxed = relax_batch(
                    [atoms for _, atoms in c], fmax=fmax, steps=steps,
                    memory_scaler=memory_scaler, optimizer=optimizer,
                    steps_between_swaps=steps_between_swaps, pre_relax=pre_relax)
            except Exception as exc:
                relaxed = [{"energy_atom": None, "converged": False,
                            "error": str(exc)[:160]} for _ in c]
            _write_rows(fh, [t for t, _ in c], relaxed)
            _log(f"chunk flushed: {len(c)} systems x {natoms} atoms")


def _claimable(chunks_dir: Path) -> list[Path] | None:
    """Return one unclaimed chunk path (claiming it atomically) or None when done.

    Claims are O_EXCL marker files holding the claiming pid; markers whose pid
    is dead on this host are stale (crashed worker) and get reclaimed."""
    import os
    names = sorted(f.name for f in chunks_dir.glob("chunk-*.jsonl"))
    for name in names:
        claim = chunks_dir / (name + ".claim")
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(claim.read_text().strip() or "0")
                os.kill(pid, 0)
                continue
            except (ValueError, ProcessLookupError):
                pass
            except PermissionError:
                continue
            try:
                claim.unlink()
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (OSError, FileExistsError):
                continue
        except OSError:
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return chunks_dir / name
    return None


def relax_batch_worker(in_path: Path, out_path: Path, batch_size: int | None = None,
                       memory_scaler: float | None = None, optimizer: str = "bfgs",
                       steps_between_swaps: int = 5, pre_relax: int = 0,
                       chunk_atoms: int = 4096) -> None:
    """Worker entry for the batched relaxer; same resumable JSONL contract.

    in_path is either a task JSONL (static mode: the worker re-chunks its
    shard and processes it in order) or a chunks directory of chunk-*.jsonl
    files (work-stealing mode: the worker repeatedly claims the next
    unclaimed chunk via O_EXCL markers, so GPUs never idle behind a straggler
    and a crashed worker's unflushed chunk is picked up elsewhere).
    Results flush after every chunk; restarts skip finished tids."""
    kw = dict(batch_size=batch_size, memory_scaler=memory_scaler, optimizer=optimizer,
              steps_between_swaps=steps_between_swaps, pre_relax=pre_relax,
              chunk_atoms=chunk_atoms)
    if in_path.is_dir():
        with out_path.open("a") as fh:
            while True:
                chunk = _claimable(in_path)
                if chunk is None:
                    break
                tasks = [json.loads(line)
                         for line in chunk.read_text().splitlines() if line.strip()]
                _process_chunk(tasks, fh, _all_done_tids(out_path.parent), **kw)
        return
    done = _all_done_tids(out_path.parent)
    tasks = [
        task
        for task in (json.loads(line) for line in in_path.read_text().splitlines() if line.strip())
        if task["tid"] not in done
    ]
    by_size: dict[int, list[dict]] = {}
    for task in tasks:
        try:
            n = len(atoms_from_task(task))
        except Exception:
            n = -1
        by_size.setdefault(n, []).append(task)
    chunks: list[list[dict]] = []
    for natoms, group in sorted(by_size.items()):
        if natoms <= 0:
            chunks.append(group)
            continue
        for start in range(0, len(group), max(1, min(chunk_atoms // natoms,
                                                     batch_size or chunk_atoms // natoms))):
            chunks.append(group[start : start + max(1, min(chunk_atoms // natoms,
                                                           batch_size or chunk_atoms // natoms))])
    with out_path.open("a") as fh:
        for chunk in chunks:
            _process_chunk(chunk, fh, done, **kw)


def run_workers(tasks: list[dict], work_dir: Path, workers: int) -> None:
    """Spawn one worker process per shard; resumable via existing outputs."""
    todo = [t for t in tasks if t["tid"] not in read_results(work_dir)]
    print(f"{len(tasks)} structures, {len(todo)} to relax on {workers} workers")
    if not todo:
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(workers):
        shard = todo[i::workers]
        if not shard:
            continue
        in_path = work_dir / f"shard{i}.in.jsonl"
        out_path = work_dir / f"shard{i}.jsonl"
        in_path.write_text("".join(json.dumps(t) + "\n" for t in shard))
        log = out_path.with_suffix(".log")
        handle = log.open("w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "loopml.mlip", "relax-shard", str(in_path), str(out_path)],
            stdout=handle, stderr=handle, stdin=subprocess.DEVNULL,
        )
        procs.append((proc, out_path, handle, in_path))
    t0 = time.time()
    while any(proc.poll() is None for proc, *_ in procs):
        time.sleep(15)
        done_now = sum(len(out.read_text().splitlines()) if out.is_file() else 0
                       for _, out, _, _ in procs)
        rate = done_now / max(time.time() - t0, 1e-9)
        print(f"  relaxed {done_now}/{len(todo)} ({rate:.1f}/s, {time.time() - t0:.0f}s)", flush=True)
    for proc, out, handle, in_path in procs:
        handle.close()
        in_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            print(f"  worker for {out.name} exited {proc.returncode} (see {out.with_suffix('.log')})")


def refs_command(elements: list[str], fmax: float, steps: int) -> int:
    refs = json.loads(REFS_PATH.read_text()) if REFS_PATH.is_file() else {}
    for el in elements:
        if el in refs:
            print(f"{el}: {refs[el]['energy_atom']:.4f} eV/atom ({refs[el]['poly']}) cached")
            continue
        value = element_reference(el, refs, fmax, steps)
        print(f"{el}: {value:.4f} eV/atom ({refs[el]['poly']})")
    return 0


def thermalize_batch(atoms_list: list, temperature: float = 300.0, steps: int = 2000,
                     timestep: float = 0.002, memory_scaler: float | None = None) -> list:
    """Batched NVT-Langevin MD; returns final thermal snapshots (bucketed by natoms).

    Feeds the 300K half of the lattice-distortion descriptors (distortion.py):
    thermal displacement sampled at the 0K-relaxed cell, matching the descriptor
    definition (sampled structure vs ideal input), not AFLOW's VASP run protocol.
    """
    import warnings

    import torch
    import torch_sim as ts

    warnings.filterwarnings("ignore")
    model = get_batch_model()
    out: dict[int, object] = {}
    groups: dict[int, list[int]] = {}
    for i, atoms in enumerate(atoms_list):
        groups.setdefault(len(atoms), []).append(i)
    for natoms, idxs in groups.items():
        chunk = [atoms_list[i] for i in idxs]
        autobatcher: object = False
        if memory_scaler and torch.cuda.is_available():
            from torch_sim.autobatching import BinningAutoBatcher

            autobatcher = BinningAutoBatcher(
                model=model, memory_scales_with=model.memory_scales_with,
                max_memory_scaler=memory_scaler,
            )
        state = ts.integrate(
            system=chunk, model=model, integrator=ts.Integrator.nvt_langevin,
            n_steps=steps, temperature=temperature, timestep=timestep,
            autobatcher=autobatcher,
        )
        for j, atoms in enumerate(state.to_atoms()):
            out[idxs[j]] = atoms
    return [out[i] for i in range(len(atoms_list))]


def distortion_command(ensemble_dirs: list[str], out_path: Path, fmax: float = 0.02,
                       steps: int = 1500, temperature: float = 300.0, md_steps: int = 2000,
                       memory_scaler: float | None = None) -> int:
    """Per composition ensemble: 0K relax + 300K sample -> lattice distortion stats."""
    from . import distortion as ds

    ideal, weights, names = [], [], []
    for d in ensemble_dirs:
        wmap = ds.ensemble_weights(d)
        for poscar in sorted(Path(d).glob("POSCAR_*")):
            try:
                atoms = atoms_from_task({"poscar": str(poscar)})
            except Exception:
                continue
            ideal.append(atoms)
            weights.append(wmap.get(poscar.name, 1.0))
            names.append(str(poscar))
    if not ideal:
        print("no readable structures", file=sys.stderr)
        return 2
    print(f"relaxing {len(ideal)} structures ...", flush=True)
    relaxed = relax_batch(ideal, fmax=fmax, steps=steps, memory_scaler=memory_scaler)
    rel_atoms = [r["atoms"] if r["atoms"] is not None else a.copy()
                 for r, a in zip(relaxed, ideal)]
    print(f"thermalizing at {temperature:g} K ({md_steps} steps) ...", flush=True)
    hot_atoms = thermalize_batch(rel_atoms, temperature=temperature, steps=md_steps,
                                 memory_scaler=memory_scaler)
    rows = []
    for name, w, a0, r, hot in zip(names, weights, ideal, relaxed, hot_atoms):
        row = {"poscar": name, "weight": w}
        if r["atoms"] is None:
            row["error"] = "relax failed"
        else:
            row.update(
                energy_atom=r["energy_atom"], converged=r["converged"],
                relax_distance_0K=ds.relax_distance(a0, r["atoms"]),
                misfit_0K=ds.misfit(a0, r["atoms"]),
                relax_distance_300K=ds.relax_distance(a0, hot),
                misfit_300K=ds.misfit(a0, hot),
            )
        rows.append(row)
    ok = [r for r in rows if "misfit_0K" in r]
    w = [r["weight"] for r in ok]
    summary = {key: ds.ensemble_stats([r[key] for r in ok], w)
               for key in ("relax_distance_0K", "misfit_0K",
                           "relax_distance_300K", "misfit_300K")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"report -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["refs", "relax-shard", "relax-batch",
                                            "distortion"])
    parser.add_argument("--elements", nargs="*", default=[])
    parser.add_argument("--fmax", type=float, default=0.02, help="descriptor-grade relax; audits use 0.001")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--pre-relax", type=int, default=0,
                        help="relax-batch: FIRE steps before the polish optimizer "
                             "(keeps the batch wide through easy structures)")
    parser.add_argument("--optimizer", choices=["bfgs", "lbfgs", "fire"], default="bfgs",
                        help="relax-batch: bfgs matches ASE closest; lbfgs usually needs "
                             "fewer steps on rugged cells")
    parser.add_argument("--swap-every", type=int, default=5,
                        help="relax-batch: steps between convergence checks/batch swaps")
    parser.add_argument("--memory-scaler", type=float, default=0,
                        help="relax-batch: per-batch admit budget, n_atoms^2/volume(nm^3) "
                             "(8-atom POCC cell ~512; 250000 ~ 490 cells). 0 = estimate")
    parser.add_argument("--chunk-atoms", type=int, default=4096,
                       help="relax-batch: flush + resume granularity, max atoms "
                            "per relaxed chunk (default 4096)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="relax-batch: cap structures per call (0 = whole size bucket; "
                             "on CUDA the TorchSim autobatcher manages memory instead)")
    parser.add_argument("--ensemble-dir", nargs="*", default=[],
                        help="distortion: protogen ensemble dirs (POSCAR_* + manifest.json)")
    parser.add_argument("--temperature", type=float, default=300.0,
                        help="distortion: sampling temperature (K)")
    parser.add_argument("--md-steps", type=int, default=2000,
                        help="distortion: NVT steps per snapshot (timestep 2 fs)")
    parser.add_argument("shard_paths", nargs="*",
                        help="relax-shard/relax-batch: IN OUT; distortion: OUT.json")
    args = parser.parse_args(argv)
    if args.command == "distortion":
        if len(args.shard_paths) != 1 or not args.ensemble_dir:
            print("distortion needs OUT.json and --ensemble-dir ...", file=sys.stderr)
            return 2
        return distortion_command(args.ensemble_dir, Path(args.shard_paths[0]),
                                  fmax=args.fmax, steps=args.steps,
                                  temperature=args.temperature, md_steps=args.md_steps,
                                  memory_scaler=args.memory_scaler or None)
    if args.command in ("relax-shard", "relax-batch"):
        if len(args.shard_paths) != 2:
            print(f"{args.command} needs IN OUT", file=sys.stderr)
            return 2
        in_path, out_path = (Path(p) for p in args.shard_paths)
        if args.command == "relax-shard":
            relax_shard(in_path, out_path)
        else:
            relax_batch_worker(in_path, out_path, batch_size=args.batch_size or None,
                               memory_scaler=args.memory_scaler or None,
                               optimizer=args.optimizer,
                               steps_between_swaps=args.swap_every,
                               pre_relax=args.pre_relax,
                               chunk_atoms=args.chunk_atoms)
        return 0
    return refs_command(args.elements, args.fmax, args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
