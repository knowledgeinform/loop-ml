#!/usr/bin/env python3
"""Step 3: relax every structure of a prototype folder with the MLIP.

Scans <ensembles>/*/manifest.json, builds a de-duplicated task list (one
job per structure, keyed by task id), then runs resumable batched-BFGS workers
(loopml.mlip relax-batch / TorchSim).  Results append to <work>/bshard*.jsonl
and are flushed after every ~chunk-atoms chunk, so Ctrl-C or a crash loses
at most one chunk; rerun the SAME command to resume (completed structures
are skipped automatically).  Worker count defaults to one per visible GPU
(honors $CUDA_VISIBLE_DEVICES); CPU-only hosts get sharded sequential-safe
batches with a memory guard.

Usage:
    uv run python 3-relax-prototypes.py [--ensembles data/pocc]
        [--work data/mlip_work] [--tasks data/mlip_tasks.jsonl]
        [--workers 0] [--dry-run] [--engine torchsim|sequential]
        [--chunk-atoms 4096] [--batch-size N] [--fresh-chunks]
        [--optimizer bfgs|lbfgs|fire] [--pre-relax N]

For a separate GPU box: tar loopml/ + models/ + the ensemble folder,
run the same command there, return the work folder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loopml.mlip import read_results, task_key

ROOT = Path(__file__).resolve().parent


def build_tasks(regen: Path, tasks_path: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    n_structs = 0
    for man in sorted(regen.glob("*/manifest.json")):
        manifest = json.loads(man.read_text())
        for st in manifest["structures"]:
            poscar = (man.parent / st["file"]).resolve()
            tid = task_key(poscar)
            n_structs += 1
            seen.setdefault(tid, {"tid": tid, "poscar": str(poscar), "fmax": 0.02,
                                  "steps": 500, "natoms": sum(st["counts"].values())})
    tasks = list(seen.values())
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"{n_structs} ensemble structures -> {len(tasks)} relax tasks")
    return tasks


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def _visible_cuda_devices() -> list[str]:
    """Physical GPU ids the server is allowed to use (honors $CUDA_VISIBLE_DEVICES)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [tok.strip() for tok in env.split(",") if tok.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def write_chunks(tasks: list[dict], chunks_dir: Path, chunk_atoms: int,
                 batch_size: int = 0) -> int:
    """Pre-slice tasks into <=chunk_atoms chunks (LPT order) for work-stealing."""
    fresh = not list(chunks_dir.glob("chunk-*.jsonl")) if chunks_dir.is_dir() else True
    if fresh:
        for old in list(chunks_dir.glob("chunk-*")) + list(chunks_dir.glob("*.claim")):
            old.unlink()
        chunks_dir.mkdir(parents=True, exist_ok=True)
        by_size: dict[int, list[dict]] = {}
        for t in tasks:
            by_size.setdefault(t.get("natoms", 0), []).append(t)
        specs: list[tuple[int, list[dict]]] = []
        for natoms, group in by_size.items():
            per = max(1, chunk_atoms // natoms) if natoms else 1
            if batch_size:
                per = min(per, batch_size)
            for i in range(0, len(group), per):
                c = group[i : i + per]
                specs.append((natoms * len(c), c))
        specs.sort(key=lambda nc: (-nc[0], nc[1][0]["tid"]))
        for i, (cost, c) in enumerate(specs):
            path = chunks_dir / f"chunk-{10**12 - cost:013d}-{i:06d}.jsonl"
            path.write_text("".join(json.dumps(t) + "\n" for t in c))
        _log(f"work-stealing ledger: {len(specs)} chunks in {chunks_dir}")
    else:
        n = len(list(chunks_dir.glob("chunk-*.jsonl")))
        _log(f"resuming work-stealing ledger ({n} chunk files, claims kept)")
    return 0


def run_batch_workers(tasks: list[dict], work_dir: Path, workers: int,
                      extra_args: list[str], engine: str = "torchsim",
                      chunks_dir: Path | None = None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    if chunks_dir is not None and engine == "torchsim":
        write_chunks(tasks, chunks_dir,
                     int(extra_args[extra_args.index("--chunk-atoms") + 1])
                     if "--chunk-atoms" in extra_args else 4096,
                     int(extra_args[extra_args.index("--batch-size") + 1])
                     if "--batch-size" in extra_args else 0)
    done = read_results(work_dir)
    todo = [t for t in tasks if t["tid"] not in done]
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    _log(f"env: python={sys.version.split()[0]} torch-cuda={cuda} workers={workers or 'auto'}")
    devices = _visible_cuda_devices() if cuda else []
    if not workers:
        workers = max(1, len(devices)) if cuda else min(4, os.cpu_count() or 2)
        _log(f"auto worker count: {workers} "
             + (f"one per visible GPU {devices}" if cuda else "(CPU)"))
    if not extra_args:
        if cuda:
            extra_args = ["--chunk-atoms", "4096"]
            _log("CUDA: flushing every ~4096-atom chunk (resumable after Ctrl-C)")
        else:
            try:
                with open("/proc/meminfo") as fh:
                    mem_gb = float(fh.readline().split()[1]) / 2**20
            except Exception:
                mem_gb = 8.0
            cap = max(16, int(mem_gb * 12 // max(1, workers)))
            extra_args = ["--batch-size", str(min(cap, 64))]
            _log(f"CPU-only: batch-size cap {extra_args[1]} per worker (memory guard)")
    _log(f"{len(tasks)} structures, {len(todo)} to relax, {len(done)} already done "
         f"on {workers} {engine} workers")
    gpu_log = work_dir / "gpu.log"
    gpu_proc = None
    try:
        gpu_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pum", "-d", "30"],
            stdout=gpu_log.open("a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        _log(f"nvidia-smi dmon sampler pid={gpu_proc.pid} -> {gpu_log}")
    except Exception as exc:
        _log(f"WARNING: no nvidia-smi sampler ({exc}); GPU stats unavailable")
    if not todo:
        _log("nothing to do")
        return
    procs = []
    for i in range(workers):
        shard = todo[i::workers]
        if not shard:
            continue
        out_path = work_dir / f"bshard{i}.jsonl"
        if chunks_dir is not None and engine == "torchsim":
            in_path = chunks_dir
        else:
            in_path = work_dir / f"bshard{i}.in.jsonl"
            in_path.write_text("".join(json.dumps(t) + "\n" for t in shard))
        log_path = out_path.with_suffix(".log")
        handle = log_path.open("a")
        cmd = ([sys.executable, "-m", "loopml.mlip", "relax-shard", str(in_path), str(out_path)]
               if engine == "sequential" else
               [sys.executable, "-m", "loopml.mlip", "relax-batch", str(in_path), str(out_path),
                *extra_args])
        env = os.environ.copy()
        dev = None
        if cuda and devices:
            dev = devices[i % len(devices)]
            env["CUDA_VISIBLE_DEVICES"] = dev
        proc = subprocess.Popen(cmd, stdout=handle, stderr=handle,
                                stdin=subprocess.DEVNULL, env=env)
        procs.append([proc, out_path, handle, in_path, len(shard), log_path])
        _log(f"worker {i}: pid={proc.pid} device={dev or 'cpu:all'} tasks={len(shard)} "
             f"in={in_path.name} out={out_path.name} log={log_path.name}")
    t0 = time.time()
    baseline = len(done)
    last_total, last_change = None, time.time()
    try:
        while any(p[0].poll() is None for p in procs):
            time.sleep(60)
            total = len(read_results(work_dir))
            new = max(0, total - baseline)
            alive = sum(1 for p in procs if p[0].poll() is None)
            rate = new / max(time.time() - t0, 1e-9)
            line = (f"progress: {new}/{len(todo)} relaxed this session ({rate:.2f}/s, "
                    f"{time.time() - t0:.0f}s elapsed, {alive}/{len(procs)} workers alive)")
            if total == last_total:
                line += f" [NO OUTPUT for {time.time() - last_change:.0f}s — check worker logs]"
            else:
                last_change, last_total = time.time(), total
            _log(line)
    except KeyboardInterrupt:
        _log("INTERRUPTED: terminating workers; rerun the SAME command to resume "
             "(completed structures are skipped automatically)")
        for p, *_ in procs:
            p.terminate()
        time.sleep(5)
        for p, *_ in procs:
            if p.poll() is None:
                p.kill()
    for proc, out, handle, in_path, n, log_path in procs:
        handle.close()
        _log(f"worker {out.name}: exit={proc.returncode} tasks_assigned={n} "
             f"output_lines={len(out.read_text().splitlines()) if out.is_file() else 0} log={log_path}")
        if proc.returncode == 0:
            in_path.unlink(missing_ok=True)
    if gpu_proc:
        gpu_proc.terminate()
    out_files = [p for p in work_dir.glob("bshard*.jsonl") if ".in." not in p.name]
    rows = [json.loads(l) for p in out_files for l in p.read_text().splitlines() if l.strip()]
    bad = [r for r in rows if r.get("energy_atom") is None or str(r.get("converged")) != "True"]
    final_done = len(read_results(work_dir))
    joined = " ".join(str(p) for p in out_files) or "(none)"
    _log(f"FINAL: {final_done}/{len(tasks)} relaxed; bad rows this session: {len(bad)} "
         f"(count: grep -hc '\"converged\": false' {joined}; "
         f"errors: grep -h '\"error\"' {joined} | head)")
    if final_done < len(tasks):
        _log(f"WARNING: {len(tasks) - final_done} structures still missing — workers "
             f"died (check exit codes above / worker logs for OOM); rerun the SAME command to resume")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ensembles", type=Path, default=ROOT / "data" / "pocc",
                    help="folder of <chem>__<proto>/ ensemble dirs (step 2 output)")
    ap.add_argument("--work", type=Path, default=ROOT / "data" / "mlip_work",
                    help="results folder (bshard*.jsonl + chunk ledger)")
    ap.add_argument("--tasks", type=Path, default=None,
                    help="task list jsonl (default: <work>/mlip_tasks.jsonl)")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto: one worker per visible GPU (honors "
                         "$CUDA_VISIBLE_DEVICES), or min(4, cores) on CPU")
    ap.add_argument("--dry-run", action="store_true", help="only build the task list")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--chunk-atoms", type=int, default=0,
                    help="relax-batch flush/resume granularity in atoms (0 = worker default)")
    ap.add_argument("--engine", choices=["torchsim", "sequential"], default="torchsim",
                    help="sequential = one structure at a time (no TorchSim "
                         "auto-batching; slower but immune to VRAM estimation bugs)")
    ap.add_argument("--memory-scaler", type=float, default=0)
    ap.add_argument("--static-shards", action="store_true",
                    help="legacy fixed split instead of work-stealing chunks")
    ap.add_argument("--fresh-chunks", action="store_true",
                    help="discard the chunk ledger and rebuild it (claims reset)")
    ap.add_argument("--optimizer", choices=["bfgs", "lbfgs", "fire"], default="bfgs")
    ap.add_argument("--pre-relax", type=int, default=0)
    args = ap.parse_args(argv)
    tasks = build_tasks(args.ensembles, args.tasks or (args.work / "mlip_tasks.jsonl"))
    if args.dry_run:
        return 0
    extra = []
    if args.chunk_atoms:
        extra += ["--chunk-atoms", str(args.chunk_atoms)]
    if args.batch_size:
        extra += ["--batch-size", str(args.batch_size)]
    if args.memory_scaler:
        extra += ["--memory-scaler", str(args.memory_scaler)]
    if args.optimizer != "bfgs":
        extra += ["--optimizer", args.optimizer]
    if args.pre_relax:
        extra += ["--pre-relax", str(args.pre_relax)]
    chunks_dir = None
    if not args.static_shards and args.engine == "torchsim":
        chunks_dir = args.work / "chunks"
        if args.fresh_chunks:
            subprocess.run(["rm", "-rf", str(chunks_dir)], check=True)
    run_batch_workers(tasks, args.work, workers=args.workers, extra_args=extra,
                      engine=args.engine, chunks_dir=chunks_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
