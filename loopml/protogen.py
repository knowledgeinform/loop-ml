"""POCC prototype generation via AFLOW, for downstream MLIP relaxation.

Given a composition (e.g. equimolar Hf/Nb/Ta/Ti/Zr + O) we build a
partially occupied parent cell (PARTCAR) from an AFLOW prototype template,
then let AFLOW's POCC machinery enumerate or randomly sample symmetry-distinct
fully ordered derivative supercells ("subphases" in ChemScreen's EFA/DEED
sense). Each ordered structure is a POSCAR ready for MLIP relaxation; the
manifest keeps the POCC metadata (HnCm id, HNF matrix, DG degeneracy) needed
to weight subphase energies later.

Deterministic: AFLOW samples with a fixed seed, so the same composition +
flags reproduce byte-identical structures. ``--sample-hash`` pins one exact
HnCm configuration if needed.

Usage:
    python -m loopml.protogen --composition Hf0.2Nb0.2Ta0.2Ti0.2Zr0.2O \
        [--prototype AB_cF8_225_a_b] [--sample-number 10] \
        [--lattice-a 4.2] [--outdir data/pocc/...]

Requires the AFLOW binary (built from ../AFLOW-src_dev/src via
``make -j aflow``; override location with $LOOPML_AFLOW_BIN).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

def canonical_key(elements: dict[str, float]) -> str:
    """Order-independent composition key: sorted El=share, shares normalized."""
    total = sum(elements.values())
    parts = sorted(f"{s}={round(v / total, 6)}" for s, v in elements.items())
    return "|".join(parts)
from .config import AFLOW_BIN, POCC_ROOT, PROTO_TEMPLATE_CACHE

TEMPLATE_CACHE_PATH = PROTO_TEMPLATE_CACHE

# Single-cation-site AB prototypes generatable offline via the ANRL library.

DEFAULT_LATTICE_A = 4.2  # Angstrom; MLIP relaxation fixes the volume anyway
MAX_DENOMINATOR = 1000  # occupancies must be rationals with small denominators
ANRL_TITLE_RE = re.compile(r"^\S+/(\S+?)\.")
POCC_TITLE_RE = re.compile(
    r"^(?P<formula>\S+)\s+H(?P<hnf_id>\d+)C(?P<config_id>\d+)\s+"
    r"HNF\(n=(?P<det>\d+),(?P<hnf_index>\d+)/(?P<hnf_count>\d+)\)="
    r"\[(?P<hnf_matrix>[^\]]*)\]\s+site_config\((?P<config_index>\d+)/"
    r"(?P<config_count>\d+)\)=\[[^\]]*\]\s+DG=(?P<dg>\d+)"
)
START_MARKER = "[VASP_POSCAR_MODE_EXPLICIT]START"
STOP_MARKER = "[VASP_POSCAR_MODE_EXPLICIT]STOP"


class ProtogenError(RuntimeError):
    pass


def aflow_binary() -> Path:
    candidate = os.environ.get("LOOPML_AFLOW_BIN")
    if candidate:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise ProtogenError(f"LOOPML_AFLOW_BIN not executable: {candidate}")
    which = shutil.which("aflow")
    if which:
        return Path(which)
    if Path(AFLOW_BIN).is_file():
        return Path(AFLOW_BIN)
    raise ProtogenError(
        "aflow binary not found; build ../AFLOW-src_dev/src (make -j aflow) "
        "or set $LOOPML_AFLOW_BIN"
    )


def aflow_version(binary: Path) -> str:
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    for line in out.stdout.splitlines():
        if line.startswith("AFLOW VERSION"):
            return line.split()[2]
    return "unknown"


def parse_composition(text: str) -> dict[str, float]:
    """Parse 'Hf0.2Nb0.2Ta0.2Ti0.2Zr0.2O' or 'La1Sr1Mn3O8' into fractions.

    Written numbers are counts (integers) or relative fractions; the result
    is normalized to sum 1 (bare element = one unit: '...Zr0.2O' gives O 0.5
    for 1:1 oxides).
    """
    tokens = re.findall(r"([A-Z][a-z]?)([0-9.]*)", text.replace(" ", ""))
    if not text or any(not sym for sym, _ in tokens):
        raise ProtogenError(f"unparsable composition: {text!r}")
    counts = {}
    for sym, num in tokens:
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    total = sum(counts.values())
    if total <= 0:
        raise ProtogenError(f"empty composition: {text!r}")
    return {sym: cnt / total for sym, cnt in counts.items()}


def _occupancy(fraction: float) -> Fraction:
    occ = Fraction(fraction).limit_denominator(MAX_DENOMINATOR)
    if float(occ) - fraction > 1e-9 or occ.denominator > MAX_DENOMINATOR:
        raise ProtogenError(f"occupancy {fraction} is not a small-denominator rational")
    return occ


def _run_aflow(args: list[str], stdin_text: str | None = None, timeout: int = 3600) -> str:
    binary = str(aflow_binary())
    proc = subprocess.run(
        [binary] + args, input=stdin_text, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        tail = re.sub(r"\x1b\[[0-9;]*m", "", proc.stderr + proc.stdout)[-800:]
        raise ProtogenError(f"aflow {' '.join(args)} failed rc={proc.returncode}: {tail}")
    return proc.stdout


# Screening hypothesis registry: family -> (ANRL label, dummy species order).
# The dummy "O" marks which site blocks are anionic; cation blocks take the
# real elements. AB3C_cP5's middle block is the O3 site (BaNiO3-type).
HYPOTHESES = {
    "rocksalt": ("AB_cF8_225_a_b", ("Ti", "O")),
    "cscl": ("AB_cP2_221_a_b", ("Ti", "O")),
    "fluorite": ("AB2_cF12_225_a_c", ("Mg", "O")),
    # WIP: spinel needs element-splitting across sites; perovskite hits a
    # POCC permutation-count error on 4-cation parents. Kept registered so
    # the tournament design stays explicit; tournament code must handle
    # ProtogenError per hypothesis.
    "spinel": ("AB2C4_cF56_227_b_c_e-001", ("Mg", "Ti", "O")),
    "perovskite": ("AB3C_cP5_221_a_c_b", ("La", "O", "Ti")),
    # A/B cation sublattices (equal capacity); generate_prototypes
    # (split_sites=True) partitions cations across them, POCC samples the
    # resulting partially occupied parent.
    "pyrochlore": ("A2B2C7_cF88_227_c_d_af-001", ("Gd", "Ti", "O")),
    "bixbyite": ("AB3C6_cI80_206_b_d_e-001", ("Gd", "Y", "O")),
    "perovskite": ("AB3C_cP5_221_a_c_b", ("Ca", "Ti", "O")),
}


def prototype_template_multi(prototype: str, dummies: tuple[str, ...]) -> dict:
    """Fetch a prototype's geometry; group rows by dummy-species block.

    Returns lattice (a=1 units) and one block per dummy: multiplicity and
    coordinates; blocks whose dummy is 'O' are anion sites.
    """
    stdout = _run_aflow(
        ["--op=protos", f"--proto={prototype}",
         f"--species=[{','.join(dummies)}]"],
        timeout=120,
    )
    lines = [ln for ln in stdout.splitlines() if ln.strip() and not ln.startswith("00000")]
    if not ANRL_TITLE_RE.match(lines[0]):
        raise ProtogenError(f"unexpected protos output for {prototype}: {lines[0]!r}")
    lattice = [[float(v) for v in lines[2 + i].split()] for i in range(3)]
    counts = [int(v) for v in lines[5].split()]
    if len(counts) != len(dummies):
        raise ProtogenError(f"{prototype}: counts {counts} vs dummies {dummies}")
    blocks = []
    for offset, (dummy, count) in enumerate(zip(dummies, counts)):
        rows = lines[7 + sum(counts[:offset]) : 7 + sum(counts[: offset + 1])]
        coords = [tuple(float(v) for v in row.split()[:3]) for row in rows]
        blocks.append({"dummy": dummy, "mult": count, "coords": coords})
    return {"prototype": prototype, "lattice": lattice, "blocks": blocks}


def get_template(name: str) -> dict:
    """Hypothesis template, cached on disk after first AFLOW fetch."""
    if name not in HYPOTHESES:
        raise ProtogenError(f"unknown hypothesis {name!r}; have {sorted(HYPOTHESES)}")
    cache: dict = {}
    if TEMPLATE_CACHE_PATH.is_file():
        cache = json.loads(TEMPLATE_CACHE_PATH.read_text())
    if name in cache:
        return cache[name]
    prototype, dummies = HYPOTHESES[name]
    template = prototype_template_multi(prototype, dummies)
    cache[name] = template
    TEMPLATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return template


def _largest_remainder(fracs: dict[str, float], total: int) -> dict[str, int]:
    raw = {el: f * total for el, f in fracs.items()}
    counts = {el: int(v) for el, v in raw.items()}
    for el, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if sum(counts.values()) >= total:
            break
        counts[el] += 1
    return counts


def assign_sites(composition: dict[str, float], template: dict,
                 split: bool = False) -> dict | None:
    """Map each element to cation block(s).

    Default: one block per element, or None if no packing fits (element supply
    is x_e * (total cation multiplicity), packed largest-first into the
    smallest fitting block).

    split=True: elements are partitioned across cation blocks (POCC allows
    several species per block, but each element must live on exactly one
    block), with occupancies normalized so every block is fully occupied;
    for multi-sublattice prototypes like spinel and pyrochlore where
    one-element-per-block does not scale past a few species.
    """
    cat_blocks = [b for b in template["blocks"] if b["dummy"] != "O"]
    total = sum(b["mult"] for b in cat_blocks)
    cat_total = sum(f for el, f in composition.items() if el != "O")
    if not cat_blocks or cat_total <= 0 or composition.get("O", 0.0) <= 0:
        raise ProtogenError(f"composition needs cations plus O: {composition}")
    if split:
        # POCC requires one site group per element, so partition elements
        # across the cation sublattices (heaviest supply first onto the
        # emptiest block) and normalize occupancies so every block is fully
        # occupied: sublattice compositions are the nearest ideal-filling
        # limit of the input cation mix.
        shares = {el: f / cat_total for el, f in composition.items() if el != "O"}
        loads = [0.0] * len(cat_blocks)
        block_of: dict[str, int] = {}
        for el, _ in sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])):
            b = min(range(len(cat_blocks)), key=lambda i: (loads[i], i))
            block_of[el] = b
            loads[b] += shares[el]
        return {el: [(block_of[el], share / loads[block_of[el]])]
                for el, share in shares.items()}
    supply = {el: f / cat_total * total for el, f in composition.items() if el != "O"}
    remaining = [float(b["mult"]) for b in cat_blocks]
    assignment: dict[str, int] = {}
    for el, need in sorted(supply.items(), key=lambda kv: -kv[1]):
        fitting = [i for i, cap in enumerate(remaining) if cap >= need - 1e-9]
        if not fitting:
            return None
        site = min(fitting, key=lambda i: remaining[i])
        remaining[site] -= need
        assignment[el] = site
    return assignment


def build_partcar_multi(composition: dict[str, float], template: dict,
                        lattice_a: float, assignment: dict[str, int]) -> str:
    """PARTCAR parent for a multi-cation-site template (one block per element).

    Element e sitting on site of multiplicity m gets occupancy
    x_e * M_cation_sites / m, i.e. its exact share of the cation sublattice.
    """
    cat_blocks = [b for b in template["blocks"] if b["dummy"] != "O"]
    anion_block = next(b for b in template["blocks"] if b["dummy"] == "O")
    cat_total = sum(f for el, f in composition.items() if el != "O")
    total = sum(b["mult"] for b in cat_blocks)
    formula = "".join(f"{el}{composition[el] / cat_total * total:g}"
                      for el in sorted(assignment)) + f"O{anion_block['mult']}"
    lines = [formula, f"{lattice_a:.6f}"]
    for row in template["lattice"]:
        lines.append("".join(f"{v * lattice_a:20.14f}" for v in row))
    placements: dict[str, list[tuple[int, Fraction]]] = {}
    for el, target in assignment.items():
        sites = [(target, None)] if isinstance(target, int) else list(target)
        for site, occ in sites:
            share = occ if occ is not None else composition[el] / cat_total * total / cat_blocks[site]["mult"]
            occ_frac = Fraction(share).limit_denominator(MAX_DENOMINATOR)
            if float(occ_frac) > 1 + 1e-9 or occ_frac <= 0:
                raise ProtogenError(f"occupancy {occ_frac} for {el} invalid on site {site}")
            placements.setdefault(el, []).append((site, occ_frac))
    # PARTCAR requires globally alphabetized species blocks
    entries: list[tuple[str, list]] = []
    for i, block in enumerate(cat_blocks):
        for el in sorted(placements):
            for site, occ_frac in placements[el]:
                if site != i:
                    continue
                entries.append((el, [f"{block['mult']}*{float(occ_frac):.6g}"]
                                + [[coord, el] for coord in block["coords"]]))
    entries.append(("O", [f"{anion_block['mult']}"]
                    + [[coord, "O"] for coord in anion_block["coords"]]))
    entries.sort(key=lambda kv: kv[0])
    counts_line: list[str] = []
    coord_rows: list[tuple[tuple[float, float, float], str]] = []
    for _, body in entries:
        counts_line.append(body[0])
        coord_rows.extend(((x, y, z), el) for [x, y, z], el in body[1:])
    lines.append(" ".join(counts_line))
    lines.append("Direct(2)")
    for (x, y, z), el in coord_rows:
        lines.append(f"{x:20.14f}{y:20.14f}{z:20.14f}  {el}")
    return "\n".join(lines) + "\n"


def run_pocc(partcar: str, sample_number: int | None, sample_hash: str | None = None, timeout: int | None = None) -> str:
    if timeout is None:
        # AFLOW occasionally hangs on unlucky PARTCARs; cap hard by default
        timeout = int(os.environ.get("LOOPML_POCC_TIMEOUT", "300"))
    args = ["--pocc_show_unique"]
    if sample_number is not None:
        if sample_number < 1:
            raise ProtogenError("sample number must be positive")
        args.append(f"--pocc_sample_number={sample_number}")
    if sample_hash:
        args.append(f"--pocc_sample_hash={sample_hash}")
    return _run_aflow(args, stdin_text=partcar, timeout=timeout)


def _clean_poscar(body: list[str]) -> tuple[str, dict[str, int]]:
    """Rewrite a POCC POSCAR block with species names and a plain mode line."""
    scale = float(body[0])
    lattice = [[scale * float(v) for v in body[1 + i].split()] for i in range(3)]
    counts = [int(v) for v in body[4].split()]
    coords = []
    for offset, count in enumerate(counts):
        start = 6 + sum(counts[:offset])
        for row in body[start : start + count]:
            parts = row.split()
            coords.append((tuple(float(v) for v in parts[:3]), parts[3]))
    by_element: dict[str, list[tuple[float, float, float]]] = {}
    for pos, el in coords:
        by_element.setdefault(el, []).append(pos)
    groups = [[(pos, el) for pos in poss] for el, poss in by_element.items()]
    names = [grp[0][1] for grp in groups]
    formula = "".join(f"{name}{len(grp)}" for name, grp in zip(names, groups))
    out = [formula, "1.0"]
    for row in lattice:
        out.append("".join(f"{v:20.14f}" for v in row))
    out.append(" ".join(names))
    out.append(" ".join(str(len(grp)) for grp in groups))
    out.append("Direct")
    for grp in groups:
        for (x, y, z), _ in grp:
            out.append(f"{x:20.14f}{y:20.14f}{z:20.14f}")
    return "\n".join(out) + "\n", {name: len(grp) for name, grp in zip(names, groups)}


def parse_pocc_output(stdout: str) -> list[dict]:
    structures = []
    block: list[str] | None = None
    title = None
    for line in stdout.splitlines():
        if line.startswith(START_MARKER):
            block, title = [], None
        elif line.startswith(STOP_MARKER):
            if block is None or title is None:
                raise ProtogenError("malformed POCC output: STOP without title")
            match = POCC_TITLE_RE.match(title)
            if not match:
                raise ProtogenError(f"unparsable POCC title: {title!r}")
            poscar, counts = _clean_poscar(block)
            meta = match.groupdict()
            structures.append(
                {
                    "id": f"H{meta['hnf_id']}C{meta['config_id']}",
                    "det": int(meta["det"]),
                    "hnf_index": int(meta["hnf_index"]),
                    "hnf_count": int(meta["hnf_count"]),
                    "hnf_matrix": [[int(v) for v in row.split()] for row in meta["hnf_matrix"].split(";")],
                    "site_config_index": int(meta["config_index"]),
                    "site_config_count": int(meta["config_count"]),
                    "degeneracy": int(meta["dg"]),
                    "poscar": poscar,
                    "counts": counts,
                }
            )
            block, title = None, None
        elif block is not None:
            if title is None:
                title = line
            else:
                block.append(line)
    return structures


def generate_prototypes(
    composition: dict[str, float],
    outdir: Path,
    prototype: str = "rocksalt",
    lattice_a: float = DEFAULT_LATTICE_A,
    sample_number: int | None = 10,
    sample_hash: str | None = None,
    force: bool = False,
    timeout: int | None = None,
    split_sites: bool = False,
) -> dict:
    comp_key = canonical_key(composition)
    if outdir.exists() and any(outdir.iterdir()) and not force:
        raise ProtogenError(f"outdir not empty (use --force): {outdir}")
    if prototype not in HYPOTHESES:
        raise ProtogenError(f"unknown hypothesis {prototype!r}; have {sorted(HYPOTHESES)}")
    template = get_template(prototype)
    assignment = assign_sites(composition, template, split=split_sites)
    if assignment is None:
        raise ProtogenError(f"no site packing fits {comp_key} in {prototype}")
    partcar = build_partcar_multi(composition, template, lattice_a=lattice_a,
                                  assignment=assignment)
    stdout = run_pocc(partcar, sample_number=sample_number, sample_hash=sample_hash, timeout=timeout)
    structures = parse_pocc_output(stdout)
    if not structures:
        raise ProtogenError(f"POCC produced no structures for {comp_key} ({prototype})")
    manifest = {
        "schema": "pocc-protogen-v2",
        "composition_key": comp_key,
        "composition": dict(sorted(composition.items())),
        "hypothesis": prototype,
        "prototype": template["prototype"],
        "site_assignment": {
            el: ([f"site{t}:{float(o):.6g}" for t, o in v] if isinstance(v, list) else f"site{v}")
            for el, v in sorted(assignment.items())
        },
        "lattice_a": lattice_a,
        "sample_number": sample_number,
        "sample_hash": sample_hash,
        "aflow_binary": str(aflow_binary()),
        "aflow_version": aflow_version(aflow_binary()),
        "partcar_sha256": hashlib.sha256(partcar.encode()).hexdigest(),
        "structures": [],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "parent.POSCAR").write_text(partcar)
    for st in structures:
        path = outdir / f"POSCAR_{st['id']}"
        path.write_text(st["poscar"])
        manifest["structures"].append(
            {
                **{k: v for k, v in st.items() if k != "poscar"},
                "file": path.name,
                "sha256": hashlib.sha256(st["poscar"].encode()).hexdigest(),
            }
        )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--composition", required=True, help="e.g. Hf0.2Nb0.2Ta0.2Ti0.2Zr0.2O")
    parser.add_argument("--prototype", default="rocksalt",
                        help=f"hypothesis name {sorted(HYPOTHESES)}")
    parser.add_argument("--lattice-a", type=float, default=DEFAULT_LATTICE_A)
    parser.add_argument("--sample-number", type=int, default=10, help="ordered configs per HNF superlattice (omit with --all)")
    parser.add_argument("--all", action="store_true", help="full POCC enumeration (only for small configuration spaces)")
    parser.add_argument("--sample-hash", default=None, help="pin one configuration, e.g. H3C12")
    parser.add_argument("--outdir", type=Path, default=None, help="default data/pocc/<prototype>/<composition>/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-sites", action="store_true",
                        help="put every cation on every cation sublattice at its mole fraction")
    parser.add_argument("--records-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    composition = parse_composition(args.composition)
    prototype = args.prototype
    outdir = args.outdir or POCC_ROOT / prototype / canonical_key(composition)
    try:
        manifest = generate_prototypes(
            composition,
            outdir,
            prototype=prototype,
            lattice_a=args.lattice_a,
            sample_number=None if args.all else args.sample_number,
            sample_hash=args.sample_hash,
            force=args.force,
            split_sites=args.split_sites,
        )
    except ProtogenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{outdir}: {len(manifest['structures'])} POCC configurations ({prototype}, aflow {manifest['aflow_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
