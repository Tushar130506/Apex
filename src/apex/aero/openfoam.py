"""OpenFOAM (openfoam.com / ESI) case generation and result import.

One CFD case per design-of-experiments row. A design changes only the car's
attitude, which is applied to the geometry before meshing:

- ``aoa_deg``: pitch, rotation about the y axis (positive = nose up), about
  ``origin``;
- ``yaw_deg``: rotation about the z axis, about ``origin``;
- ``ride_height_mm``: upward shift in z, applied after the rotation.

The geometry must face upstream: flow along +x, nose towards -x, z up (as in
the motorBike tutorial). The run sheet's ``velocity_ms`` is written into the
base case's free-stream velocity entry, so flow and coefficients agree.

Everything else comes from a base case you have already run once by hand (a
copy of the motorBike tutorial with your own geometry is a good start). Its
``Allrun`` must not copy the geometry into the case, because that would undo
the design's attitude. The
base case keeps its own mesh, solver and boundary settings; this module only
adds a force-coefficient function object and a run script around it, and
reads the results back into the table that ``apex.aero.cfd_results`` checks.

Sign convention: lift is measured along ``(0 0 -1)``, so ``cl`` is positive
for downforce, as the rest of ``apex.aero`` expects.

Written against OpenFOAM v1912 and the ``-rollPitchYaw``/``-origin`` options
of ``surfaceTransformPoints``; the openfoam.org line uses a different syntax.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from apex.aero.doe import DEFAULT_BOUNDS

# Attitude range for a whole car; ride_height_mm is the lift above the base
# geometry's own position. (apex.aero.doe.DEFAULT_BOUNDS is sized for a wing.)
CAR_ATTITUDE_BOUNDS = {"aoa_deg": (-1.0, 2.0), "ride_height_mm": (0.0, 40.0), "yaw_deg": (0.0, 4.0)}
FORCE_FUNCTION = "apexForceCoeffs"
DESIGN_FILE = "apex_design.json"
RUN_SCRIPT = "Allrun.apex"


@dataclass(frozen=True)
class CaseSettings:
    """Reference values and names shared by every case of one study."""

    geometry: str                    # file in the base case's constant/triSurface, e.g. car.obj
    patches: tuple[str, ...]         # wall patches that make up the car, e.g. ("carGroup",)
    reference_area_m2: float         # A used for Cd and Cl (must match the vehicle model)
    reference_length_m: float        # wheelbase, used for moments
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)   # rotation centre and CofR
    end_time: int | None = None      # override the base case's iteration count
    # Where the base case sets its free-stream velocity vector; rewritten to the
    # run sheet's velocity so the force coefficients and the flow agree.
    inlet_file: str = "0.orig/include/initialConditions"
    inlet_entry: str = "flowVelocity"


def _vector(values) -> str:
    return "(" + " ".join(f"{float(value):g}" for value in values) + ")"


def force_coeffs_dict(settings: CaseSettings, velocity_ms: float, rho: float) -> str:
    """The forceCoeffs function object for one study."""
    return f"""{FORCE_FUNCTION}
{{
    type            forceCoeffs;
    libs            (forces);
    writeControl    timeStep;
    writeInterval   1;
    log             yes;
    patches         ({" ".join(settings.patches)});
    rho             rhoInf;
    rhoInf          {rho:g};
    liftDir         (0 0 -1);
    dragDir         (1 0 0);
    pitchAxis       (0 1 0);
    CofR            {_vector(settings.origin)};
    magUInf         {velocity_ms:g};
    lRef            {settings.reference_length_m:g};
    Aref            {settings.reference_area_m2:g};
}}
"""


def _add_function(control_dict: str) -> str:
    """Include the force function in controlDict's functions block."""
    include = f'    #include "{FORCE_FUNCTION}"'
    if include.strip() in control_dict:
        return control_dict
    match = re.search(r"^functions\s*\{", control_dict, flags=re.MULTILINE)
    if match:
        return control_dict[: match.end()] + "\n" + include + control_dict[match.end():]
    return control_dict.rstrip() + "\n\nfunctions\n{\n" + include + "\n}\n"


def _set_end_time(control_dict: str, end_time: int) -> str:
    updated, count = re.subn(r"^(endTime\s+)[^;]+;", rf"\g<1>{end_time};", control_dict, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("controlDict needs exactly one endTime entry")
    return updated


def _skip_outputs(directory: str, names: list[str]) -> set[str]:
    """Leave out results of earlier runs of the base case."""
    skipped = set()
    for name in names:
        if name.startswith(("processor", "log.")) or name in {"postProcessing", "dynamicCode"}:
            skipped.add(name)
        elif re.fullmatch(r"[0-9.eE+-]+", name) and Path(directory, name).is_dir():
            try:
                if float(name) > 0:
                    skipped.add(name)
            except ValueError:
                pass
    return skipped


def _set_inlet(text: str, entry: str, velocity_ms: float) -> str:
    updated, count = re.subn(
        rf"^(\s*{re.escape(entry)}\s+)\([^)]*\);", rf"\g<1>({velocity_ms:g} 0 0);", text, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"expected one '{entry} (u v w);' entry for the inlet velocity")
    return updated


def run_script(row: pd.Series, settings: CaseSettings) -> str:
    """Shell script that positions the geometry and then runs the base Allrun."""
    surface = f"constant/triSurface/{settings.geometry}"
    original = f"constant/triSurface/apex_original_{settings.geometry}"
    rotation = _vector((0.0, row["aoa_deg"], row["yaw_deg"]))
    shift = _vector((0.0, 0.0, row["ride_height_mm"] / 1000.0))
    return f"""#!/bin/sh
# Generated by apex.aero.openfoam for {row["geometry_id"]}. Edit the generator, not this file.
cd "${{0%/*}}" || exit 1
set -e
[ -f {original} ] || cp {surface} {original}
surfaceTransformPoints -origin '{_vector(settings.origin)}' -rollPitchYaw '{rotation}' {original} {surface} > log.apexRotate 2>&1
surfaceTransformPoints -translate '{shift}' {surface} {surface} > log.apexTranslate 2>&1
./Allrun
"""


def write_cases(
    run_sheet: pd.DataFrame,
    base_case: str | Path,
    out_dir: str | Path,
    settings: CaseSettings,
) -> list[Path]:
    """Copy the base case once per design and prepare it to run.

    ``run_sheet`` comes from :func:`apex.aero.doe.run_sheet` and needs one row
    per ``geometry_id`` with the design variables, ``rho`` and ``velocity_ms``.
    """
    base = Path(base_case)
    for required in ("system/controlDict", "Allrun", f"constant/triSurface/{settings.geometry}"):
        if not (base / required).exists():
            raise FileNotFoundError(f"base case has no {required}")
    if not (base / settings.inlet_file).exists():
        raise FileNotFoundError(
            f"base case has no {settings.inlet_file}; set CaseSettings.inlet_file to the file "
            "that holds the free-stream velocity"
        )
    _set_inlet((base / settings.inlet_file).read_text(), settings.inlet_entry, 1.0)
    if re.search(rf"\bcp\b[^\n]*{re.escape(settings.geometry)}", (base / "Allrun").read_text()):
        raise ValueError(
            f"the base Allrun copies {settings.geometry} into the case, which would undo the "
            "design's rotation and ride height; remove that line"
        )
    variables = list(DEFAULT_BOUNDS)
    missing = {"geometry_id", "rho", "velocity_ms", *variables} - set(run_sheet.columns)
    if missing:
        raise ValueError(f"run sheet missing columns: {sorted(missing)}")
    if run_sheet["geometry_id"].duplicated().any():
        raise ValueError("duplicate geometry_id in run sheet")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for _, row in run_sheet.iterrows():
        case = out / str(row["geometry_id"])
        if case.exists():
            raise FileExistsError(f"{case} exists; remove it or choose another out_dir")
        shutil.copytree(base, case, ignore=_skip_outputs)
        control = case / "system" / "controlDict"
        text = _add_function(control.read_text())
        if settings.end_time is not None:
            text = _set_end_time(text, settings.end_time)
        control.write_text(text)
        inlet = case / settings.inlet_file
        inlet.write_text(_set_inlet(inlet.read_text(), settings.inlet_entry, float(row["velocity_ms"])))
        (case / "system" / FORCE_FUNCTION).write_text(
            force_coeffs_dict(settings, float(row["velocity_ms"]), float(row["rho"]))
        )
        script = case / RUN_SCRIPT
        script.write_text(run_script(row, settings))
        script.chmod(0o755)
        design = {key: (value.item() if hasattr(value, "item") else value)
                  for key, value in row[["geometry_id", *variables, "rho", "velocity_ms"]].items()}
        (case / DESIGN_FILE).write_text(json.dumps({**design, "settings": settings.__dict__}, indent=2, default=list))
        cases.append(case)
    return cases


# ---- reading results ---------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    """Whitespace table whose column names are on the last '#' line."""
    header: list[str] | None = None
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            names = line.lstrip("#").split()
            if names and names[0] == "Time":
                header = names
        elif line.strip():
            rows.append(line.split())
    if header is None:
        raise ValueError(f"{path} has no column header")
    width = len(header)
    frame = pd.DataFrame([row[:width] for row in rows if len(row) >= width], columns=header)
    return frame.apply(pd.to_numeric, errors="coerce")


def force_history(case: str | Path) -> pd.DataFrame:
    """Force coefficients per iteration, joined across restarts."""
    root = Path(case) / "postProcessing" / FORCE_FUNCTION
    files = sorted(root.glob("*/coefficient*.dat")) + sorted(root.glob("*/forceCoeffs*.dat"))
    if not files:
        raise FileNotFoundError(f"no force coefficients under {root}")
    history = pd.concat([_read_table(path) for path in files], ignore_index=True)
    return history.drop_duplicates("Time", keep="last").sort_values("Time").reset_index(drop=True)


def final_residuals(log: str | Path) -> dict[str, float]:
    """Initial residual of every field in the last iteration of a solver log."""
    text = Path(log).read_text(errors="replace")
    starts = [match.end() for match in re.finditer(r"^Time = \S+$", text, flags=re.MULTILINE)]
    if not starts:
        raise ValueError(f"{log} has no iterations")
    last = text[starts[-1]:]
    if len(starts) > 1 and "Initial residual" not in last:
        last = text[starts[-2]:starts[-1]]
    found = re.findall(r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)", last)
    return {field: float(value) for field, value in found}


def mesh_cells(case: str | Path) -> float:
    """Cell count from the mesh header or the checkMesh log."""
    case = Path(case)
    owner = case / "constant" / "polyMesh" / "owner"
    if owner.exists():
        head = owner.read_bytes()[:4000].decode("latin-1")
        match = re.search(r"nCells:\s*(\d+)", head)
        if match:
            return float(match.group(1))
    log = case / "log.checkMesh"
    if log.exists():
        match = re.search(r"^\s*cells:\s*(\d+)", log.read_text(errors="replace"), flags=re.MULTILINE)
        if match:
            return float(match.group(1))
    return float("nan")


def solver_setup(case: str | Path) -> str:
    """Solver and turbulence model, so studies with different set-ups are not mixed."""
    case = Path(case)
    control = (case / "system" / "controlDict").read_text()
    application = re.search(r"^application\s+(\w+);", control, flags=re.MULTILINE)
    model = "unknown"
    for name in ("turbulenceProperties", "momentumTransport"):
        path = case / "constant" / name
        if path.exists():
            found = re.search(r"(?:RASModel|model)\s+(\w+);", path.read_text())
            if found:
                model = found.group(1)
                break
    return f"openfoam:{application.group(1) if application else 'unknown'}:{model}"


def read_case(case: str | Path, *, window: int = 50, coefficient_tol: float = 0.01) -> dict:
    """One results row for :data:`apex.schemas.contracts.AERO_COLUMNS`.

    ``cd`` and ``cl`` are averaged over the last ``window`` iterations. A case
    counts as converged when both averages vary by less than
    ``coefficient_tol`` (relative standard deviation) over that window;
    ``residual_max`` is the largest initial residual of the last iteration,
    which ``validate_cfd`` checks separately.
    """
    case = Path(case)
    design = json.loads((case / DESIGN_FILE).read_text())
    history = force_history(case)
    tail = history.tail(window)
    cd, cl = tail["Cd"].mean(), tail["Cl"].mean()
    steady = bool(
        len(history) >= window
        and tail["Cd"].std() <= coefficient_tol * abs(cd)
        and tail["Cl"].std() <= coefficient_tol * max(abs(cl), 1e-9)
    )
    logs = sorted(case.glob("log.*Foam"))
    residuals = final_residuals(logs[-1]) if logs else {}
    return {
        "geometry_id": design["geometry_id"],
        "provenance": "openfoam",
        **{variable: design[variable] for variable in DEFAULT_BOUNDS},
        "rho": design["rho"],
        "velocity_ms": design["velocity_ms"],
        "cd": float(cd),
        "cl": float(cl),
        "mesh_cells": mesh_cells(case),
        "solver_setup": solver_setup(case),
        "residual_max": max(residuals.values()) if residuals else float("nan"),
        "converged": steady,
        "iterations": int(history["Time"].max()),
    }


def collect_results(cases_dir: str | Path, **kwargs) -> pd.DataFrame:
    """Read every generated case under ``cases_dir``; unfinished cases are listed, not dropped."""
    rows = []
    for design in sorted(Path(cases_dir).glob(f"*/{DESIGN_FILE}")):
        try:
            rows.append(read_case(design.parent, **kwargs))
        except (FileNotFoundError, ValueError) as error:
            info = json.loads(design.read_text())
            rows.append({"geometry_id": info["geometry_id"], "provenance": "openfoam_pending",
                         "converged": False, "error": str(error)})
    return pd.DataFrame(rows)


__all__ = [
    "CAR_ATTITUDE_BOUNDS",
    "CaseSettings",
    "collect_results",
    "final_residuals",
    "force_coeffs_dict",
    "force_history",
    "mesh_cells",
    "read_case",
    "write_cases",
]
