"""OpenFOAM case generation and result reading, with hand-written case files."""
import json

import numpy as np
import pandas as pd
import pytest

from apex.aero.cfd_results import validate_cfd
from apex.aero.doe import latin_hypercube, run_sheet
from apex.aero.openfoam import CaseSettings, collect_results, final_residuals, read_case, write_cases

SETTINGS = CaseSettings(geometry="car.obj", patches=("carGroup",), reference_area_m2=1.2,
                        reference_length_m=3.6, origin=(1.8, 0, 0), end_time=400)


def _base(tmp_path, allrun="#!/bin/sh\nblockMesh\n"):
    base = tmp_path / "base"
    (base / "system").mkdir(parents=True)
    (base / "constant" / "triSurface").mkdir(parents=True)
    (base / "0.orig" / "include").mkdir(parents=True)
    (base / "system" / "controlDict").write_text(
        "application     simpleFoam;\nendTime         2000;\nfunctions\n{\n    #include \"streamLines\"\n}\n")
    (base / "Allrun").write_text(allrun)
    (base / "constant" / "triSurface" / "car.obj").write_text("v 0 0 0\n")
    (base / "constant" / "turbulenceProperties").write_text("RAS\n{\n    RASModel        kOmegaSST;\n}\n")
    (base / "0.orig" / "include" / "initialConditions").write_text("flowVelocity         (20 0 0);\npressure 0;\n")
    (base / "500").mkdir()                     # results of an earlier run of the base case
    (base / "log.simpleFoam").write_text("old")
    return base


def test_write_cases_prepares_one_case_per_design(tmp_path):
    sheet = run_sheet(latin_hypercube(3, seed=1), velocity_ms=50.0)
    cases = write_cases(sheet, _base(tmp_path), tmp_path / "study", SETTINGS)
    assert [case.name for case in cases] == sheet["geometry_id"].tolist()
    case = cases[0]
    control = (case / "system" / "controlDict").read_text()
    assert '#include "apexForceCoeffs"' in control and "endTime         400;" in control
    assert "flowVelocity         (50 0 0);" in (case / "0.orig" / "include" / "initialConditions").read_text()
    forces = (case / "system" / "apexForceCoeffs").read_text()
    assert "liftDir         (0 0 -1);" in forces and "magUInf         50;" in forces and "Aref            1.2;" in forces
    script = (case / "Allrun.apex").read_text()
    row = sheet.iloc[0]
    assert f"-rollPitchYaw '(0 {row.aoa_deg:g} {row.yaw_deg:g})'" in script
    assert f"-translate '(0 0 {row.ride_height_mm / 1000:g})'" in script and script.rstrip().endswith("./Allrun")
    assert not (case / "500").exists() and not (case / "log.simpleFoam").exists()
    assert json.loads((case / "apex_design.json").read_text())["geometry_id"] == row.geometry_id
    with pytest.raises(FileExistsError):
        write_cases(sheet, tmp_path / "base", tmp_path / "study", SETTINGS)


def test_base_allrun_that_recopies_geometry_is_refused(tmp_path):
    base = _base(tmp_path, allrun="#!/bin/sh\ncp $FOAM_TUTORIALS/car.obj constant/triSurface/\n")
    with pytest.raises(ValueError, match="copies car.obj"):
        write_cases(run_sheet(latin_hypercube(1)), base, tmp_path / "study", SETTINGS)


def _finished_case(tmp_path, cd_noise=0.0):
    sheet = run_sheet(latin_hypercube(1, seed=2), velocity_ms=50.0)
    case = write_cases(sheet, _base(tmp_path), tmp_path / "study", SETTINGS)[0]
    out = case / "postProcessing" / "apexForceCoeffs" / "0"
    out.mkdir(parents=True)
    steps = np.arange(1, 201)
    rng = np.random.default_rng(0)
    cd = 0.9 + 0.5 * np.exp(-steps / 20) + cd_noise * rng.standard_normal(len(steps))
    cl = 1.6 - 0.4 * np.exp(-steps / 20)
    lines = ["# Force coefficients", "# Time      Cd   Cd(f)   Cd(r)   Cl   Cl(f)   Cl(r)   CmPitch"]
    lines += [f"{t}\t{a:.6f}\t0\t0\t{b:.6f}\t0\t0\t0" for t, a, b in zip(steps, cd, cl)]
    (out / "coefficient.dat").write_text("\n".join(lines) + "\n")
    (case / "log.simpleFoam").write_text(
        "Time = 199\n\nsmoothSolver:  Solving for Ux, Initial residual = 0.01, Final residual = 1e-4, No Iterations 2\n"
        "Time = 200\n\nsmoothSolver:  Solving for Ux, Initial residual = 0.0004, Final residual = 1e-5, No Iterations 2\n"
        "GAMG:  Solving for p, Initial residual = 0.0009, Final residual = 1e-5, No Iterations 4\nEnd\n")
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "owner").write_bytes(
        b'FoamFile\n{\n    note        "nPoints:100  nCells:345678  nFaces:900";\n}\n\x00\x01binary')
    return case, sheet.iloc[0]


def test_read_case_fills_the_cfd_contract(tmp_path):
    case, row = _finished_case(tmp_path)
    result = read_case(case)
    assert result["cd"] == pytest.approx(0.9, abs=1e-3) and result["cl"] == pytest.approx(1.6, abs=1e-3)
    assert result["converged"] and result["residual_max"] == pytest.approx(0.0009)
    assert result["mesh_cells"] == 345678 and result["solver_setup"] == "openfoam:simpleFoam:kOmegaSST"
    assert result["aoa_deg"] == pytest.approx(row.aoa_deg) and result["iterations"] == 200
    table = collect_results(tmp_path / "study")
    validate_cfd(table)


def test_noisy_coefficients_are_not_converged(tmp_path):
    case, _ = _finished_case(tmp_path, cd_noise=0.05)
    assert not read_case(case)["converged"]


def test_unfinished_cases_are_listed_not_dropped(tmp_path):
    write_cases(run_sheet(latin_hypercube(2)), _base(tmp_path), tmp_path / "study", SETTINGS)
    table = collect_results(tmp_path / "study")
    assert len(table) == 2 and set(table["provenance"]) == {"openfoam_pending"} and not table["converged"].any()


def test_final_residuals_reads_the_last_iteration(tmp_path):
    log = tmp_path / "log.simpleFoam"
    log.write_text("Time = 1\nsmoothSolver:  Solving for k, Initial residual = 1, Final residual = 0.1, No Iterations 1\n"
                   "Time = 2\nsmoothSolver:  Solving for k, Initial residual = 0.2, Final residual = 0.01, No Iterations 1\n")
    assert final_residuals(log) == {"k": 0.2}
