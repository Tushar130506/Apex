#!/usr/bin/env python
"""Set up an OpenFOAM study over car attitude, and read its results back.

  # 1. One case per design (Latin hypercube over pitch, ride height and yaw)
  python scripts/cfd_study.py write --base cases/base --out cases/study \
      --geometry car.obj --patches carGroup --aref 1.2 --lref 3.6 \
      --origin 1.8 0 0 --designs 20 --velocity 50 --end-time 1500
  #    Designs span pitch -1 to 2 deg, ride height +0 to 40 mm, yaw 0 to 4 deg.
  #    Pass the same bounds to apex.aero.optimize.propose when searching.
  # 2. Run each case: cases/study/<geometry_id>/Allrun.apex (on your machine or cluster)
  # 3. Collect results; validated rows go to the surrogate
  python scripts/cfd_study.py collect --cases cases/study --out cases/study/results.csv

``--base`` is a case you have already run by hand (for example a copy of the
motorBike tutorial with your geometry); see ``apex.aero.openfoam`` for what it
must contain.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, "src")
from apex.aero.cfd_results import validate_cfd
from apex.aero.doe import latin_hypercube, run_sheet
from apex.aero.openfoam import CAR_ATTITUDE_BOUNDS, CaseSettings, collect_results, write_cases
from apex.schemas.contracts import ContractError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write", help="create one OpenFOAM case per design")
    write.add_argument("--base", required=True)
    write.add_argument("--out", required=True)
    write.add_argument("--geometry", required=True, help="file name in the base case's constant/triSurface")
    write.add_argument("--patches", nargs="+", required=True)
    write.add_argument("--aref", type=float, required=True, help="reference area in m^2")
    write.add_argument("--lref", type=float, required=True, help="reference length in m")
    write.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    write.add_argument("--designs", type=int, default=20)
    write.add_argument("--seed", type=int, default=0)
    write.add_argument("--velocity", type=float, default=50.0, help="m/s")
    write.add_argument("--rho", type=float, default=1.225)
    write.add_argument("--end-time", type=int, default=None)
    collect = sub.add_parser("collect", help="read finished cases into one results table")
    collect.add_argument("--cases", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--window", type=int, default=50, help="iterations averaged for Cd and Cl")
    args = parser.parse_args()

    if args.command == "write":
        sheet = run_sheet(latin_hypercube(args.designs, CAR_ATTITUDE_BOUNDS, seed=args.seed), rho=args.rho,
                          velocity_ms=args.velocity, solver="openfoam")
        settings = CaseSettings(
            geometry=args.geometry, patches=tuple(args.patches), reference_area_m2=args.aref,
            reference_length_m=args.lref, origin=tuple(args.origin), end_time=args.end_time,
        )
        cases = write_cases(sheet, args.base, args.out, settings)
        sheet.to_csv(Path(args.out) / "run_sheet.csv", index=False)
        print(f"wrote {len(cases)} cases to {args.out}; run each one's Allrun.apex")
        return

    table = collect_results(args.cases, window=args.window)
    table.to_csv(args.out, index=False)
    done = table[table["provenance"].eq("openfoam")]
    print(f"{len(done)} of {len(table)} cases finished; {int(done['converged'].sum())} converged -> {args.out}")
    if done.empty:
        return
    try:
        validate_cfd(done)
        print("all finished cases pass validate_cfd")
    except ContractError as error:
        print(f"not ready for the surrogate yet: {error}")


if __name__ == "__main__":
    main()
