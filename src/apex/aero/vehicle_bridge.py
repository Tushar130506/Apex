"""EXPERIMENTAL vehicle-performance bridge: (Cd, Cl) -> force curves -> lap-time delta.

Quasi-steady point-mass lap simulation on a generic made-up circuit layout. Parameters are generic assumptions,
NOT a validated F1 car. Output is an *estimate* for comparing designs under identical assumptions.
Drag = 0.5 rho v^2 Cd A ; downforce = 0.5 rho v^2 Cl A (positive = downforce here).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

G = 9.81


@dataclass(frozen=True)
class Vehicle:
    mass: float = 800.0          # kg incl. driver/fuel
    power_w: float = 700e3
    frontal_area: float = 1.5    # m^2 (reference area A used by Cd/Cl - must match CFD definition)
    rho: float = 1.225
    mu_tyre: float = 1.8
    max_brake_g: float = 4.0     # tyre-limited braking at 1g normal load, scaled with load
    v_max: float = 100.0         # m/s hard cap
    rolling_c: float = 0.015


@dataclass(frozen=True)
class Circuit:
    """Piecewise track: segment lengths (m) and corner radii (m; np.inf = straight)."""
    lengths: tuple
    radii: tuple

    @staticmethod
    def generic() -> "Circuit":
        L = (900, 120, 180, 350, 90, 250, 600, 140, 200, 500, 80, 300)
        R = (np.inf, 60, 120, np.inf, 35, 180, np.inf, 70, 250, np.inf, 28, np.inf)
        return Circuit(L, R)


def forces(v, cd: float, cl: float, veh: Vehicle):
    q = 0.5 * veh.rho * np.asarray(v) ** 2
    return q * cd * veh.frontal_area, q * cl * veh.frontal_area  # drag, downforce


def lap_time(cd: float, cl: float, veh: Vehicle = Vehicle(), circuit: Circuit = Circuit.generic(),
             ds: float = 5.0) -> float:
    xs, rad = [], []
    for L, R in zip(circuit.lengths, circuit.radii):
        n = max(1, int(round(L / ds)))
        xs += [L / n] * n; rad += [R] * n
    dx, R = np.array(xs), np.array(rad)
    m = veh.mass
    # corner-limited speed: m v^2/R = mu (m g + k v^2), k = 0.5 rho Cl A
    k = 0.5 * veh.rho * cl * veh.frontal_area
    denom = m / R - veh.mu_tyre * k
    with np.errstate(divide="ignore", invalid="ignore"):
        vc = np.where(np.isfinite(R), np.sqrt(np.where(denom > 0, veh.mu_tyre * m * G / np.maximum(denom, 1e-9), np.inf)), np.inf)
    vlim = np.minimum(vc, veh.v_max)
    v = vlim.copy()
    # backward pass: braking
    for i in range(len(v) - 2, -1, -1):
        v_next = v[(i + 1) % len(v)] if i + 1 < len(v) else v[0]
        drag, down = forces(v[i], cd, cl, veh)
        a_brake = (veh.mu_tyre * (m * G + down) / m) * min(1.0, veh.max_brake_g / (veh.mu_tyre)) + drag / m
        v[i] = min(v[i], np.sqrt(v_next ** 2 + 2 * a_brake * dx[i]))
    # forward pass: power / traction limited acceleration
    v[0] = min(v[0], v[-1]) if False else v[0]
    for _ in range(2):  # two sweeps so the start/finish wrap is consistent
        for i in range(len(v) - 1):
            drag, down = forces(v[i], cd, cl, veh)
            f_trac = veh.mu_tyre * (m * G + down)
            f_drive = min(veh.power_w / max(v[i], 5.0), f_trac)
            a = (f_drive - drag - veh.rolling_c * m * G) / m
            v[i + 1] = min(vlim[i + 1], v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * a * dx[i], 1.0)))
        v[0] = min(v[0], v[-1])
    return float(np.sum(dx / np.maximum(0.5 * (v + np.roll(v, -1)), 1.0)))


def lap_time_delta(design: tuple[float, float], baseline: tuple[float, float], **kw) -> float:
    """Negative = faster than baseline. Same vehicle/circuit assumptions for both."""
    return lap_time(*design, **kw) - lap_time(*baseline, **kw)


def sensitivity(design, baseline, grid: dict[str, list[float]] | None = None) -> list[dict]:
    """Does the sign/size of the lap-time delta survive plausible changes to the assumptions?"""
    grid = grid or {"mass": [750, 800, 850], "power_w": [600e3, 700e3, 800e3], "mu_tyre": [1.5, 1.8, 2.1]}
    base = Vehicle(); out = []
    for key, vals in grid.items():
        for v in vals:
            out.append(dict(param=key, value=v, delta_s=lap_time_delta(design, baseline, veh=replace(base, **{key: v}))))
    return out
