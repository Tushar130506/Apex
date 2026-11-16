"""Plackett-Luce finish-order model.

A race is modelled as sequential choice: the winner is drawn with probability
proportional to ``exp(strength)``, then second place from the remaining
drivers, and so on. Training on the observed finishing order uses every
position as evidence instead of one winner label per race, and the win
probabilities are an exact softmax, so no two drivers tie unless their
strengths are identical and nobody is assigned zero probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

# Race-winner model: the win decides most of the loss, the podium refines it.
DEFAULT_STAGE_WEIGHTS = (1.0, 0.5, 0.25)
# Season model: every points-paying position matters for a championship.
SEASON_STAGE_WEIGHTS = (1.0, 0.6, 0.45, 0.3, 0.3, 0.2, 0.2, 0.15, 0.15, 0.1)
# Order-only tie-break noise is far below any real strength difference.
_TIE_EPSILON = 1e-9


@dataclass
class RaceLayout:
    """Padded (race x slot) index of rows sorted by finishing order."""

    index: np.ndarray  # row number per slot, -1 for padding
    valid: np.ndarray  # slot holds a driver
    ranked: np.ndarray  # slot has a known finishing position
    weight: np.ndarray  # one weight per race

    @property
    def races(self) -> int:
        return self.index.shape[0]


def build_layout(
    race_id: pd.Series,
    finish_position: pd.Series | None = None,
    race_weight: pd.Series | None = None,
) -> RaceLayout:
    """Arrange rows into padded race blocks, best finisher first."""
    frame = pd.DataFrame({"race_id": np.asarray(race_id)})
    frame["row"] = np.arange(len(frame))
    if finish_position is None:
        frame["finish"] = np.nan
    else:
        finish = pd.to_numeric(pd.Series(np.asarray(finish_position)), errors="coerce")
        frame["finish"] = finish.where(finish.gt(0)).to_numpy()
    frame["weight"] = 1.0 if race_weight is None else np.asarray(race_weight, dtype=float)
    frame = frame.sort_values(["race_id", "finish", "row"], na_position="last", kind="stable")
    codes, uniques = pd.factorize(frame["race_id"], sort=False)
    slot = frame.groupby(codes).cumcount().to_numpy()
    width = int(slot.max()) + 1 if len(frame) else 0
    index = np.full((len(uniques), width), -1, dtype=int)
    ranked = np.zeros((len(uniques), width), dtype=bool)
    index[codes, slot] = frame["row"].to_numpy()
    ranked[codes, slot] = frame["finish"].notna().to_numpy()
    weight = np.ones(len(uniques))
    weight[codes] = frame["weight"].to_numpy()
    return RaceLayout(index=index, valid=index >= 0, ranked=ranked, weight=weight)


def _stage_terms(
    strengths: np.ndarray, layout: RaceLayout, stage_weights: tuple[float, ...]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted exploded-logit loss with per-row gradient and Hessian diagonal."""
    padded = np.where(layout.valid, strengths[np.clip(layout.index, 0, None)], -np.inf)
    gradient = np.zeros_like(padded, dtype=float)
    hessian = np.zeros_like(padded, dtype=float)
    loss = 0.0
    available = layout.valid.copy()
    for stage, stage_weight in enumerate(stage_weights):
        if stage >= padded.shape[1]:
            break
        # A stage is usable when its finisher is known and at least one rival remains.
        active = layout.ranked[:, stage] & (available.sum(axis=1) > 1)
        if not active.any():
            break
        masked = np.where(available, padded, -np.inf)
        peak = masked.max(axis=1, keepdims=True)
        peak = np.where(np.isfinite(peak), peak, 0.0)
        exponent = np.exp(masked - peak)
        total = exponent.sum(axis=1, keepdims=True)
        probability = exponent / np.where(total > 0, total, 1.0)
        scale = (layout.weight * stage_weight * active)[:, None]
        chosen = np.clip(probability[:, stage], 1e-300, None)
        loss -= float((scale[:, 0] * np.log(chosen))[active].sum())
        gradient += scale * probability
        gradient[:, stage] -= scale[:, 0]
        hessian += scale * probability * (1.0 - probability)
        available[:, stage] = False
    row_gradient = np.zeros(len(strengths))
    row_hessian = np.zeros(len(strengths))
    row_gradient[layout.index[layout.valid]] = gradient[layout.valid]
    row_hessian[layout.index[layout.valid]] = hessian[layout.valid]
    return loss, row_gradient, row_hessian


def exploded_log_loss(
    strengths: np.ndarray, layout: RaceLayout, stage_weights: tuple[float, ...] = (1.0,)
) -> float:
    """Mean weighted negative log-likelihood per race."""
    loss, _, _ = _stage_terms(np.asarray(strengths, dtype=float), layout, stage_weights)
    return loss / max(float(layout.weight.sum()), 1e-12)


def win_probabilities(race_id: pd.Series, strengths: np.ndarray) -> np.ndarray:
    """Exact within-race softmax of log-strengths."""
    values = pd.Series(np.asarray(strengths, dtype=float))
    groups = pd.Series(np.asarray(race_id))
    shifted = values - values.groupby(groups).transform("max")
    exponent = np.exp(shifted)
    return (exponent / exponent.groupby(groups).transform("sum")).to_numpy()


def break_ties(strengths: np.ndarray, *tiebreakers: np.ndarray) -> np.ndarray:
    """Make an ordering strict using secondary keys (smaller key ranks higher).

    Identical strengths are legitimate (two rookies with no history), but a
    predicted finishing order must still be a strict order that does not depend
    on row order. Ties are resolved by the supplied keys, e.g. grid position.
    """
    adjusted = np.asarray(strengths, dtype=float).copy()
    step = _TIE_EPSILON
    for key in tiebreakers:
        numeric = pd.to_numeric(pd.Series(np.asarray(key)), errors="coerce")
        rank = numeric.rank(method="dense", na_option="bottom").to_numpy()
        adjusted -= step * rank
        step /= 100.0
    return adjusted


def sample_orders(
    strengths: np.ndarray, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample complete finishing orders; returns ``(simulations, drivers)`` indices.

    Adding independent Gumbel noise to log-strengths and sorting is an exact
    Plackett-Luce draw of the whole order.
    """
    if simulations < 1:
        raise ValueError("simulations must be positive")
    values = np.asarray(strengths, dtype=float)
    noise = rng.gumbel(size=(simulations, len(values)))
    return np.argsort(-(values[None, :] + noise), axis=1, kind="stable")


def position_probabilities(
    strengths: np.ndarray,
    *,
    later_strengths: np.ndarray | None = None,
    simulations: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Return ``P[driver, position]``; column 0 is the exact softmax.

    ``later_strengths`` optionally gives a second, usually flatter, set of
    log-strengths used for every place behind the winner. The fight for the
    win is more predictable than the order behind it, so one temperature
    cannot be right for both.
    """
    values = np.asarray(strengths, dtype=float)
    later = values if later_strengths is None else np.asarray(later_strengths, dtype=float)
    count = len(values)
    exact = np.exp(values - values.max())
    exact /= exact.sum()
    matrix = np.zeros((count, count))
    matrix[:, 0] = exact
    if count == 1:
        return matrix
    rng = np.random.default_rng(seed)
    winners = np.argmax(values[None, :] + rng.gumbel(size=(simulations, count)), axis=1)
    keys = later[None, :] + rng.gumbel(size=(simulations, count))
    keys[np.arange(simulations), winners] = np.inf
    orders = np.argsort(-keys, axis=1, kind="stable")
    for position in range(1, count):
        matrix[:, position] = np.bincount(orders[:, position], minlength=count)
    # Rescale the simulated places so each driver's row sums to exactly one.
    rest = matrix[:, 1:].sum(axis=1, keepdims=True)
    matrix[:, 1:] *= (1.0 - exact)[:, None] / np.where(rest > 0, rest, 1.0)
    return matrix


def fit_temperature(
    strengths: np.ndarray,
    layout: RaceLayout,
    *,
    stage_weights: tuple[float, ...] = (1.0,),
    prior_strength: float = 10.0,
    bounds: tuple[float, float] = (0.4, 2.5),
) -> float:
    """Choose a strength divisor on already-scored earlier races.

    The fit is continuous (no coarse grid) and shrunk towards 1 with a
    penalty worth ``prior_strength`` pseudo-races, so a short history cannot
    produce an extreme temperature.
    """
    values = np.asarray(strengths, dtype=float)
    if layout.races == 0:
        return 1.0

    def objective(log_temperature: float) -> float:
        loss, _, _ = _stage_terms(values / np.exp(log_temperature), layout, stage_weights)
        return loss + 0.5 * prior_strength * log_temperature**2

    solution = minimize_scalar(
        objective, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded"
    )
    return float(np.exp(solution.x))


@dataclass
class LinearPlackettLuce:
    """Regularised linear exploded-logit model (conditional logit over a race)."""

    l2: float = 3.0
    stage_weights: tuple[float, ...] = DEFAULT_STAGE_WEIGHTS
    coefficients_: np.ndarray | None = None
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fill_: np.ndarray | None = None
    indicator_: np.ndarray | None = None
    names_: list[str] = field(default_factory=list)

    def _design(self, features: pd.DataFrame, fit: bool) -> np.ndarray:
        raw = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        if fit:
            observed = ~np.isnan(raw).all(axis=0)
            self.fill_ = np.zeros(raw.shape[1])
            self.fill_[observed] = np.nanmedian(raw[:, observed], axis=0)
            missing_rate = np.isnan(raw).mean(axis=0)
            self.indicator_ = (missing_rate > 0.01) & (missing_rate < 0.99)
        missing = np.isnan(raw)
        filled = np.where(missing, self.fill_, raw)
        design = np.hstack([filled, missing[:, self.indicator_].astype(float)])
        if fit:
            self.center_ = design.mean(axis=0)
            spread = design.std(axis=0)
            self.scale_ = np.where(spread > 1e-9, spread, 1.0)
            self.names_ = list(features.columns) + [
                f"{name}__missing" for name, keep in zip(features.columns, self.indicator_) if keep
            ]
        return (design - self.center_) / self.scale_

    def fit(self, features: pd.DataFrame, layout: RaceLayout) -> "LinearPlackettLuce":
        design = self._design(features, fit=True)
        total = max(float(layout.weight.sum()), 1e-12)

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            loss, gradient, _ = _stage_terms(design @ coefficients, layout, self.stage_weights)
            penalty = 0.5 * self.l2 * float(coefficients @ coefficients)
            return (loss + penalty) / total, (design.T @ gradient + self.l2 * coefficients) / total

        solution = minimize(
            objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.coefficients_ = solution.x
        return self

    def strengths(self, features: pd.DataFrame) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("model is not fitted")
        return self._design(features, fit=False) @ self.coefficients_

    def coefficient_table(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": self.names_, "coefficient": self.coefficients_})
            .assign(magnitude=lambda frame: frame["coefficient"].abs())
            .sort_values("magnitude", ascending=False)
            .drop(columns="magnitude")
            .reset_index(drop=True)
        )


@dataclass
class BoostedPlackettLuce:
    """Gradient-boosted trees trained directly on the exploded-logit loss."""

    rounds: int = 120
    learning_rate: float = 0.04
    num_leaves: int = 5
    min_child_samples: int = 25
    feature_fraction: float = 0.7
    reg_lambda: float = 5.0
    stage_weights: tuple[float, ...] = DEFAULT_STAGE_WEIGHTS
    seed: int = 0
    booster_: object | None = None

    def fit(self, features: pd.DataFrame, layout: RaceLayout) -> "BoostedPlackettLuce":
        import lightgbm as lgb

        matrix = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        dataset = lgb.Dataset(matrix, label=np.zeros(len(matrix)), free_raw_data=False)
        # Normalise so regularisation strength does not depend on race weights.
        scale = layout.races / max(float(layout.weight.sum()), 1e-12)

        def objective(prediction: np.ndarray, _: object) -> tuple[np.ndarray, np.ndarray]:
            _, gradient, hessian = _stage_terms(prediction, layout, self.stage_weights)
            return gradient * scale, np.maximum(hessian * scale, 1e-6)

        parameters = {
            "objective": objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": min(self.min_child_samples, max(1, len(matrix) // 4)),
            "feature_fraction": self.feature_fraction,
            "lambda_l2": self.reg_lambda,
            "min_sum_hessian_in_leaf": 1e-3,
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
        }
        self.booster_ = lgb.train(parameters, dataset, num_boost_round=self.rounds)
        return self

    def strengths(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("model is not fitted")
        matrix = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return np.asarray(self.booster_.predict(matrix), dtype=float)
