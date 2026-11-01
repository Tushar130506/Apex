"""Gaussian-process surrogate for CFD coefficients with extrapolation flags (blueprint section 7)."""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=Warning, module="sklearn.gaussian_process")
DESIGN_VARS = ["aoa_deg", "ride_height_mm", "yaw_deg"]


@dataclass
class AeroPrediction:
    mean: np.ndarray            # columns: cd, cl
    std: np.ndarray
    extrapolating: np.ndarray   # outside sampled bounds or std above threshold


class AeroSurrogate:
    def __init__(self, design_vars: list[str] | None = None, std_flag: float = 0.05, seed: int = 0):
        self.vars = design_vars or DESIGN_VARS
        self.targets = ["cd", "cl"]
        self.std_flag, self.seed = std_flag, seed
        self.models: dict[str, GaussianProcessRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "AeroSurrogate":
        self.scaler = StandardScaler().fit(df[self.vars])
        X = self.scaler.transform(df[self.vars])
        self.bounds = (df[self.vars].min().to_numpy(), df[self.vars].max().to_numpy())
        for t in self.targets:
            y = df[t].to_numpy()
            self.ymu, self.ysd = getattr(self, "ymu", {}), getattr(self, "ysd", {})
            self.ymu[t], self.ysd[t] = y.mean(), y.std() + 1e-12
            k = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(self.vars)), nu=2.5) + WhiteKernel(1e-3)
            self.models[t] = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=3,
                                                      random_state=self.seed).fit(X, (y - self.ymu[t]) / self.ysd[t])
        return self

    def predict(self, df: pd.DataFrame) -> AeroPrediction:
        X = self.scaler.transform(df[self.vars])
        means, stds = [], []
        for t in self.targets:
            m, s = self.models[t].predict(X, return_std=True)
            means.append(m * self.ysd[t] + self.ymu[t]); stds.append(s * self.ysd[t])
        vals = df[self.vars].to_numpy()
        outside = ((vals < self.bounds[0]) | (vals > self.bounds[1])).any(axis=1)
        std = np.column_stack(stds)
        return AeroPrediction(np.column_stack(means), std, outside | (std > self.std_flag).any(axis=1))

    def loo_error(self, df: pd.DataFrame) -> dict:
        """Leave-one-design-out error (refits hyperparameters each fold; fine for DoE-sized data)."""
        errs = {t: [] for t in self.targets}
        for tr, te in LeaveOneOut().split(df):
            m = AeroSurrogate(self.vars, self.std_flag, self.seed).fit(df.iloc[tr])
            p = m.predict(df.iloc[te]).mean[0]
            for i, t in enumerate(self.targets):
                errs[t].append(p[i] - df.iloc[te][t].iloc[0])
        return {t: dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(np.square(e)))),
                        max_abs=float(np.max(np.abs(e)))) for t, e in errs.items()}
