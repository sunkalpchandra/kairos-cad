"""Learned surrogate for expensive geometric metrics.

Optimizing a design means evaluating thousands of candidates. Building one in
FreeCAD costs ~0.3 s and measuring its wall thickness costs ~0.5 s more, so a
ten-thousand-candidate search is hours. The surrogate predicts mass and minimum
wall thickness directly from a family's parameters in microseconds, which is
what makes search practical; FreeCAD then verifies only the winner.

Mass is cheap to compute analytically for these families; minimum wall
thickness is not, since it requires the ray casting in
:mod:`kairos.evaluation.wall_thickness`. Both are predicted so the optimizer
can trade them off, and both are verified exactly before any result is reported.

The surrogate is a closed-form ridge fit on polynomial features. Training sets
here are hundreds of rows, since each row costs a real FreeCAD build, and a
closed-form fit needs no torch, so search and verification share one
interpreter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sample:
    """One evaluated design: its parameters and what they produced."""

    parameters: dict[str, float]
    mass_g: float
    min_wall_thickness_mm: float | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SurrogateData:
    """Training rows for one family, plus the parameter order they use."""

    family: str
    parameter_names: list[str] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` over samples with a measured thickness.

        Rows whose thickness could not be measured are dropped rather than
        imputed: a fabricated target would teach the surrogate a value the
        geometry never had.
        """
        usable = [s for s in self.samples if s.valid and s.min_wall_thickness_mm is not None]
        if not usable:
            return np.zeros((0, len(self.parameter_names))), np.zeros((0, 2))
        x = np.array(
            [[s.parameters[n] for n in self.parameter_names] for s in usable], dtype=np.float64
        )
        y = np.array(
            [[s.mass_g, s.min_wall_thickness_mm] for s in usable], dtype=np.float64
        )
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "parameter_names": self.parameter_names,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SurrogateData:
        return cls(
            family=payload["family"],
            parameter_names=list(payload["parameter_names"]),
            samples=[Sample(**s) for s in payload["samples"]],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SurrogateData:
        return cls.from_dict(json.loads(Path(path).read_text()))


class Standardizer:
    """Zero-mean, unit-variance scaling fitted on the training rows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Standardizer:
        self.mean = x.mean(axis=0)
        # A constant column has zero spread; dividing by it would produce inf.
        self.scale = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer used before fit()")
        return (x - self.mean) / self.scale

    def inverse(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer used before fit()")
        return x * self.scale + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Standardizer:
        s = cls()
        s.mean = None if payload["mean"] is None else np.asarray(payload["mean"])
        s.scale = None if payload["scale"] is None else np.asarray(payload["scale"])
        return s


@dataclass
class SurrogateMetrics:
    """Held-out accuracy, per target."""

    mass_mae: float
    thickness_mae: float
    mass_r2: float
    thickness_r2: float
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """1 - SSE/SST; 0 means no better than predicting the mean."""
    variance = float(((actual - actual.mean()) ** 2).sum())
    if variance < 1e-12:
        return float("nan")
    return float(1.0 - ((actual - predicted) ** 2).sum() / variance)


class RidgeSurrogate:
    """Closed-form ridge regression on polynomial features.

    Chosen over a neural network on purpose: with a few hundred rows per family, each
    one a real FreeCAD build, a closed-form fit has no optimizer, no
    seed, and no training curve to misread, and mass is genuinely close to
    polynomial in the parameters. It also has no torch dependency, so the
    optimizer runs under FreeCAD's interpreter where the verification happens.
    """

    def __init__(self, degree: int = 3, alpha: float = 1e-3) -> None:
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.x_scaler = Standardizer()
        self.parameter_names: list[str] = []

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Bias, linear, then all monomials up to ``degree``.

        Degree 3 is the default rather than 2 because mass is a *three-way*
        product of dimensions, a plate's is width x height x thickness, and a
        quadratic simply cannot represent that. Fitted at degree 2 the surrogate
        still scores R^2 ~ 0.997 overall while getting the thickness direction
        wrong, so an optimizer following it walks the wrong way on exactly the
        parameter the manufacturing constraint governs.
        """
        columns = [np.ones((x.shape[0], 1)), x]
        n = x.shape[1]
        if self.degree >= 2:
            for i in range(n):
                for j in range(i, n):
                    columns.append((x[:, i] * x[:, j]).reshape(-1, 1))
        if self.degree >= 3:
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        columns.append((x[:, i] * x[:, j] * x[:, k]).reshape(-1, 1))
        return np.hstack(columns)

    def fit(self, x: np.ndarray, y: np.ndarray, parameter_names: list[str]) -> RidgeSurrogate:
        self.parameter_names = list(parameter_names)
        design = self._features(self.x_scaler.fit(x).transform(x))
        identity = np.eye(design.shape[1])
        identity[0, 0] = 0.0  # never penalize the bias
        self.weights = np.linalg.solve(
            design.T @ design + self.alpha * identity, design.T @ y
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("surrogate used before fit()")
        return self._features(self.x_scaler.transform(np.atleast_2d(x))) @ self.weights

    def predict_one(self, parameters: dict[str, float]) -> tuple[float, float]:
        """Predict ``(mass_g, min_wall_thickness_mm)`` for one parameter set."""
        row = np.array([[parameters[n] for n in self.parameter_names]], dtype=np.float64)
        out = self.predict(row)[0]
        return float(out[0]), float(out[1])

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        predicted = self.predict(x)
        return {
            "mass_mae": float(np.abs(predicted[:, 0] - y[:, 0]).mean()),
            "thickness_mae": float(np.abs(predicted[:, 1] - y[:, 1]).mean()),
            "mass_r2": r_squared(predicted[:, 0], y[:, 0]),
            "thickness_r2": r_squared(predicted[:, 1], y[:, 1]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "alpha": self.alpha,
            "weights": None if self.weights is None else self.weights.tolist(),
            "x_scaler": self.x_scaler.to_dict(),
            "parameter_names": self.parameter_names,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RidgeSurrogate:
        model = cls(degree=payload["degree"], alpha=payload["alpha"])
        model.weights = None if payload["weights"] is None else np.asarray(payload["weights"])
        model.x_scaler = Standardizer.from_dict(payload["x_scaler"])
        model.parameter_names = list(payload["parameter_names"])
        return model

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RidgeSurrogate:
        return cls.from_dict(json.loads(Path(path).read_text()))


def train_surrogate(
    data: SurrogateData, degree: int = 3, alpha: float = 1e-3, test_fraction: float = 0.2, seed: int = 0
) -> tuple[RidgeSurrogate, SurrogateMetrics]:
    """Fit a surrogate and report held-out accuracy."""
    x, y = data.matrix()
    if len(x) < 8:
        raise ValueError(f"need at least 8 usable samples, have {len(x)}")

    order = np.random.default_rng(seed).permutation(len(x))
    n_test = max(2, int(round(len(x) * test_fraction)))
    test, train = order[:n_test], order[n_test:]

    model = RidgeSurrogate(degree=degree, alpha=alpha)
    model.fit(x[train], y[train], data.parameter_names)
    scores = model.evaluate(x[test], y[test])
    return model, SurrogateMetrics(
        mass_mae=scores["mass_mae"],
        thickness_mae=scores["thickness_mae"],
        mass_r2=scores["mass_r2"],
        thickness_r2=scores["thickness_r2"],
        train_rows=len(train),
        test_rows=len(test),
    )
