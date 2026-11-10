"""Regularized catch and EPA models with training-only feature normalization."""

from dataclasses import dataclass

import numpy as np


def _sigmoid(logits):
    tail = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + tail), tail / (1 + tail))


@dataclass(frozen=True, slots=True)
class FittedModel:
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    binary: bool

    def predict(self, x):
        """Predict catch probability for a binary fit or mean EPA for a continuous fit."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Prediction features must be finite and match the training columns.")
        linear = self.intercept + ((x - self.mean) / self.scale) @ self.coefficients
        return _sigmoid(linear) if self.binary else linear


def _training_arrays(x, y, ridge, sample_weight):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Provide a nonempty feature matrix and one outcome per row.")
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Features and outcomes must be finite and ridge must be finite and nonnegative.")
    weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Supply finite nonnegative weights with positive total weight.")

    mean = np.average(x, weights=weights, axis=0)
    centered = x - mean
    scale = np.sqrt(np.average(centered ** 2, weights=weights, axis=0))
    scale = np.where(scale > np.finfo(float).eps * np.maximum(np.abs(mean), 1), scale, 1.0)
    return np.column_stack((np.ones(len(x)), centered / scale)), y, weights, mean, scale


def fit_ridge(x, y, ridge=10.0, sample_weight=None):
    """Fit weighted squared error plus a ridge penalty on standardized slopes."""
    design, y, weights, mean, scale = _training_arrays(x, y, ridge, sample_weight)
    if ridge:
        weighted = design.T * weights
        gram = weighted @ design
        slopes = np.arange(1, len(gram))
        gram[slopes, slopes] += ridge
        coefficients = np.linalg.solve(gram, weighted @ y)
    else:
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    return FittedModel(mean, scale, float(coefficients[0]), coefficients[1:], False)


def fit_logistic(x, y, ridge=10.0, sample_weight=None):
    """Fit weighted logistic loss with an unpenalized intercept and ridge on slopes."""
    design, y, weights, mean, scale = _training_arrays(x, y, ridge, sample_weight)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Logistic outcomes must be zero or one.")
    coefficients = np.zeros(design.shape[1])
    positives, negatives = weights[y == 1].sum(), weights[y == 0].sum()
    constant_class = positives == 0 or negatives == 0
    smoothing = 0.5 if constant_class else 0.0
    coefficients[0] = np.log(positives + smoothing) - np.log(negatives + smoothing)
    if constant_class or design.shape[1] == 1:
        return FittedModel(mean, scale, float(coefficients[0]), coefficients[1:], True)

    penalty = np.full(len(coefficients), ridge)
    penalty[0] = 0
    diagonal = np.diag_indices(len(coefficients))
    logits = design @ coefficients
    objective = weights @ (np.logaddexp(0, logits) - y * logits)
    for _ in range(60):
        probability = _sigmoid(logits)
        gradient = design.T @ (weights * (probability - y)) + penalty * coefficients
        curvature = weights * probability * (1 - probability)
        hessian = (design.T * curvature) @ design
        hessian[diagonal] += penalty
        if ridge:
            direction = np.linalg.solve(hessian, gradient)
        else:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.max(np.abs(direction)) < 1e-8:
            break

        # Backtrack when a full Newton step overshoots.
        logit_step = design @ direction
        for power in range(20):
            step = 2.0 ** -power
            candidate = coefficients - step * direction
            candidate_logits = logits - step * logit_step
            loss = weights @ (np.logaddexp(0, candidate_logits) - y * candidate_logits)
            loss += 0.5 * (penalty * candidate) @ candidate
            if loss < objective:
                coefficients, logits, objective = candidate, candidate_logits, loss
                break
        else:
            break
    return FittedModel(mean, scale, float(coefficients[0]), coefficients[1:], True)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    auc: float
    log_loss: float
    brier: float


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


def _prediction_vectors(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("Outcomes and predictions must be nonempty, aligned vectors.")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Outcomes and predictions must be finite.")
    return y, prediction


def binary_metrics(y, probability):
    """Score probabilities with tie-aware AUC, natural-log loss, and Brier loss."""
    y, probability = _prediction_vectors(y, probability)
    if not np.isin(y, (0, 1)).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("A binary score requires binary outcomes and probabilities in [0, 1].")

    positives = int(y.sum())
    negatives = len(y) - positives
    auc = float("nan")
    if positives and negatives:
        _, group, counts = np.unique(probability, return_inverse=True, return_counts=True)
        rank = counts.cumsum() - (counts - 1) / 2
        positive_rank_sum = y @ rank[group]
        auc = float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
    epsilon = np.finfo(float).eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    log_loss = -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    return BinaryMetrics(auc, float(log_loss), float(np.mean((y - probability) ** 2)))


def regression_metrics(y, prediction):
    """Score continuous EPA predictions against the supplied outcomes."""
    y, prediction = _prediction_vectors(y, prediction)
    error = y - prediction
    mse = np.mean(error ** 2)
    variance = y.var()
    return RegressionMetrics(float(np.sqrt(mse)), float(np.mean(np.abs(error))),
                             float(1 - mse / variance) if variance else float("nan"))
