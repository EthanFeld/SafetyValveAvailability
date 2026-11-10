"""Whole-game cross-validation and game-clustered EPA estimates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True, slots=True)
class IncrementalEPA:
    baseline: PredictionMetrics
    augmented: PredictionMetrics
    baseline_predictions: np.ndarray
    augmented_predictions: np.ndarray
    delta_rmse: float
    delta_rmse_ci: tuple[float, float]
    delta_r2: float
    delta_r2_ci: tuple[float, float]
    coefficient_per_10: float
    coefficient_ci: tuple[float, float]


def game_folds(games, n_folds=5, seed=2021):
    """Assign plays from the same game to one seeded, balanced fold."""
    games = np.asarray(games)
    if games.ndim != 1 or n_folds < 2:
        raise ValueError("Provide a vector of game IDs and at least two folds.")
    game_ids, index = np.unique(games, return_inverse=True)
    n_games = len(game_ids)
    if n_games < 2:
        raise ValueError("At least two games are required for cross-validation.")
    rng = np.random.default_rng(seed)
    game_fold = np.empty(n_games, dtype=int)
    game_fold[rng.permutation(n_games)] = np.arange(n_games) % min(n_games, n_folds)
    return game_fold[index]


def out_of_fold_predictions(outcome, design, folds, train_mask=None, predict_mask=None, ridge=0.0):
    """Fit on other folds, optionally using distinct training and prediction populations.

    An intercept is fitted automatically and excluded from the optional ridge penalty.
    Ridge strength applies in the supplied units. Rows outside predict_mask return NaN.
    """
    outcome, design, folds = np.asarray(outcome, dtype=float), np.asarray(design, dtype=float), np.asarray(folds)
    if outcome.ndim != 1 or design.ndim != 2 or len(design) != len(outcome) or folds.shape != outcome.shape:
        raise ValueError("Outcome, design and fold assignments must align by play.")
    train_mask = np.ones(len(outcome), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    predict_mask = (np.ones(len(outcome), dtype=bool) if predict_mask is None
                    else np.asarray(predict_mask, dtype=bool))
    if train_mask.shape != outcome.shape or predict_mask.shape != outcome.shape:
        raise ValueError("Masks must align by play.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge strength must be finite and nonnegative.")
    if not (np.isfinite(outcome[train_mask]).all() and np.isfinite(design[train_mask | predict_mask]).all()):
        raise ValueError("Training outcomes and used design rows must be finite.")

    design = np.column_stack((np.ones(len(outcome)), design))
    predictions = np.full(len(outcome), np.nan)
    n_features = design.shape[1]
    penalty = np.sqrt(ridge) * np.eye(n_features)[1:]
    for fold in np.unique(folds[predict_mask]):
        train = train_mask & (folds != fold)
        predict = predict_mask & (folds == fold)
        if not train.any():
            raise ValueError(f"No training plays outside fold {fold}.")
        x, y = design[train], outcome[train]
        if ridge:
            x = np.vstack((x, penalty))
            y = np.concatenate((y, np.zeros(n_features - 1)))
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        predictions[predict] = design[predict] @ coefficients
    return predictions


def clustered_coefficient(outcome, predictor, controls, games):
    """OLS slope and 95% interval using finite-sample CR1 errors clustered by game."""
    outcome, predictor = np.asarray(outcome, dtype=float), np.asarray(predictor, dtype=float)
    controls, games = np.asarray(controls, dtype=float), np.asarray(games)
    design = np.column_stack((np.ones(len(outcome)), controls))
    pair = np.column_stack((predictor, outcome))
    fit, _, rank, _ = np.linalg.lstsq(design, pair, rcond=None)
    x, y = (pair - design @ fit).T
    x_ss = x @ x
    _, game_index = np.unique(games, return_inverse=True)
    n_games = len(np.unique(game_index))
    dof = len(outcome) - rank - 1
    if n_games < 2 or dof < 1 or x_ss < 1e-12:
        return float("nan"), (float("nan"), float("nan"))

    slope = float((x @ y) / x_ss)
    error = y - slope * x
    scores = np.bincount(game_index, weights=x * error)
    correction = n_games / (n_games - 1) * (len(outcome) - 1) / dof
    half_width = float(1.96 * np.sqrt(correction * (scores @ scores)) / x_ss)
    return slope, (slope - half_width, slope + half_width)


def _prediction_metrics(outcome, predictions):
    error = outcome - predictions
    mse = (error ** 2).mean()
    variance = outcome.var()
    r2 = float(1 - mse / variance) if variance > 0 else float("nan")
    return PredictionMetrics(float(np.sqrt(mse)), float(np.abs(error).mean()), r2)


def _interval(values):
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan"), float("nan")
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high)


def _paired_gains(outcome, baseline, augmented, games, n_bootstrap, seed):
    """Resample whole games, holding each cross-fitted prediction fixed."""
    game_ids, game_index = np.unique(games, return_inverse=True)
    baseline_sq = (outcome - baseline) ** 2
    augmented_sq = (outcome - augmented) ** 2
    moments = np.column_stack([np.bincount(game_index, weights=values) for values in (
        np.ones(len(outcome)), outcome, outcome ** 2, baseline_sq, augmented_sq,
    )])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(game_ids), size=(n_bootstrap, len(game_ids)))
    n, y, y2, base_error, aug_error = moments[draws].sum(axis=1).T
    rmse_gains = np.sqrt(base_error / n) - np.sqrt(aug_error / n)
    total_ss = y2 - y ** 2 / n
    r2_gains = np.divide(base_error - aug_error, total_ss, out=np.full(n_bootstrap, np.nan), where=total_ss > 0)
    return _interval(rmse_gains), _interval(r2_gains)


def incremental_epa(outcome, controls, best_sva, games, *, folds=None, ridge=0.0, n_bootstrap=2000, seed=2021):
    """Compare held-out EPA prediction with and without a single SVA predictor.

    Use the same rows and folds for both designs. Positive delta_rmse and delta_r2
    denote improvement. The full-sample adjusted slope uses OLS regardless of ridge.
    """
    outcome = np.asarray(outcome, dtype=float)
    controls = np.asarray(controls, dtype=float)
    best_sva = np.asarray(best_sva, dtype=float)
    games = np.asarray(games)
    if (outcome.ndim != 1 or best_sva.shape != outcome.shape or games.shape != outcome.shape
            or controls.ndim != 2 or len(controls) != len(outcome) or n_bootstrap < 0):
        raise ValueError("EPA, SVA, controls and games must align, with a nonnegative bootstrap count.")
    if folds is None:
        folds = game_folds(games, seed=seed)
    else:
        folds = np.asarray(folds)
        if folds.shape != games.shape:
            raise ValueError("Fold assignments must align by play.")
        _, first, inverse = np.unique(games, return_index=True, return_inverse=True)
        if not np.array_equal(folds[first][inverse], folds):
            raise ValueError("A game cannot be split across cross-validation folds.")

    sva_per_10 = best_sva / 10
    baseline = out_of_fold_predictions(outcome, controls, folds, ridge=ridge)
    augmented = out_of_fold_predictions(outcome, np.column_stack((controls, sva_per_10)), folds, ridge=ridge)
    baseline_metrics = _prediction_metrics(outcome, baseline)
    augmented_metrics = _prediction_metrics(outcome, augmented)
    rmse_ci, r2_ci = _paired_gains(outcome, baseline, augmented, games, n_bootstrap, seed)
    coefficient, coefficient_ci = clustered_coefficient(outcome, sva_per_10, controls, games)
    return IncrementalEPA(
        baseline_metrics, augmented_metrics, baseline, augmented,
        baseline_metrics.rmse - augmented_metrics.rmse, rmse_ci,
        augmented_metrics.r2 - baseline_metrics.r2, r2_ci, coefficient, coefficient_ci,
    )
