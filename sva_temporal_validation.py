"""Matched, whole-game tests for incremental temporal SVA features."""

from dataclasses import dataclass

import numpy as np

from sva_epa_models import _paired_gains, game_folds
from sva_joint_models import BinaryMetrics, RegressionMetrics, binary_metrics, fit_logistic, fit_ridge, regression_metrics
from sva_joint_validation import _auc_groups, _weighted_auc


@dataclass(frozen=True, slots=True)
class TemporalEvaluation:
    catch_context: BinaryMetrics
    catch_onset: BinaryMetrics
    catch_temporal: BinaryMetrics
    epa_context: RegressionMetrics
    epa_onset: RegressionMetrics
    epa_temporal: RegressionMetrics
    catch_auc_gain_ci: tuple[float, float]
    catch_brier_gain_ci: tuple[float, float]
    epa_rmse_gain_ci: tuple[float, float]
    catch_predictions: np.ndarray
    epa_predictions: np.ndarray


def _game_splits(games, n_folds, seed, forecast_games):
    if forecast_games is None:
        fold_ids = game_folds(games, n_folds=n_folds, seed=seed)
        for fold in np.unique(fold_ids):
            test = fold_ids == fold
            yield np.flatnonzero(~test), np.flatnonzero(test)
        return

    # Restrict fits to games that were excluded from the same original forecast
    # model. Otherwise training-row forecasts could encode test-game outcomes.
    forecast_games = np.asarray(forecast_games)
    original = dict(zip(forecast_games, game_folds(forecast_games, n_folds=5, seed=2021)))
    if set(games) - original.keys():
        raise ValueError("Evaluation includes a game absent from the original forecast folds.")
    outer = np.array([original[game] for game in games])
    for fold in np.unique(outer):
        rows = np.flatnonzero(outer == fold)
        inner = game_folds(games[rows], n_folds=n_folds, seed=seed + int(fold))
        for test_fold in np.unique(inner):
            test = inner == test_fold
            yield rows[~test], rows[test]


def _predictions(y, games, features, fit, ridge, n_folds, seed, forecast_games):
    if y.ndim != 1 or games.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("Supply one finite outcome and game ID per row.")
    features = tuple(np.asarray(x, dtype=float) for x in features)
    if any(x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() for x in features):
        raise ValueError("Context, onset and temporal features must be finite matrices aligned with outcomes.")
    context, onset, temporal = features
    designs = (context, np.column_stack((context, onset)), np.column_stack((context, onset, temporal)))
    predictions = np.empty((len(y), len(designs)))
    for train, test in _game_splits(games, n_folds, seed, forecast_games):
        for column, design in enumerate(designs):
            model = fit(design[train], y[train], ridge=ridge)
            predictions[test, column] = model.predict(design[test])
    return predictions


def _catch_gain_intervals(caught, onset, temporal, games, n_bootstrap, seed):
    _, game_index = np.unique(games, return_inverse=True)
    n_games = int(game_index.max()) + 1
    onset_auc, temporal_auc = _auc_groups(onset), _auc_groups(temporal)
    brier_gain = (caught - onset) ** 2 - (caught - temporal) ** 2
    totals = np.bincount(game_index, weights=brier_gain)
    counts = np.bincount(game_index)
    rng = np.random.default_rng(seed)
    gains = np.empty((n_bootstrap, 2))
    for i in range(n_bootstrap):
        draw = rng.integers(n_games, size=n_games)
        multiplicity = np.bincount(draw, minlength=n_games)
        weights = multiplicity[game_index]
        gains[i] = (_weighted_auc(caught, *temporal_auc, weights) - _weighted_auc(caught, *onset_auc, weights),
                    (multiplicity @ totals) / (multiplicity @ counts))
    intervals = []
    for values in gains.T:
        finite = values[np.isfinite(values)]
        intervals.append(tuple(map(float, np.quantile(finite, [0.025, 0.975]))) if len(finite)
                         else (float("nan"), float("nan")))
    return tuple(intervals)


def evaluate_temporal(caught, catch_games, catch_context, catch_onset, catch_temporal,
                      epa, epa_games, epa_context, epa_onset, epa_temporal, *, n_folds=5, seed=2021,
                      n_bootstrap=2000, forecast_games=None, catch_ridge=35, epa_ridge=120):
    """Compare context, onset and temporal models on identical rows within each outcome.

    Temporal matrices are the additional features after onset. The caller must
    enforce the landmark cutoff and common landmark cohort before calling this
    function. When features include forecasts trained with the original five
    game folds, pass the full game population from that forecast run so every
    validation fit stays within one original held-out fold.

    All intervals compare the temporal model with the onset model, resampling
    whole games in pairs and holding the held-out predictions fixed. Positive
    gains indicate better discrimination or lower prediction error.
    """
    caught, epa = np.asarray(caught, dtype=float), np.asarray(epa, dtype=float)
    catch_games, epa_games = np.asarray(catch_games), np.asarray(epa_games)
    if n_bootstrap < 1:
        raise ValueError("Use at least one bootstrap resample.")
    catch_pred = _predictions(caught, catch_games, (catch_context, catch_onset, catch_temporal),
                              fit_logistic, catch_ridge, n_folds, seed, forecast_games)
    epa_pred = _predictions(epa, epa_games, (epa_context, epa_onset, epa_temporal),
                            fit_ridge, epa_ridge, n_folds, seed, forecast_games)
    auc_ci, brier_ci = _catch_gain_intervals(caught, catch_pred[:, 1], catch_pred[:, 2],
                                             catch_games, n_bootstrap, seed)
    rmse_ci, _ = _paired_gains(epa, epa_pred[:, 1], epa_pred[:, 2], epa_games, n_bootstrap, seed)
    return TemporalEvaluation(
        *(binary_metrics(caught, pred) for pred in catch_pred.T),
        *(regression_metrics(epa, pred) for pred in epa_pred.T),
        auc_ci, brier_ci, rmse_ci, catch_pred, epa_pred,
    )
