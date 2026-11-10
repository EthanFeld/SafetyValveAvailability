"""Select joint catch/EPA models with nested whole-game validation."""

from dataclasses import dataclass

import numpy as np

from sva_epa_models import game_folds


@dataclass(frozen=True, slots=True)
class JointMetrics:
    catch_auc: float
    catch_brier: float
    catch_log_loss: float
    epa_rmse: float
    epa_mae: float
    epa_r2: float


@dataclass(frozen=True, slots=True)
class JointValidation:
    candidates: tuple[str, ...]
    baseline: int
    deployment_choice: int
    fold_choices: tuple[int, ...]
    fold_inner_metrics: tuple[tuple[JointMetrics, ...], ...]
    candidate_metrics: tuple[JointMetrics, ...]
    play_folds: np.ndarray
    candidate_catch: np.ndarray
    candidate_epa: np.ndarray
    baseline_metrics: JointMetrics
    selected_metrics: JointMetrics
    baseline_catch: np.ndarray
    baseline_epa: np.ndarray
    selected_catch: np.ndarray
    selected_epa: np.ndarray
    catch_auc_gain_ci: tuple[float, float]
    catch_brier_gain_ci: tuple[float, float]
    epa_rmse_gain_ci: tuple[float, float]


def _auc_groups(scores):
    order = np.argsort(scores, kind="stable")
    if not len(order):
        return order, order.copy()
    groups = np.cumsum(np.r_[False, np.diff(scores[order]) != 0]).astype(int)
    return order, groups


def _weighted_auc(caught, order, groups, weights):
    positive = np.bincount(groups, weights=(caught * weights)[order])
    negative = np.bincount(groups, weights=((1 - caught) * weights)[order])
    pairs = positive.sum() * negative.sum()
    if pairs == 0:
        return float("nan")
    return float(positive @ (np.cumsum(negative) - 0.5 * negative) / pairs)


def joint_metrics(caught, catch_probability, epa, epa_prediction):
    """Score probability calibration, catch discrimination, and all-play EPA error."""
    caught = np.asarray(caught, dtype=float)
    catch_probability = np.asarray(catch_probability, dtype=float)
    epa, epa_prediction = np.asarray(epa, dtype=float), np.asarray(epa_prediction, dtype=float)
    auc = _weighted_auc(caught, *_auc_groups(catch_probability), np.ones(len(caught)))
    p = np.clip(catch_probability, 1e-12, 1 - 1e-12)
    residual = epa - epa_prediction
    mse = float((residual ** 2).mean())
    variance = float(epa.var())
    return JointMetrics(
        auc, float(((caught - catch_probability) ** 2).mean()) if len(caught) else float("nan"),
        float(-(caught * np.log(p) + (1 - caught) * np.log1p(-p)).mean()) if len(caught) else float("nan"),
        float(np.sqrt(mse)), float(np.abs(residual).mean()),
        1 - mse / variance if variance > 0 else float("nan"),
    )


def choose_candidate(metrics, baseline=0):
    """Require gains in both outcomes and cap the increase in catch log loss."""
    reference = metrics[baseline]
    if not np.isfinite(reference.catch_auc) or reference.epa_rmse <= 0:
        return baseline
    best, utility = baseline, 0.0
    for i, result in enumerate(metrics):
        auc_gain = result.catch_auc - reference.catch_auc
        rmse_gain = 1 - result.epa_rmse / reference.epa_rmse
        if (np.isfinite(auc_gain) and min(auc_gain, rmse_gain) > utility
                and result.catch_log_loss <= reference.catch_log_loss + 0.005):
            best, utility = i, min(auc_gain, rmse_gain)
    return best


def _interval(values):
    finite = values[np.isfinite(values)]
    return tuple(map(float, np.quantile(finite, [0.025, 0.975]))) if len(finite) else (np.nan, np.nan)


def paired_game_intervals(epa, caught, play_games, target_play_rows, baseline_epa, selected_epa,
                          baseline_catch, selected_catch, *, n_bootstrap=2000, seed=2021):
    """Return AUC, Brier, and RMSE gain intervals with fixed OOF predictions."""
    _, play_groups = np.unique(play_games, return_inverse=True)
    target_groups = play_groups[target_play_rows]
    n_games = int(play_groups.max()) + 1
    auc_base, auc_new = _auc_groups(baseline_catch), _auc_groups(selected_catch)
    catch_error_gain = (caught - baseline_catch) ** 2 - (caught - selected_catch) ** 2
    game_totals = np.column_stack([np.bincount(groups, weights=weights, minlength=n_games)
                                  for groups, weights in (
                                      (play_groups, np.ones(len(epa))),
                                      (play_groups, (epa - baseline_epa) ** 2),
                                      (play_groups, (epa - selected_epa) ** 2),
                                      (target_groups, np.ones(len(caught))),
                                      (target_groups, catch_error_gain),
                                  )])
    rng = np.random.default_rng(seed)
    gains = np.empty((n_bootstrap, 3))
    for i in range(n_bootstrap):
        draw = rng.integers(n_games, size=n_games)
        multiplicity = np.bincount(draw, minlength=n_games)
        n_play, base_sq, new_sq, n_target, brier_gain = multiplicity @ game_totals
        weights = multiplicity[target_groups]
        gains[i] = (
            _weighted_auc(caught, *auc_new, weights) - _weighted_auc(caught, *auc_base, weights),
            brier_gain / n_target if n_target else np.nan,
            np.sqrt(base_sq / n_play) - np.sqrt(new_sq / n_play),
        )
    return tuple(_interval(gain) for gain in gains.T)


def _collect_predictions(play_games, target_play_rows, train_pool, fold_ids, n_candidates, predict):
    n_plays, n_targets = len(play_games), len(target_play_rows)
    catch_oof = np.full((n_targets, n_candidates), np.nan)
    epa_oof = np.full((n_plays, n_candidates), np.nan)
    for fold in np.unique(fold_ids[train_pool]):
        train = train_pool & (fold_ids != fold)
        test = train_pool & (fold_ids == fold)
        test_plays = np.flatnonzero(test)
        test_targets = np.flatnonzero(test[target_play_rows])
        catch_pred, epa_pred = predict(np.flatnonzero(train), test_plays,
                                      np.flatnonzero(train[target_play_rows]), test_targets)
        catch_pred, epa_pred = np.asarray(catch_pred, dtype=float), np.asarray(epa_pred, dtype=float)
        if catch_pred.shape != (len(test_targets), n_candidates) or epa_pred.shape != (len(test_plays), n_candidates):
            raise ValueError("Prediction callback must align rows to evaluation indices and columns to candidates.")
        if not (np.isfinite(epa_pred).all() and np.isfinite(catch_pred).all()
                and ((0 <= catch_pred) & (catch_pred <= 1)).all()):
            raise ValueError("All candidates must give finite EPA predictions and catch probabilities in [0, 1].")
        catch_oof[test_targets], epa_oof[test_plays] = catch_pred, epa_pred
    return catch_oof, epa_oof


def _column_metrics(caught, catch_predictions, epa, epa_predictions):
    return tuple(joint_metrics(caught, catch_column, epa, epa_column)
                 for catch_column, epa_column in zip(catch_predictions.T, epa_predictions.T))


def nested_game_search(play_games, play_epa, target_play_rows, caught, candidates, predict, *, baseline=0,
                       outer_folds=5, inner_folds=4, seed=2021, n_bootstrap=2000):
    """Evaluate a candidate selection policy with inner and outer whole-game folds.

    `target_play_rows` maps each catch observation to its global play row. `predict`
    receives global (training plays, evaluation plays, training targets, evaluation
    targets) indices and returns catch and EPA prediction matrices. Each column is
    one candidate, each row an evaluation observation. Fit every transformation
    and model from training rows alone. `deployment_choice` is an all-data OOF
    tuning choice. Use `selected_metrics` to report independent policy performance.
    """
    games, epa = np.asarray(play_games), np.asarray(play_epa, dtype=float)
    target_rows = np.asarray(target_play_rows)
    caught = np.asarray(caught, dtype=float)
    candidates = tuple(candidates)
    if (games.ndim != 1 or games.shape != epa.shape or len(games) < 3
            or target_rows.ndim != 1 or target_rows.shape != caught.shape or not len(caught)
            or not np.issubdtype(target_rows.dtype, np.integer)
            or np.any(target_rows < 0) or np.any(target_rows >= len(games))
            or not np.isfinite(epa).all() or not np.isin(caught, (0, 1)).all()
            or not 0 <= baseline < len(candidates) or outer_folds < 2 or inner_folds < 2
            or n_bootstrap < 0):
        raise ValueError("Provide aligned play outcomes, binary target outcomes, and valid fold settings.")
    if len(np.unique(games)) < 3:
        raise ValueError("At least three games are needed for nested validation.")

    outer_ids = game_folds(games, n_folds=outer_folds, seed=seed)
    all_rows = np.ones(len(epa), dtype=bool)
    candidate_catch, candidate_epa = _collect_predictions(games, target_rows, all_rows, outer_ids,
                                                         len(candidates), predict)
    selected_catch, selected_epa = np.full(len(caught), np.nan), np.full(len(epa), np.nan)
    fold_choices, fold_metrics = [], []
    for fold in np.unique(outer_ids):
        pool = outer_ids != fold
        if len(np.unique(games[pool])) < 2:
            raise ValueError("Every outer training split needs at least two games for inner validation.")
        inner_ids = np.full(len(games), -1, dtype=int)
        inner_ids[pool] = game_folds(games[pool], n_folds=inner_folds, seed=seed + int(fold) + 1)
        catch_pred, epa_pred = _collect_predictions(games, target_rows, pool, inner_ids, len(candidates), predict)
        target_pool = pool[target_rows]
        metrics = _column_metrics(caught[target_pool], catch_pred[target_pool], epa[pool], epa_pred[pool])
        choice = choose_candidate(metrics, baseline)
        fold_choices.append(choice)
        fold_metrics.append(metrics)
        selected_catch[~target_pool] = candidate_catch[~target_pool, choice]
        selected_epa[~pool] = candidate_epa[~pool, choice]

    metrics = _column_metrics(caught, candidate_catch, epa, candidate_epa)
    baseline_catch, baseline_epa = candidate_catch[:, baseline], candidate_epa[:, baseline]
    selected_metrics = joint_metrics(caught, selected_catch, epa, selected_epa)
    intervals = paired_game_intervals(epa, caught, games, target_rows, baseline_epa, selected_epa,
                                      baseline_catch, selected_catch, n_bootstrap=n_bootstrap, seed=seed)
    return JointValidation(
        candidates, baseline, choose_candidate(metrics, baseline), tuple(fold_choices), tuple(fold_metrics),
        metrics, outer_ids, candidate_catch, candidate_epa, metrics[baseline], selected_metrics,
        baseline_catch, baseline_epa, selected_catch, selected_epa,
        *intervals,
    )
