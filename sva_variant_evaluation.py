"""Select SVA variants with whole-game splits and a separate final test set."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VariantEvaluation:
    names: tuple[str, ...]
    baseline: int
    winner: int
    development_n: int
    test_n: int
    development_games: np.ndarray
    test_games: np.ndarray
    development_raw: np.ndarray
    development_adjusted: np.ndarray
    test_raw: np.ndarray
    test_adjusted: np.ndarray
    test_auc: np.ndarray
    test_raw_gain_ci: tuple[float, float]
    test_adjusted_gain_ci: tuple[float, float]


def _split_games(games, weeks, holdout_weeks, rng):
    game_ids, first, game_index = np.unique(games, return_index=True, return_inverse=True)
    if weeks is None:
        game_weeks = np.zeros(len(game_ids), dtype=int)
    else:
        game_weeks = weeks[first]
        if not np.array_equal(game_weeks[game_index], weeks):
            raise ValueError("Each game must belong to a single week.")

    test = np.zeros(len(game_ids), dtype=bool)
    development_groups = []
    for week in np.unique(game_weeks):
        group = rng.permutation(np.flatnonzero(game_weeks == week))
        if weeks is None:
            n_test = round(len(group) / 4)
            test[group[:n_test]] = True
            development_groups.append(group[n_test:])
        elif week in holdout_weeks:
            test[group] = True
        else:
            development_groups.append(group)

    n_folds = min(5, int((~test).sum()))
    if test.sum() < 2 or n_folds < 2:
        raise ValueError("At least two games are required in both the development and test sets.")

    # Rotate the starting fold each week to balance the total number of games per fold.
    development_games = np.concatenate(development_groups)
    game_folds = np.full(len(game_ids), -1, dtype=int)
    game_folds[development_games] = np.arange(len(development_games)) % n_folds
    return game_folds[game_index], game_ids[~test], game_ids[test]


def _pearson(scores, caught):
    scores = scores - scores.mean(axis=0)
    caught = caught - caught.mean()
    scale = np.sqrt((scores ** 2).sum(axis=0) * (caught @ caught))
    return np.divide(caught @ scores, scale, out=np.full(scores.shape[1], np.nan), where=scale > 1e-8)


def _auc(scores, caught):
    positive = scores[caught == 1]
    negative = scores[caught == 0]
    if not len(positive) or not len(negative):
        return np.full(scores.shape[1], np.nan)
    difference = positive[:, None, :] - negative[None, :, :]
    return (difference > 0).mean(axis=(0, 1)) + 0.5 * (difference == 0).mean(axis=(0, 1))


def _gain_intervals(raw, adjusted, games, rng, n_bootstrap):
    _, game_index = np.unique(games, return_inverse=True)
    groups = [np.flatnonzero(game_index == i) for i in range(game_index.max() + 1)]
    gains = np.empty((n_bootstrap, 2))
    for i in range(n_bootstrap):
        sample = rng.integers(len(groups), size=len(groups))
        rows = np.concatenate([groups[group] for group in sample])
        for j, data in enumerate((raw, adjusted)):
            baseline_r, winner_r = _pearson(data[rows, :2], data[rows, 2])
            gains[i, j] = winner_r - baseline_r

    intervals = []
    for gain in gains.T:
        valid = gain[np.isfinite(gain)]
        intervals.append(tuple(np.quantile(valid, [0.025, 0.975])) if len(valid) else (np.nan, np.nan))
    return tuple(intervals)


def evaluate_variants(scores, caught, games, controls, names, *, weeks=None, holdout_weeks=(7, 8),
                      baseline=0, seed=2021,
                      n_bootstrap=2000):
    """Select by pooled out-of-fold partial correlation on development games.

    With week labels, reserve weeks 7 and 8 for the final test by default.
    Without them, reserve a seeded 25% of games. For each fold, adjustments are fitted on other
    development games and applied to the held-out scores and catch indicators.
    For the final test they are fitted on all development games. Only the selected
    variant and baseline are evaluated on test games. Paired bootstrap intervals
    resample whole test games and condition on the fitted control adjustments.
    """
    scores, caught, games = np.asarray(scores, dtype=float), np.asarray(caught, dtype=float), np.asarray(games)
    controls = np.asarray(controls, dtype=float) if controls is not None else np.empty((len(caught), 0))
    names = tuple(names)
    if (caught.ndim != 1 or games.shape != caught.shape or scores.shape != (len(caught), len(names))
            or controls.ndim != 2 or len(controls) != len(caught)):
        raise ValueError("Scores, catch outcomes, game IDs and controls must align by target.")
    if not (0 <= baseline < len(names) and n_bootstrap >= 1 and len(caught)
            and np.isfinite(scores).all() and np.isfinite(controls).all() and np.isin(caught, (0, 1)).all()):
        raise ValueError("Evaluation requires finite scores and controls, binary outcomes and a valid baseline.")
    if weeks is not None:
        weeks = np.asarray(weeks)
        if weeks.shape != caught.shape:
            raise ValueError("Week labels must align by target.")

    rng = np.random.default_rng(seed)
    fold, development_games, test_games = _split_games(games, weeks, holdout_weeks, rng)
    development = fold >= 0
    test = ~development
    design = np.column_stack((np.ones(len(caught)), controls))
    development_outcomes = np.column_stack((scores[development], caught[development]))
    development_design = design[development]
    development_fold = fold[development]
    residuals = np.empty_like(development_outcomes)
    for i in np.unique(development_fold):
        valid = development_fold == i
        coefficients = np.linalg.lstsq(development_design[~valid], development_outcomes[~valid], rcond=None)[0]
        residuals[valid] = development_outcomes[valid] - development_design[valid] @ coefficients

    development_raw = _pearson(development_outcomes[:, :-1], development_outcomes[:, -1])
    development_adjusted = _pearson(residuals[:, :-1], residuals[:, -1])
    if not np.isfinite(development_adjusted).any():
        raise ValueError("Development data have insufficient variation to compare variants.")
    winner = int(np.nanargmax(development_adjusted))
    if np.isclose(development_adjusted[baseline], development_adjusted[winner], atol=1e-12, rtol=0):
        winner = baseline

    comparison = [baseline, winner]
    selected_development = development_outcomes[:, [*comparison, -1]]
    selected_test = np.column_stack((scores[test][:, comparison], caught[test]))
    coefficients = np.linalg.lstsq(development_design, selected_development, rcond=None)[0]
    test_residuals = selected_test - design[test] @ coefficients
    test_raw, test_adjusted = np.full(len(names), np.nan), np.full(len(names), np.nan)
    test_auc = np.full(len(names), np.nan)
    test_raw[comparison] = _pearson(selected_test[:, :2], selected_test[:, 2])
    test_adjusted[comparison] = _pearson(test_residuals[:, :2], test_residuals[:, 2])
    test_auc[comparison] = _auc(selected_test[:, :2], selected_test[:, 2])
    raw_interval, adjusted_interval = _gain_intervals(selected_test, test_residuals, games[test], rng, n_bootstrap)

    return VariantEvaluation(
        names, baseline, winner, int(development.sum()), int(test.sum()), development_games, test_games,
        development_raw, development_adjusted, test_raw, test_adjusted, test_auc, raw_interval, adjusted_interval,
    )
