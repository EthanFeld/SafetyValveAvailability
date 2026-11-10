"""Train and compare catch and EPA models without sharing games between folds."""

import argparse
from pathlib import Path

import numpy as np

from safety_valve import ROOT, write_csv
from sva_epa_models import game_folds
from sva_joint_features import collect_joint_data
from sva_joint_models import fit_logistic, fit_ridge, regression_metrics
from sva_joint_validation import joint_metrics, nested_game_search, paired_game_intervals


NAMES = ("Revised onset SVA", "Geometry", "Sticks and context", "Joint hurdle stack",
         "Outcome mixture", "Outcome mixture, more shrinkage", "Sticks and pressure-risk mixture",
         "Geometry and pressure-risk mixture", "Context benchmark")
CATCH_RECIPES = ((0, 15), (1, 35), (2, 55), (2, 55), (2, 55), (2, 120), (3, 45), (1, 35), (4, 55))
CHALLENGER = NAMES.index("Geometry and pressure-risk mixture")


class JointSVA:
    def __init__(self, data):
        self.data = data
        self.games = np.array([row["game_id"] for row in data.plays])
        self.epa = np.array([row["epa"] for row in data.plays])
        self.result = np.array([row["result"] for row in data.plays])
        self.route_to_play = data.route_play_rows
        self.target_routes = data.target_route_rows
        self.target_plays = self.route_to_play[self.target_routes]
        self.caught = data.target_caught
        geometry_and_sticks = data.route_geometry.shape[1] + len(data.routes[0]["sticks"])
        self.catch_designs = (
            data.route_geometry[:, :1], data.route_geometry, data.route_sticks, data.route_sticks[:, :geometry_and_sticks],
            np.column_stack((data.route_geometry[:, :1], data.play_context[self.route_to_play])),
        )
        self.play_designs = (
            np.column_stack((data.play_context, data.play_geometry[:, :1])),
            np.column_stack((data.play_context, data.play_geometry)),
            np.column_stack((data.play_context, data.play_geometry, data.play_sticks)),
        )
        distance = np.array([row["rush_distance"] for row in data.plays])
        closing = np.array([row["rush_closing"] for row in data.plays])
        onset = np.array([row["onset_seconds"] for row in data.plays])
        self.risk_design = np.column_stack((self.play_designs[2], np.minimum(distance / np.maximum(closing, .25), 2),
                                            np.maximum(closing, 0) / np.maximum(distance, .5), onset ** 2))

    def catch_head(self, train_targets, design, ridge):
        x = self.catch_designs[design]
        return fit_logistic(x[self.target_routes[train_targets]], self.caught[train_targets], ridge=ridge)

    def hurdle_heads(self, train_targets):
        x = self.data.route_sticks[self.target_routes[train_targets]]
        y = self.epa[self.target_plays[train_targets]]
        caught = self.caught[train_targets].astype(bool)
        completion = self.catch_head(train_targets, 2, 55)
        on_catch = fit_ridge(x[caught], y[caught], ridge=100) if caught.any() else fit_ridge(x, y, ridge=100)
        on_miss = fit_ridge(x[~caught], y[~caught], ridge=100) if (~caught).any() else fit_ridge(x, y, ridge=100)
        return completion, on_catch, on_miss

    def predict_routes(self, heads, route_rows):
        completion, on_catch, on_miss = heads
        x = self.data.route_sticks[route_rows]
        probability = completion.predict(x)
        expected_epa = probability * on_catch.predict(x) + (1 - probability) * on_miss.predict(x)
        return probability, expected_epa

    def outlet_summary(self, heads, play_rows):
        index = np.full(len(self.games), -1, dtype=int)
        index[play_rows] = np.arange(len(play_rows))
        group = index[self.route_to_play]
        use = group >= 0
        probability, epa = self.predict_routes(heads, np.flatnonzero(use))
        group = group[use]
        max_probability = np.zeros(len(play_rows))
        max_epa = np.full(len(play_rows), -np.inf)
        np.maximum.at(max_probability, group, probability)
        np.maximum.at(max_epa, group, epa)
        counts = np.bincount(group, minlength=len(play_rows))
        mean_epa = np.bincount(group, weights=epa, minlength=len(play_rows)) / np.maximum(counts, 1)
        return np.column_stack((max_probability, np.where(counts, max_epa, 0), mean_epa))

    def stack(self, train_plays, eval_plays, train_targets, heads):
        # The play head sees route predictions from models that excluded each training game's outcomes.
        inner_fold = game_folds(self.games[train_plays], n_folds=3)
        global_fold = np.full(len(self.games), -1, dtype=int)
        global_fold[train_plays] = inner_fold
        train_features = np.empty((len(train_plays), 3))
        for fold in np.unique(inner_fold):
            valid = inner_fold == fold
            subtrain = train_targets[global_fold[self.target_plays[train_targets]] != fold]
            train_features[valid] = self.outlet_summary(self.hurdle_heads(subtrain), train_plays[valid])
        x = self.play_designs[2]
        model = fit_ridge(np.column_stack((x[train_plays], train_features)), self.epa[train_plays], ridge=120)
        return model.predict(np.column_stack((x[eval_plays], self.outlet_summary(heads, eval_plays))))

    def mixture(self, train_plays, eval_plays, ridge, design):
        x = design
        results = self.result[train_plays]
        probability = []
        value = []
        # These are training labels only. No realized outcome enters the design matrix.
        for code in np.unique(results):
            flag = results == code
            weight = fit_logistic(x[train_plays], flag, ridge=ridge).predict(x[eval_plays])
            group = train_plays[flag]
            prediction = fit_ridge(x[group], self.epa[group], ridge=120).predict(x[eval_plays])
            probability.append(weight)
            value.append(weight * prediction)
        return np.sum(value, axis=0) / np.sum(probability, axis=0)

    def predict(self, train_plays, eval_plays, train_targets, eval_targets):
        route_rows = self.target_routes[eval_targets]
        catch = {recipe: self.catch_head(train_targets, *recipe).predict(self.catch_designs[recipe[0]][route_rows])
                 for recipe in dict.fromkeys(CATCH_RECIPES)}
        epa = [fit_ridge(x[train_plays], self.epa[train_plays], ridge=120).predict(x[eval_plays])
               for x in self.play_designs]
        heads = self.hurdle_heads(train_targets)
        epa.append(self.stack(train_plays, eval_plays, train_targets, heads))
        epa.append(self.mixture(train_plays, eval_plays, 70, self.play_designs[2]))
        epa.append(self.mixture(train_plays, eval_plays, 200, self.play_designs[2]))
        risk_epa = self.mixture(train_plays, eval_plays, 160, self.risk_design)
        epa.extend((risk_epa, risk_epa))
        x = self.data.play_context
        epa.append(fit_ridge(x[train_plays], self.epa[train_plays], ridge=120).predict(x[eval_plays]))
        return np.column_stack([catch[recipe] for recipe in CATCH_RECIPES]), np.column_stack(epa)

    def route_scorecard(self, evaluation, catch_recipe):
        catch = np.empty(len(self.data.routes))
        epa = np.empty_like(catch)
        direct_epa = np.empty_like(catch)
        target_context = np.empty(len(self.target_routes))
        for fold in np.unique(evaluation.play_folds):
            validation = evaluation.play_folds == fold
            train_targets = np.flatnonzero(~validation[self.target_plays])
            valid_targets = np.flatnonzero(validation[self.target_plays])
            route_rows = np.flatnonzero(validation[self.route_to_play])
            catch[route_rows] = self.catch_head(train_targets, *catch_recipe).predict(
                self.catch_designs[catch_recipe[0]][route_rows]
            )
            _, on_catch, on_miss = self.hurdle_heads(train_targets)
            route_x = self.data.route_sticks[route_rows]
            probability = catch[route_rows]
            epa[route_rows] = probability * on_catch.predict(route_x) + (1 - probability) * on_miss.predict(route_x)
            training_epa = self.epa[self.target_plays[train_targets]]
            direct = fit_ridge(self.data.route_sticks[self.target_routes[train_targets]], training_epa, ridge=100)
            direct_epa[route_rows] = direct.predict(route_x)
            x = self.data.play_context[self.target_plays]
            baseline = fit_ridge(x[train_targets], training_epa, ridge=100)
            target_context[valid_targets] = baseline.predict(x[valid_targets])

        observed = self.epa[self.target_plays]
        base = regression_metrics(observed, target_context)
        p = catch[self.target_routes]
        for name, predictions in (("Hurdle", epa), ("Direct", direct_epa)):
            estimate = predictions[self.target_routes]
            fit = regression_metrics(observed, estimate)
            interval = paired_game_intervals(observed, self.caught, self.games[self.target_plays],
                                             np.arange(len(observed)), target_context, estimate, p, p)[2]
            print(f"Target-conditional {name} EPA RMSE {fit.rmse:.4f} vs context only {base.rmse:.4f}; "
                  f"reduction {base.rmse - fit.rmse:+.4f} (95% CI {interval[0]:+.4f} to {interval[1]:+.4f}).")
        return [dict(game_id=row["game_id"], play_id=row["play_id"], nfl_id=row["nfl_id"], player=row["player"],
                     position=row["position"], onset_sva=row["revised_sva"], projected_depth_yd=row["projected_depth"],
                     projected_margin_to_sticks_yd=row["projected_margin"], catch_probability=catch[i],
                     hurdle_epa_on_comparable_targets=epa[i], direct_epa_on_comparable_targets=direct_epa[i],
                     observed_catch=row["caught"])
                for i, row in enumerate(self.data.routes)]


def report(label, metrics):
    print(f"{label}: catch AUC {metrics.catch_auc:.4f}, log loss {metrics.catch_log_loss:.4f}, "
          f"Brier {metrics.catch_brier:.4f}; team EPA RMSE {metrics.epa_rmse:.4f}, R2 {metrics.epa_r2:+.4f}.")


def choose_heads(evaluation, target_plays):
    catch = np.empty(len(target_plays))
    epa = np.empty(len(evaluation.play_folds))
    catch_choices = {}
    epa_choices = {}
    for fold, metrics in zip(np.unique(evaluation.play_folds), evaluation.fold_inner_metrics):
        baseline = metrics[evaluation.baseline]
        catch_candidates = [i for i, m in enumerate(metrics) if m.catch_auc >= baseline.catch_auc]
        catch_choice = min(catch_candidates, key=lambda i: metrics[i].catch_log_loss)
        epa_choice = min(range(len(metrics)), key=lambda i: metrics[i].epa_rmse)
        catch_choices[fold], epa_choices[fold] = catch_choice, epa_choice
        plays = evaluation.play_folds == fold
        targets = plays[target_plays]
        catch[targets] = evaluation.candidate_catch[targets, catch_choice]
        epa[plays] = evaluation.candidate_epa[plays, epa_choice]
    return catch, epa, catch_choices, epa_choices


def run(data, route_output, play_output):
    model = JointSVA(data)
    print(f"{len(data.plays):,} pressure plays across {len(np.unique(model.games))} games; "
          f"{len(data.routes):,} onset outlets and {len(model.caught):,} named targets "
          f"({int(model.caught.sum()):,} catches).", flush=True)
    evaluation = nested_game_search(model.games, model.epa, model.target_plays, model.caught, NAMES, model.predict)
    for name, metrics in zip(NAMES, evaluation.candidate_metrics):
        report(name, metrics)
    report("Nested joint selection", evaluation.selected_metrics)
    catch, epa, catch_choices, epa_choices = choose_heads(evaluation, model.target_plays)
    report("Nested separate selection", joint_metrics(model.caught, catch, model.epa, epa))
    print("Separate catch choices:", ", ".join(NAMES[i] for i in catch_choices.values()))
    print("Separate EPA choices:", ", ".join(NAMES[i] for i in epa_choices.values()))
    intervals = paired_game_intervals(model.epa, model.caught, model.games, model.target_plays,
                                      evaluation.baseline_epa, epa, evaluation.baseline_catch, catch)
    for label, interval in zip(("AUC gain", "Brier reduction", "EPA RMSE reduction"), intervals):
        print(f"Separate selector {label} 95% paired game-bootstrap interval {interval[0]:+.4f} to {interval[1]:+.4f}.")

    challenger_epa = evaluation.candidate_epa[:, CHALLENGER]
    challenger_catch = evaluation.candidate_catch[:, CHALLENGER]
    intervals = paired_game_intervals(model.epa, model.caught, model.games, model.target_plays,
                                      evaluation.baseline_epa, challenger_epa, evaluation.baseline_catch, challenger_catch)
    for label, interval in zip(("AUC gain", "Brier reduction", "EPA RMSE reduction"), intervals):
        print(f"Fixed challenger {label} 95% paired game-bootstrap interval {interval[0]:+.4f} to {interval[1]:+.4f}.")

    routes = model.route_scorecard(evaluation, CATCH_RECIPES[CHALLENGER])
    write_csv(route_output, tuple(routes[0]), routes)
    plays = [dict(game_id=row["game_id"], play_id=row["play_id"], offense=row["offense"], actual_epa=row["epa"],
                  revised_onset_baseline_epa=evaluation.baseline_epa[i], challenger_epa=challenger_epa[i],
                  nested_selected_epa=epa[i], nested_selected_model=NAMES[epa_choices[evaluation.play_folds[i]]],
                  onset_frame=row["onset_frame"],
                  in_range_outlets=int(data.play_geometry[i, 1])) for i, row in enumerate(data.plays)]
    write_csv(play_output, tuple(plays[0]), plays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "nfl-big-data-bowl-regional-event-data-main.zip")
    parser.add_argument("--epa", type=Path, default=ROOT / "nflverse_2021_tracking_epa.csv")
    parser.add_argument("--hands", type=Path, default=ROOT / "qb_handedness.csv")
    parser.add_argument("--route-output", type=Path, default=ROOT / "sva_joint_outlets.csv")
    parser.add_argument("--play-output", type=Path, default=ROOT / "sva_joint_plays.csv")
    args = parser.parse_args()
    output_paths = {args.route_output.resolve(), args.play_output.resolve()}
    if len(output_paths) < 2 or output_paths.intersection(p.resolve() for p in (args.archive, args.epa, args.hands)):
        parser.error("Use separate files for each output and input.")
    data = collect_joint_data(args.archive, args.epa, args.hands)
    run(data, args.route_output, args.play_output)


if __name__ == "__main__":
    main()
