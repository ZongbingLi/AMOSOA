"""
Model-development configuration selection for AMOSOA.

Implements the hyperparameter search described in Appendix C (Step 6) of
the manuscript. Each candidate configuration

    h = (n_estimators, learning_rate, max_depth, alpha, sigma, c_max)

is evaluated by fitting the full AMOSOA pipeline on the training set and
selecting the objective-weight vector and transition thresholds on the
validation set (Steps 2-5). The configuration, weight vector, and
threshold pair that maximize the optimized validation objective are
retained:

    (h*, lambda*, b*) = argmax_{h in H, lambda in G_lambda}
                            T_val(b~(h, lambda); h, lambda)

Candidate grids (as reported in the manuscript)
-----------------------------------------------
LightGBM:  n_estimators in {300, 500, 700}
           learning_rate in {0.03, 0.05, 0.10}
           max_depth    in {4, 6, 8}
Ridge:     alpha         in {1, 10, 100}
Weighting: sigma         in {0.03, 0.05, 0.10}
Bounding:  c_max         in {0.10, 0.15, 0.20}

Note: this script evaluates 729 configurations and is computationally
intensive; it was run once during model development. For day-to-day use,
train_amosoa.py runs the pipeline directly with the selected
configuration reported in the manuscript.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from amosoa import AMOSOA, DataLoader, RANDOM_STATE

GRID = {
    "fx_n_estimators": [300, 500, 700],
    "fx_learning_rate": [0.03, 0.05, 0.10],
    "fx_max_depth": [4, 6, 8],
    "gx_ridge_alpha": [1.0, 10.0, 100.0],
    "sigma": [0.03, 0.05, 0.10],
    "c_max": [0.10, 0.15, 0.20],
}


def main(fx_path, gx_path, target_col, exclude_cols, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(fx_path, gx_path, target_col=target_col,
                        exclude_cols=exclude_cols)
    loader.load_data()
    X_fx, X_gx, y = loader.X_fx, loader.X_gx, loader.y
    q = (y - 1) / 2.0

    # Shared stratified 70/15/15 partition (fixed seed), identical for all
    # candidate configurations and all comparison models.
    (X_fx_train, X_fx_temp, X_gx_train, X_gx_temp,
     y_train, y_temp, q_train, q_temp) = train_test_split(
        X_fx, X_gx, y, q, test_size=0.30, stratify=y, random_state=seed)
    (X_fx_val, X_fx_test, X_gx_val, X_gx_test,
     y_val, y_test, q_val, q_test) = train_test_split(
        X_fx_temp, X_gx_temp, y_temp, q_temp,
        test_size=0.50, stratify=y_temp, random_state=seed)

    keys = list(GRID.keys())
    configs = [dict(zip(keys, values))
               for values in itertools.product(*(GRID[k] for k in keys))]
    print(f"Evaluating {len(configs)} candidate configurations...")

    records = []
    best_objective = -np.inf
    best_config = None
    best_model = None

    for idx, cfg in enumerate(configs, start=1):
        model = AMOSOA(**cfg, random_state=seed, verbose=False)
        model.fit_with_data(
            X_fx_train, X_gx_train, y_train, q_train,
            X_fx_val, X_gx_val, y_val, q_val,
            X_fx_test, X_gx_test, y_test, q_test,
        )

        # Optimized validation objective T_val(b~(h, lambda); h, lambda)
        objective, _ = model._compute_objective(
            model.z_val_, y_val, model.b_opt_[0], model.b_opt_[1],
            model.lambda_opt_)

        records.append({
            **cfg,
            "lambda_1": model.lambda_opt_[0],
            "lambda_2": model.lambda_opt_[1],
            "lambda_3": model.lambda_opt_[2],
            "b12": model.b_opt_[0],
            "b23": model.b_opt_[1],
            "objective": objective,
        })

        # Strict improvement only: ties retain the first configuration in
        # the prespecified search order (Appendix C, Step 6).
        if objective > best_objective + 1e-10:
            best_objective = objective
            best_config = cfg
            best_model = model

        if idx % 25 == 0 or idx == len(configs):
            print(f"  [{idx}/{len(configs)}] best objective so far: "
                  f"{best_objective:.4f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(output_dir, "model_selection_results.csv"),
                      index=False)

    # Temperature calibration for the selected configuration (validation NLL)
    tau = best_model._calibrate_temperature(
        best_model.z_val_, y_val, best_model.b_opt_[0], best_model.b_opt_[1])

    print("\nSelected configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"  lambda*: ({best_model.lambda_opt_[0]:.3f}, "
          f"{best_model.lambda_opt_[1]:.3f}, {best_model.lambda_opt_[2]:.3f})")
    print(f"  b* = ({best_model.b_opt_[0]:.4f}, {best_model.b_opt_[1]:.4f})")
    print(f"  tau* = {tau:.2f}")
    print(f"\nFull search results saved to "
          f"{os.path.join(output_dir, 'model_selection_results.csv')}")

    return best_config, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AMOSOA model-development configuration selection.")
    parser.add_argument("--fx-path", default=os.path.join("data", "fx.csv"))
    parser.add_argument("--gx-path", default=os.path.join("data", "gx.csv"))
    parser.add_argument("--target-col", default="churn_state")
    parser.add_argument("--exclude-cols", nargs="*", default=["userid"])
    parser.add_argument("--output-dir", default="output_model_selection")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    main(args.fx_path, args.gx_path, args.target_col, args.exclude_cols,
         args.output_dir, args.seed)
