"""
Main experiment entry point for AMOSOA.

Runs the full pipeline with the selected model-development configuration
reported in the manuscript (Appendix C, Step 6):
    n_estimators = 500, learning_rate = 0.05, max_depth = 6,
    ridge alpha = 10, sigma = 0.05, c_max = 0.15.

Usage
-----
python src/train_amosoa.py --fx-path data/fx.csv --gx-path data/gx.csv

Input
-----
Two CSV files sharing the same rows and the same ordinal target column
(values in {1, 2, 3}). f(x) table: predictors of the global latent-risk
function. g(x) table: short-term predictors of the correction function.

Output
------
Metrics, confusion matrices, predictions, feature importances, and the
selected parameters, written to --output-dir (default: output/).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from amosoa import AMOSOA, DataLoader, ModelEvaluator, RANDOM_STATE

# Selected model-development configuration reported in the manuscript
SELECTED_CONFIG = dict(
    fx_n_estimators=500,
    fx_learning_rate=0.05,
    fx_max_depth=6,
    fx_subsample=0.8,
    fx_colsample_bytree=0.8,
    gx_ridge_alpha=10.0,
    sigma=0.05,
    c_max=0.15,
    temperature=0.05,  # initial value; calibrated on the validation set
    min_weight=0.20,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate AMOSOA (ordinal three-class churn prediction)."
    )
    parser.add_argument("--fx-path", default=os.path.join("data", "fx.csv"),
                        help="CSV file with f(x) predictors and the target column.")
    parser.add_argument("--gx-path", default=os.path.join("data", "gx.csv"),
                        help="CSV file with g(x) predictors and the target column.")
    parser.add_argument("--target-col", default="churn_state",
                        help="Name of the ordinal target column (values in {1, 2, 3}).")
    parser.add_argument("--exclude-cols", nargs="*", default=["userid"],
                        help="Columns to exclude from the feature sets.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for all output files.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Random seed (default: 42, as in the manuscript).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AMOSOA: Adaptive Multi-Objective Score Optimization Algorithm")
    print("=" * 60)
    print(f"f(x) table: {args.fx_path}")
    print(f"g(x) table: {args.gx_path}")
    print(f"Output directory: {args.output_dir}")

    # ---- Load data ----
    loader = DataLoader(args.fx_path, args.gx_path,
                        target_col=args.target_col,
                        exclude_cols=args.exclude_cols)
    loader.load_data()

    X_fx, X_gx, y, q = loader.X_fx, loader.X_gx, loader.y, loader.q
    feature_names = loader.get_feature_names()

    # ---- Train AMOSOA with the selected configuration ----
    model = AMOSOA(
        **SELECTED_CONFIG,
        random_state=args.seed,
        verbose=True,
        output_dir=args.output_dir,
    )
    model.fit(X_fx, X_gx, y, q=q, val_split=0.15, test_split=0.15)

    # ---- Evaluate on the held-out test set (and train/validation for reporting) ----
    evaluator = ModelEvaluator(model, output_dir=args.output_dir)

    test_results = evaluator.evaluate(
        model.X_fx_test, model.X_gx_test, model.y_test,
        dataset_name="Test", verbose=True)
    val_results = evaluator.evaluate(
        model.X_fx_val, model.X_gx_val, model.y_val,
        dataset_name="Validation", verbose=False)
    train_results = evaluator.evaluate(
        model.X_fx_train, model.X_gx_train, model.y_train,
        dataset_name="Training", verbose=False)

    evaluator.save_results(test_results, "Test")
    evaluator.save_results(val_results, "Validation")
    evaluator.save_results(train_results, "Training")

    # ---- Save the selected parameters ----
    params = {
        "fx_n_estimators": model.fx_n_estimators,
        "fx_learning_rate": model.fx_learning_rate,
        "fx_max_depth": model.fx_max_depth,
        "fx_subsample": model.fx_subsample,
        "fx_colsample_bytree": model.fx_colsample_bytree,
        "gx_ridge_alpha": model.gx_ridge_alpha,
        "sigma": model.sigma,
        "c_max": model.c_max,
        "min_weight": model.min_weight,
        "lambda_opt": list(model.lambda_opt_) if model.lambda_opt_ is not None else None,
        "b12_opt": model.b_opt_[0] if model.b_opt_ else None,
        "b23_opt": model.b_opt_[1] if model.b_opt_ else None,
        "tau_opt": model.tau_opt_,
        "training_time": model.training_time_,
        "oof_time": model.oof_time_,
        "optimization_time": model.optimization_time_,
        "random_state": args.seed,
    }
    with open(os.path.join(args.output_dir, "model_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Model parameters saved to "
          f"{os.path.join(args.output_dir, 'model_params.json')}")

    # ---- Feature importance ----
    importance = model.get_feature_importance(feature_names["fx"], feature_names["gx"])

    if importance.get("fx") is not None and "names" in importance.get("fx", {}):
        print("\n  f(x) feature importance (gain-based), top 10:")
        for name, val in zip(importance["fx"]["names"][:10],
                             importance["fx"]["values"][:10]):
            print(f"    {name}: {val:.4f}")
        pd.DataFrame({
            "Feature": importance["fx"]["names"],
            "Importance": importance["fx"]["values"],
        }).to_csv(os.path.join(args.output_dir, "fx_feature_importance.csv"),
                  index=False)

    if importance.get("gx") is not None and "names" in importance.get("gx", {}):
        print("\n  g(x) standardized Ridge coefficients:")
        for name, coef, imp in zip(importance["gx"]["names"],
                                   importance["gx"]["coefficients"],
                                   importance["gx"]["importance"]):
            print(f"    {name}: coef={coef:+.4f}, |coef|={imp:.4f}")
        pd.DataFrame({
            "Feature": importance["gx"]["names"],
            "Coefficient": importance["gx"]["coefficients"],
            "Importance": importance["gx"]["importance"],
        }).to_csv(os.path.join(args.output_dir, "gx_feature_importance.csv"),
                  index=False)

    # ---- Performance summary across partitions ----
    summary = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Test"],
        "Samples": [len(model.y_train), len(model.y_val), len(model.y_test)],
        "Balanced Accuracy": [train_results["balanced_accuracy"],
                              val_results["balanced_accuracy"],
                              test_results["balanced_accuracy"]],
        "Kappa (quadratic)": [train_results["kappa_quadratic"],
                              val_results["kappa_quadratic"],
                              test_results["kappa_quadratic"]],
        "Kendall's tau": [train_results["kendall_tau"],
                          val_results["kendall_tau"],
                          test_results["kendall_tau"]],
        "NLL": [train_results["nll"], val_results["nll"], test_results["nll"]],
        "Brier Score": [train_results["brier_score"],
                        val_results["brier_score"],
                        test_results["brier_score"]],
        "Recall Class 1": [train_results["class_1_recall"],
                           val_results["class_1_recall"],
                           test_results["class_1_recall"]],
        "Recall Class 2": [train_results["class_2_recall"],
                           val_results["class_2_recall"],
                           test_results["class_2_recall"]],
        "Recall Class 3": [train_results["class_3_recall"],
                           val_results["class_3_recall"],
                           test_results["class_3_recall"]],
        "Severe Error Rate": [train_results["severe_error_rate"],
                              val_results["severe_error_rate"],
                              test_results["severe_error_rate"]],
    })
    summary.to_csv(os.path.join(args.output_dir, "performance_summary.csv"),
                   index=False)
    print("\nPerformance summary:")
    print(summary.round(4))

    # ---- Plots ----
    evaluator.plot_confusion_matrix(test_results, "Test")
    if len(model.get_optimization_history()) > 0:
        evaluator.plot_threshold_optimization()

    print("\n" + "=" * 60)
    print("AMOSOA analysis completed.")
    print("=" * 60)
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"\nSelected parameters:")
    print(f"  lambda* = ({model.lambda_opt_[0]:.3f}, "
          f"{model.lambda_opt_[1]:.3f}, {model.lambda_opt_[2]:.3f})")
    print(f"  b* = (b12={model.b_opt_[0]:.4f}, b23={model.b_opt_[1]:.4f})")
    print(f"  tau* = {model.tau_opt_:.2f}")

    return model, test_results, summary


if __name__ == "__main__":
    main()
