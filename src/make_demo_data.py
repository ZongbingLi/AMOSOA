"""
Generate synthetic demo data for AMOSOA.

The behavioral data used in the manuscript cannot be shared publicly due
to partner-platform privacy agreements. This script creates a synthetic
dataset with the same file structure so that the pipeline can be tested
end to end:

    data/fx.csv : 31 f(x) predictors + userid + churn_state
    data/gx.csv :  4 g(x) predictors + userid + churn_state

The synthetic labels are generated from a latent severity score that
depends on a subset of the predictors, so the pipeline has learnable
(but artificial) signal. All values are random and contain no real
user information.
"""

import argparse
import os

import numpy as np
import pandas as pd


def main(n_samples=3000, output_dir="data", seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    n_fx, n_gx = 31, 4

    # Latent churn severity in [0, 1] driven by a few predictors
    base = rng.normal(0, 1, size=(n_samples, n_fx))
    short = rng.normal(0, 1, size=(n_samples, n_gx))

    severity = (
        0.9 * base[:, 0]
        + 0.7 * np.tanh(base[:, 1])
        - 0.8 * base[:, 2] ** 2 / 3
        + 0.6 * short[:, 0]
        - 0.5 * np.abs(short[:, 1])
        + rng.normal(0, 0.6, size=n_samples)
    )
    severity = (severity - severity.min()) / (severity.max() - severity.min())

    # Ordinal labels with approximate class proportions 14% / 29% / 57%
    b12, b23 = np.quantile(severity, [0.43, 0.86])
    y = np.ones(n_samples, dtype=int)
    y[severity >= b12] = 2
    y[severity >= b23] = 3

    userid = np.arange(1, n_samples + 1)

    fx_df = pd.DataFrame(base, columns=[f"feature_f{i:02d}" for i in range(1, n_fx + 1)])
    fx_df.insert(0, "userid", userid)
    fx_df["churn_state"] = y

    gx_df = pd.DataFrame(short, columns=[f"feature_g{i:02d}" for i in range(1, n_gx + 1)])
    gx_df.insert(0, "userid", userid)
    gx_df["churn_state"] = y

    fx_path = os.path.join(output_dir, "fx.csv")
    gx_path = os.path.join(output_dir, "gx.csv")
    fx_df.to_csv(fx_path, index=False)
    gx_df.to_csv(gx_path, index=False)

    print(f"Synthetic demo data written to {output_dir}/")
    print(f"  {fx_path}: {fx_df.shape}")
    print(f"  {gx_path}: {gx_df.shape}")
    print(f"  Class distribution: {np.bincount(y)[1:]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo data.")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_samples, args.output_dir, args.seed)
