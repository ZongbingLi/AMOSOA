# AMOSOA: Adaptive Multi-Objective Score Optimization Algorithm

Official implementation of **AMOSOA**, an ordinal three-class churn
prediction method for online mutual-help community (OMHC) users
(1 = Active, 2 = At-risk, 3 = Behaviorally Churned).

AMOSOA combines:

1. a **global latent-risk function** f(x) (LightGBM regression on the
   ordinal-encoded target q in {0, 0.5, 1}, with five-fold stratified
   out-of-fold predictions);
2. a **bounded, boundary-weighted sample-adaptive correction** g(x)
   (Ridge regression on OOF residuals with Gaussian boundary-proximity
   weights, bounded by c_max);
3. **soft ordinal decoding** with temperature-calibrated class
   probabilities; and
4. **multi-objective threshold optimization** over quadratic weighted
   kappa, Kendall's tau-b, and balanced accuracy with constrained
   objective weights.

## Repository structure

```
AMOSOA/
├── src/
│   ├── amosoa.py               # Core model, data loader, and evaluator
│   ├── train_amosoa.py         # Main experiment (selected configuration)
│   ├── run_model_selection.py  # Hyperparameter search over the grid H
│   └── make_demo_data.py       # Synthetic demo data generator
├── data/                       # Demo data (generated; see below)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The manuscript experiments used Python 3.13.5 with LightGBM 3.3.5,
scikit-learn 1.6.1, and NumPy 2.4.1 on a CPU-only workstation
(Intel Core i7-9750H, 32 GB RAM, Windows 10).

## Quick start (synthetic demo data)

The behavioral data analyzed in the manuscript cannot be shared publicly
due to partner-platform privacy agreements. To test the pipeline end to
end, generate a synthetic dataset with the same file structure:

```bash
python src/make_demo_data.py
python src/train_amosoa.py
```

Results (metrics, confusion matrices, predictions, feature importances,
and the selected parameters) are written to `output/`.

## Using your own data

Provide two CSV files that share the same rows and the same ordinal
target column (values in {1, 2, 3}): one with the f(x) predictors and
one with the g(x) predictors. Then run:

```bash
python src/train_amosoa.py --fx-path path/to/fx.csv --gx-path path/to/gx.csv \
    --target-col churn_state --exclude-cols userid --output-dir output
```

## Reproducing the manuscript experiments

- `src/train_amosoa.py` runs the full pipeline with the selected
  model-development configuration reported in the manuscript
  (n_estimators = 500, learning rate = 0.05, max_depth = 6,
  ridge alpha = 10, sigma = 0.05, c_max = 0.15; seed = 42).
- `src/run_model_selection.py` reproduces the hyperparameter search over
  the candidate grid H described in Appendix C (729 configurations;
  computationally intensive).

## License

MIT License (see LICENSE).
