"""
AMOSOA: Adaptive Multi-Objective Score Optimization Algorithm
=============================================================

Core implementation of AMOSOA for ordinal three-class churn prediction
(1 = Active, 2 = At-risk, 3 = Behaviorally Churned), as described in the
manuscript (Section 4.2.2 and Appendix C).

Pipeline overview
-----------------
Step 1. Stratified 70/15/15 train/validation/test partition (seed = 42);
        ordinal labels are mapped to q_i = (y_i - 1) / 2 in {0, 0.5, 1}.
Step 2. Global latent-risk function f(x): LightGBM regressor with a
        squared-error objective fitted on q; five-fold stratified
        out-of-fold (OOF) predictions on the training set, clipped to
        [0, 1]; the selected configuration is refitted on the complete
        training set for validation/test scoring.
Step 3. Sample-adaptive correction g(x): boundary-weighted Ridge
        regression on OOF residuals r_i = q_i - f_oof_i, with Gaussian
        boundary-proximity weights (normalized to mean 1) around two
        preliminary transition points; the fitted correction is bounded
        in [-c_max, c_max]; composite score z = clip(f + g_bounded, 0, 1).
Step 4. Soft ordinal decoding: logits l1 = (b12 - z)/tau, l2 = 0,
        l3 = (z - b23)/tau; numerically stable softmax; the hard
        prediction is equivalent to thresholding z (an observation
        exactly at a threshold is assigned to the higher-risk class).
Step 5. Validation objective T_val = w1*kappa_Q + w2*tau_b + w3*BA with
        kappa_Q and tau_b rescaled to [0, 1]; transition thresholds are
        optimized by coordinate search over {0.01, ..., 0.99} starting
        from (1/3, 2/3); objective weights are searched over
        {0.20, 0.22, ..., 0.60} subject to sum = 1 and w_j >= 0.20.
Step 6. Temperature calibration by minimizing validation negative
        log-likelihood over {0.01, 0.05, 0.10, 0.20}; final refit
        protocol on the complete training set.
Step 7. Evaluation: quadratic weighted kappa, Kendall's tau-b, balanced
        accuracy, severe error rate, NLL, and multiclass Brier score.
"""

import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy.special import logsumexp
from scipy.stats import kendalltau, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EPS = 1e-8


# ---------------------------------------------------------------------------
# Part 1: Data loading
# ---------------------------------------------------------------------------

class DataLoader:
    """Load the feature tables for f(x) and g(x) and the ordinal target.

    Parameters
    ----------
    fx_path : str
        CSV file with the predictors of the global latent-risk function.
    gx_path : str
        CSV file with the short-term predictors of the correction function.
    target_col : str
        Name of the ordinal target column (values in {1, 2, 3}).
    exclude_cols : list of str, optional
        Identifier or one-hot columns to drop from the feature sets.
    """

    def __init__(self, fx_path, gx_path, target_col="churn_state", exclude_cols=None):
        self.fx_path = fx_path
        self.gx_path = gx_path
        self.target_col = target_col
        self.exclude_cols = list(exclude_cols) if exclude_cols else ["userid"]

        self.X_fx = None
        self.X_gx = None
        self.y = None
        self.q = None  # q_i = (y_i - 1) / 2
        self.feature_names_fx = None
        self.feature_names_gx = None

    def load_data(self):
        """Load both tables and extract the target variable."""
        df_fx = pd.read_csv(self.fx_path)
        self.feature_names_fx = [
            col for col in df_fx.columns
            if col != self.target_col and col not in self.exclude_cols
        ]

        df_gx = pd.read_csv(self.gx_path)
        self.feature_names_gx = [
            col for col in df_gx.columns
            if col != self.target_col and col not in self.exclude_cols
        ]

        y_fx = df_fx[self.target_col].values
        y_gx = df_gx[self.target_col].values
        if not np.array_equal(y_fx, y_gx):
            print("Warning: target columns in the two files differ; "
                  "using the target from the f(x) table.")

        self.y = y_fx.astype(int)
        # Map ordinal labels to the unit interval: {1, 2, 3} -> {0, 0.5, 1}
        self.q = (self.y - 1) / 2.0

        self.X_fx = df_fx[self.feature_names_fx].values
        self.X_gx = df_gx[self.feature_names_gx].values

        print("Data loaded successfully:")
        print(f"  - f(x) features: {self.X_fx.shape[1]}, samples: {self.X_fx.shape[0]}")
        print(f"  - g(x) features: {self.X_gx.shape[1]}, samples: {self.X_gx.shape[0]}")
        print(f"  - Target distribution: {np.bincount(self.y)[1:]}")
        print(f"  - q values: {np.unique(self.q)}")

        return self

    def get_feature_names(self):
        return {"fx": self.feature_names_fx, "gx": self.feature_names_gx}


# ---------------------------------------------------------------------------
# Part 2: AMOSOA core model
# ---------------------------------------------------------------------------

class AMOSOA:
    """AMOSOA model, following Section 4.2.2 / Appendix C of the manuscript.

    Parameters
    ----------
    fx_n_estimators, fx_learning_rate, fx_max_depth : LightGBM settings for f(x).
    fx_subsample, fx_colsample_bytree : LightGBM sampling settings.
    gx_ridge_alpha : float
        Ridge regularization parameter alpha of the correction function.
    sigma : float
        Bandwidth of the Gaussian boundary-proximity weights.
    c_max : float
        Maximum magnitude of the sample-level correction.
    temperature : float
        Initial soft-decoding temperature; overwritten by calibration.
    min_weight : float
        Lower bound of each objective weight (kept for interface clarity;
        the candidate grid is fixed to {0.20, 0.22, ..., 0.60}).
    random_state : int
        Random seed used for all stochastic procedures.
    verbose : bool
        Print progress information.
    output_dir : str, optional
        Directory for saved artifacts.
    """

    def __init__(
        self,
        # LightGBM parameters for f(x)
        fx_n_estimators=500,
        fx_learning_rate=0.05,
        fx_max_depth=6,
        fx_subsample=0.8,
        fx_colsample_bytree=0.8,
        # Ridge parameter for g(x)
        gx_ridge_alpha=10.0,
        # Boundary weighting and correction bounding
        sigma=0.05,
        c_max=0.15,
        # Soft-decoding temperature (initial value; calibrated later)
        temperature=0.05,
        # Constrained objective weights
        min_weight=0.20,
        # Optimization
        random_state=RANDOM_STATE,
        verbose=True,
        output_dir=None,
    ):
        self.fx_n_estimators = fx_n_estimators
        self.fx_learning_rate = fx_learning_rate
        self.fx_max_depth = fx_max_depth
        self.fx_subsample = fx_subsample
        self.fx_colsample_bytree = fx_colsample_bytree

        self.gx_ridge_alpha = gx_ridge_alpha
        self.sigma = sigma
        self.c_max = c_max
        self.temperature = temperature
        self.min_weight = min_weight

        self.random_state = random_state
        self.verbose = verbose
        self.output_dir = output_dir

        # Model components
        self.fx_model = None
        self.fx_imputer = None
        self.fx_scaler = None

        self.gx_model = None
        self.gx_imputer = None
        self.gx_scaler = None

        # Selected parameters
        self.b_opt_ = None       # (b_12, b_23)
        self.lambda_opt_ = None  # (lambda_1, lambda_2, lambda_3)
        self.tau_opt_ = None     # calibrated temperature

        # Stored data partitions
        self.X_fx_train = None
        self.X_gx_train = None
        self.y_train = None
        self.q_train = None

        self.X_fx_val = None
        self.X_gx_val = None
        self.y_val = None
        self.q_val = None

        self.X_fx_test = None
        self.X_gx_test = None
        self.y_test = None
        self.q_test = None

        # OOF predictions and latent scores
        self.f_oof_train_ = None
        self.z_train_ = None
        self.z_val_ = None
        self.z_test_ = None

        # Timing
        self.training_time_ = None
        self.oof_time_ = None
        self.optimization_time_ = None

        # Optimization history
        self.optimization_history_ = []
        self.weight_search_history_ = []

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _prepare_data(self, X, fit=False, imputer=None, scaler=None):
        """Median imputation + standardization, fit on training data only."""
        if fit:
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_processed = imputer.fit_transform(X)
            X_processed = scaler.fit_transform(X_processed)
            return X_processed, imputer, scaler
        X_processed = imputer.transform(X)
        X_processed = scaler.transform(X_processed)
        return X_processed

    # ------------------------------------------------------------------
    # Step 2: f(x) - global latent-risk function (five-fold OOF)
    # ------------------------------------------------------------------

    def _fit_fx_with_oof(self, X_fx, q, y, n_folds=5):
        """Fit f(x) and generate five-fold stratified OOF predictions.

        Fold assignments are generated once from the fixed seed and held
        fixed across candidate configurations. Data-dependent preprocessing
        is re-estimated within each model-fitting fold.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 2: Fit global latent-risk function f(x) (5-fold OOF)")
            print("=" * 60)

        n_samples = len(q)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.random_state)

        f_oof = np.zeros(n_samples)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_fx, y)):
            X_train_fold, X_val_fold = X_fx[train_idx], X_fx[val_idx]
            q_train_fold = q[train_idx]

            # Preprocessing re-estimated within the model-fitting folds
            X_train_proc, imputer, scaler = self._prepare_data(X_train_fold, fit=True)
            X_val_proc = self._prepare_data(X_val_fold, fit=False,
                                            imputer=imputer, scaler=scaler)

            # LightGBM regression on q in {0, 0.5, 1} (squared-error objective)
            model = LGBMRegressor(
                n_estimators=self.fx_n_estimators,
                learning_rate=self.fx_learning_rate,
                max_depth=self.fx_max_depth,
                subsample=self.fx_subsample,
                colsample_bytree=self.fx_colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train_proc, q_train_fold)

            f_pred = model.predict(X_val_proc)
            f_oof[val_idx] = f_pred

            fold_scores.append(model.score(X_train_proc, q_train_fold))
            if self.verbose:
                print(f"  Fold {fold + 1}: R^2 = {fold_scores[-1]:.4f}")

        # Clip OOF predictions to [0, 1]
        f_oof = np.clip(f_oof, 0, 1)
        self.f_oof_train_ = f_oof

        # Refit the selected configuration on the complete training set
        if self.verbose:
            print("\n  Refitting f(x) on the complete training set...")

        X_fx_proc, self.fx_imputer, self.fx_scaler = self._prepare_data(X_fx, fit=True)

        self.fx_model = LGBMRegressor(
            n_estimators=self.fx_n_estimators,
            learning_rate=self.fx_learning_rate,
            max_depth=self.fx_max_depth,
            subsample=self.fx_subsample,
            colsample_bytree=self.fx_colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.fx_model.fit(X_fx_proc, q)

        if self.verbose:
            print(f"  f(x) OOF predictions: mean={f_oof.mean():.4f}, std={f_oof.std():.4f}")
            print("  f(x) fitted.")

        return self

    def _predict_fx(self, X_fx):
        """Global latent-risk score, clipped to [0, 1]."""
        X_proc = self._prepare_data(X_fx, fit=False,
                                    imputer=self.fx_imputer,
                                    scaler=self.fx_scaler)
        f_pred = self.fx_model.predict(X_proc)
        return np.clip(f_pred, 0, 1)

    # ------------------------------------------------------------------
    # Step 3: g(x) - sample-adaptive correction
    # ------------------------------------------------------------------

    def _compute_preliminary_thresholds(self, f_oof, y):
        """Preliminary transition points from training OOF scores.

        Grid search over {(b12, b23) : b12, b23 in {0.01, ..., 0.99}, b12 < b23}
        with the equal-weight objective (kappa_Q + tau_b + BA) / 3, where
        kappa_Q and tau_b are rescaled to [0, 1]. Ties: smaller b12, then
        smaller b23.
        """
        thresholds = np.linspace(0.01, 0.99, 99)

        best_score = -np.inf
        best_b = (0.33, 0.67)

        for b1 in thresholds:
            for b2 in thresholds:
                if b1 >= b2:
                    continue

                # Hard ordinal prediction from the OOF global-risk score
                y_pred = np.ones_like(y)
                y_pred[f_oof >= b1] = 2
                y_pred[f_oof >= b2] = 3

                kappa = cohen_kappa_score(y, y_pred, weights="quadratic")
                tau = kendalltau(y, y_pred)[0]
                ba = balanced_accuracy_score(y, y_pred)

                # Rescale kappa and tau to [0, 1]
                kappa_t = (kappa + 1) / 2
                tau_t = (tau + 1) / 2

                # Equal-weight preliminary objective
                score = (kappa_t + tau_t + ba) / 3

                if score > best_score + 1e-10:
                    best_score = score
                    best_b = (b1, b2)
                elif abs(score - best_score) < 1e-10:
                    # Tie: smaller b12, then smaller b23
                    if b1 < best_b[0]:
                        best_b = (b1, b2)
                    elif b1 == best_b[0] and b2 < best_b[1]:
                        best_b = (b1, b2)

        if self.verbose:
            print(f"  Preliminary thresholds: b12={best_b[0]:.4f}, b23={best_b[1]:.4f}")
            print(f"  Preliminary objective: {best_score:.4f}")

        return best_b

    def _compute_boundary_weights(self, f_oof, b12, b23, sigma):
        """Gaussian boundary-proximity weights, normalized to mean 1."""
        d1 = np.abs(f_oof - b12)
        d2 = np.abs(f_oof - b23)
        d = np.minimum(d1, d2)

        u = np.exp(-d ** 2 / (2 * sigma ** 2))
        w = len(f_oof) * u / (u.sum() + EPS)
        return w

    def _fit_gx(self, X_gx, f_oof, q, y, b_prelim):
        """Fit the boundary-weighted Ridge correction.

        (beta_g0, beta_g) = argmin sum_i w_i (r_i - beta_g0 - beta_g' z_i)^2
                                     + alpha ||beta_g||^2 ,
        with r_i = q_i - f_oof_i. The intercept is not penalized
        (scikit-learn Ridge behavior).
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 3: Fit sample-adaptive correction g(x)")
            print("=" * 60)

        b12, b23 = b_prelim

        # Residual target: r_i = q_i - f_oof_i
        residual = q - f_oof

        # Boundary-proximity weights
        weights = self._compute_boundary_weights(f_oof, b12, b23, self.sigma)

        if self.verbose:
            print(f"  Boundary weights: mean={weights.mean():.4f}, "
                  f"min={weights.min():.4f}, max={weights.max():.4f}")
            print(f"  Effective samples near boundaries: {(weights > 0.5).sum()}")

        # Standardize the g(x) predictors (parameters from training data)
        X_gx_proc, self.gx_imputer, self.gx_scaler = self._prepare_data(X_gx, fit=True)

        # Weighted Ridge regression (intercept fitted, not penalized)
        self.gx_model = Ridge(alpha=self.gx_ridge_alpha, random_state=self.random_state)
        self.gx_model.fit(X_gx_proc, residual, sample_weight=weights)

        # Bound the correction to [-c_max, c_max]
        g_pred = self.gx_model.predict(X_gx_proc)
        g_pred = np.clip(g_pred, -self.c_max, self.c_max)

        if self.verbose:
            print(f"  g(x) predictions: mean={g_pred.mean():.4f}, "
                  f"std={g_pred.std():.4f}, range=[{g_pred.min():.4f}, {g_pred.max():.4f}]")
            print("  g(x) fitted.")

        self.g_pred_train_ = g_pred
        return self

    def _predict_gx(self, X_gx):
        """Bounded sample-adaptive correction."""
        X_proc = self._prepare_data(X_gx, fit=False,
                                    imputer=self.gx_imputer,
                                    scaler=self.gx_scaler)
        g_pred = self.gx_model.predict(X_proc)
        return np.clip(g_pred, -self.c_max, self.c_max)

    # ------------------------------------------------------------------
    # Step 4: Boundary-based soft ordinal decoding
    # ------------------------------------------------------------------

    def _decode_proba(self, z, b12, b23, tau):
        """Soft ordinal decoding with a numerically stable softmax.

        Logits: l1 = (b12 - z) / tau, l2 = 0, l3 = (z - b23) / tau.
        """
        z = np.clip(z, 0, 1)
        b12 = np.clip(b12, 0, 1)
        b23 = np.clip(b23, 0, 1)

        logits = np.zeros((len(z), 3))
        logits[:, 0] = (b12 - z) / max(tau, 1e-10)
        logits[:, 1] = 0.0
        logits[:, 2] = (z - b23) / max(tau, 1e-10)

        logits = logits - logsumexp(logits, axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba

    def _decode_hard(self, z, b12, b23):
        """Hard ordinal prediction; ties go to the higher-risk class."""
        y_pred = np.ones(len(z), dtype=int)
        y_pred[z >= b12] = 2
        y_pred[z >= b23] = 3
        return y_pred

    def _compute_nll(self, z, y, b12, b23, tau):
        """Mean negative log-likelihood of the observed class."""
        proba = self._decode_proba(z, b12, b23, tau)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)
        nll = -np.mean(np.log(proba[np.arange(len(y)), y - 1]))
        return nll

    def _compute_brier(self, z, y, b12, b23, tau):
        """Multiclass Brier score."""
        proba = self._decode_proba(z, b12, b23, tau)
        y_bin = label_binarize(y, classes=[1, 2, 3])
        return np.mean(np.sum((proba - y_bin) ** 2, axis=1))

    # ------------------------------------------------------------------
    # Step 5: Validation objective and threshold optimization
    # ------------------------------------------------------------------

    def _compute_objective(self, z, y, b12, b23, weights):
        """Composite validation objective.

        T_val = w1 * kappa_Q_tilde + w2 * tau_b_tilde + w3 * BA,
        with kappa_Q and tau_b rescaled to [0, 1].
        """
        y_pred = self._decode_hard(z, b12, b23)

        kappa = cohen_kappa_score(y, y_pred, weights="quadratic")
        tau = kendalltau(y, y_pred)[0]
        ba = balanced_accuracy_score(y, y_pred)

        kappa_t = (kappa + 1) / 2
        tau_t = (tau + 1) / 2

        objective = weights[0] * kappa_t + weights[1] * tau_t + weights[2] * ba

        return objective, {
            "kappa": kappa,
            "kappa_t": kappa_t,
            "tau": tau,
            "tau_t": tau_t,
            "ba": ba,
            "objective": objective,
        }

    def _optimize_thresholds(self, z_val, y_val, weights, max_iter=100):
        """Coordinate search over transition thresholds.

        - Initialization: b12 = 1/3, b23 = 2/3
        - Candidate values: {0.01, 0.02, ..., 0.99} with 0 < b12 < b23 < 1
        - One threshold is varied while the other is held fixed
        - An update is retained only if it increases the objective
        - Stop after a full iteration without improvement or 100 iterations
        - Ties: smaller candidate value
        """
        if self.verbose:
            print("\n  Step 5: Coordinate search over thresholds")
            print("    Initialization: b12=1/3, b23=2/3")
            print("    Candidates: {0.01, 0.02, ..., 0.99}")
            print(f"    max_iter={max_iter}")

        thresholds = np.arange(0.01, 1.00, 0.01)

        b12 = 1 / 3
        b23 = 2 / 3

        current_score, _ = self._compute_objective(z_val, y_val, b12, b23, weights)
        history = []

        for iteration in range(max_iter):
            improved = False

            # Optimize b12 with b23 held fixed
            best_b12 = b12
            best_b12_score = -np.inf
            for candidate in thresholds:
                if candidate >= b23 - 0.01:
                    break
                score, _ = self._compute_objective(z_val, y_val, candidate, b23, weights)
                if score > best_b12_score + 1e-10:
                    best_b12_score = score
                    best_b12 = candidate
                elif abs(score - best_b12_score) < 1e-10:
                    if candidate < best_b12:
                        best_b12 = candidate

            if best_b12_score > current_score + 1e-10:
                b12 = best_b12
                current_score = best_b12_score
                improved = True

            # Optimize b23 with b12 held fixed
            best_b23 = b23
            best_b23_score = -np.inf
            for candidate in thresholds:
                if candidate <= b12 + 0.01:
                    continue
                score, _ = self._compute_objective(z_val, y_val, b12, candidate, weights)
                if score > best_b23_score + 1e-10:
                    best_b23_score = score
                    best_b23 = candidate
                elif abs(score - best_b23_score) < 1e-10:
                    if candidate < best_b23:
                        best_b23 = candidate

            if best_b23_score > current_score + 1e-10:
                b23 = best_b23
                current_score = best_b23_score
                improved = True

            score, metrics = self._compute_objective(z_val, y_val, b12, b23, weights)
            history.append({
                "iteration": iteration + 1,
                "b12": b12,
                "b23": b23,
                "objective": score,
                **metrics,
            })

            if not improved:
                if self.verbose:
                    print(f"    Converged after iteration {iteration + 1}")
                break

        if self.verbose:
            final_score, final_metrics = self._compute_objective(
                z_val, y_val, b12, b23, weights)
            print(f"    Final thresholds: b12={b12:.4f}, b23={b23:.4f}")
            print(f"    Final objective: {final_score:.4f}")
            print(f"    Kappa: {final_metrics['kappa']:.4f}")
            print(f"    Kendall's tau: {final_metrics['tau']:.4f}")
            print(f"    Balanced Accuracy: {final_metrics['ba']:.4f}")

        return (b12, b23), history

    # ------------------------------------------------------------------
    # Multi-objective weight search
    # ------------------------------------------------------------------

    def _search_weights(self, z_val, y_val):
        """Search the constrained objective-weight simplex.

        Candidate set: w_j in {0.20, 0.22, ..., 0.60}, sum_j w_j = 1.
        For each candidate, thresholds are optimized by coordinate search.
        Ties: weight vector closest to equal weighting (smallest
        D_lambda = sum_j (w_j - 1/3)^2); then the first vector in the
        prespecified search order.
        """
        if self.verbose:
            print("\n  Step 5: Multi-objective weight search")
            print("    Weight range: {0.20, 0.22, ..., 0.60}")
            print("    Constraint: w_j >= 0.20, sum w_j = 1")

        weight_values = np.arange(0.20, 0.61, 0.02)

        best_score = -np.inf
        best_weights = None
        best_thresholds = None
        best_history = None

        total_combinations = 0

        for l1 in weight_values:
            for l2 in weight_values:
                l3 = 1 - l1 - l2
                if l3 < 0.20 - EPS or l3 > 0.60 + EPS:
                    continue
                if abs(l1 + l2 + l3 - 1) > 0.01:
                    continue

                weights = (l1, l2, l3)
                total_combinations += 1

                # Optimize thresholds for this weight vector
                thresholds, history = self._optimize_thresholds(z_val, y_val, weights)
                score, _ = self._compute_objective(
                    z_val, y_val, thresholds[0], thresholds[1], weights)

                if score > best_score + 1e-10:
                    best_score = score
                    best_weights = weights
                    best_thresholds = thresholds
                    best_history = history
                elif abs(score - best_score) < 1e-10:
                    # Tie: closest to equal weighting
                    current_dist = np.sum((np.array(weights) - 1 / 3) ** 2)
                    best_dist = np.sum((np.array(best_weights) - 1 / 3) ** 2)
                    if current_dist < best_dist - 1e-10:
                        best_weights = weights
                        best_thresholds = thresholds
                        best_history = history

        if self.verbose:
            print(f"    Searched {total_combinations} weight combinations")
            print(f"    Selected weights: w1={best_weights[0]:.3f}, "
                  f"w2={best_weights[1]:.3f}, w3={best_weights[2]:.3f}")
            print(f"    Best objective: {best_score:.4f}")

        return best_weights, best_thresholds, best_history

    # ------------------------------------------------------------------
    # Step 6: Temperature calibration
    # ------------------------------------------------------------------

    def _calibrate_temperature(self, z_val, y_val, b12, b23):
        """Calibrate the soft-decoding temperature.

        Candidates: {0.01, 0.05, 0.10, 0.20}; select the temperature that
        minimizes validation NLL. Ties: smallest temperature.
        """
        if self.verbose:
            print("\n  Step 6: Temperature calibration")

        candidate_taus = [0.01, 0.05, 0.10, 0.20]

        best_nll = np.inf
        best_tau = candidate_taus[0]

        for tau in candidate_taus:
            nll = self._compute_nll(z_val, y_val, b12, b23, tau)
            if self.verbose:
                print(f"    tau={tau:.2f}: NLL={nll:.4f}")

            if nll < best_nll - 1e-10:
                best_nll = nll
                best_tau = tau
            elif abs(nll - best_nll) < 1e-10:
                if tau < best_tau:
                    best_tau = tau

        if self.verbose:
            print(f"    Selected temperature: tau={best_tau:.2f}")

        return best_tau

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def fit(self, X_fx, X_gx, y, q=None, val_split=0.15, test_split=0.15):
        """Full AMOSOA training pipeline with internal data partitioning.

        Stratified 70/15/15 train/validation/test split with the fixed
        random seed, followed by `fit_with_data`.
        """
        if q is None:
            q = (y - 1) / 2.0

        # ---- Step 1: Stratified 70/15/15 partition ----
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 1: Data partition (70/15/15, stratified)")
            print("=" * 60)

        (X_fx_train, X_fx_temp, X_gx_train, X_gx_temp,
         y_train, y_temp, q_train, q_temp) = train_test_split(
            X_fx, X_gx, y, q,
            test_size=val_split + test_split,
            stratify=y,
            random_state=self.random_state,
        )

        val_size = val_split / (val_split + test_split)
        (X_fx_val, X_fx_test, X_gx_val, X_gx_test,
         y_val, y_test, q_val, q_test) = train_test_split(
            X_fx_temp, X_gx_temp, y_temp, q_temp,
            test_size=1 - val_size,
            stratify=y_temp,
            random_state=self.random_state,
        )

        return self.fit_with_data(
            X_fx_train, X_gx_train, y_train, q_train,
            X_fx_val, X_gx_val, y_val, q_val,
            X_fx_test, X_gx_test, y_test, q_test,
        )

    def fit_with_data(self, X_fx_train, X_gx_train, y_train, q_train,
                      X_fx_val, X_gx_val, y_val, q_val,
                      X_fx_test, X_gx_test, y_test, q_test):
        """Fit AMOSOA on externally provided train/validation/test splits."""
        overall_start = time.time()

        self.X_fx_train = X_fx_train
        self.X_gx_train = X_gx_train
        self.y_train = y_train
        self.q_train = q_train

        self.X_fx_val = X_fx_val
        self.X_gx_val = X_gx_val
        self.y_val = y_val
        self.q_val = q_val

        self.X_fx_test = X_fx_test
        self.X_gx_test = X_gx_test
        self.y_test = y_test
        self.q_test = q_test

        if self.verbose:
            print(f"  Training samples: {len(y_train)}")
            print(f"  Validation samples: {len(y_val)}")
            print(f"  Test samples: {len(y_test)}")
            print(f"  Training distribution: {np.bincount(y_train)[1:]}")
            print(f"  Validation distribution: {np.bincount(y_val)[1:]}")
            print(f"  Test distribution: {np.bincount(y_test)[1:]}")

        # ---- Step 2: f(x) with OOF predictions ----
        oof_start = time.time()
        self._fit_fx_with_oof(X_fx_train, q_train, y_train, n_folds=5)
        self.oof_time_ = time.time() - oof_start

        # ---- Step 3: Preliminary transition points ----
        b_prelim = self._compute_preliminary_thresholds(self.f_oof_train_, y_train)

        # ---- Step 3: g(x) - boundary-weighted Ridge ----
        self._fit_gx(X_gx_train, self.f_oof_train_, q_train, y_train, b_prelim)

        # ---- Composite latent scores ----
        f_train = self._predict_fx(X_fx_train)
        g_train = self._predict_gx(X_gx_train)
        z_train = np.clip(f_train + g_train, 0, 1)

        f_val = self._predict_fx(X_fx_val)
        g_val = self._predict_gx(X_gx_val)
        z_val = np.clip(f_val + g_val, 0, 1)

        self.z_train_ = z_train
        self.z_val_ = z_val

        # ---- Step 5: Weight search + threshold optimization (validation) ----
        optimization_start = time.time()
        self.lambda_opt_, self.b_opt_, opt_history = self._search_weights(z_val, y_val)
        self.optimization_history_ = opt_history
        self.optimization_time_ = time.time() - optimization_start

        # ---- Step 6: Temperature calibration (validation NLL) ----
        b12_opt, b23_opt = self.b_opt_
        self.tau_opt_ = self._calibrate_temperature(z_val, y_val, b12_opt, b23_opt)
        self.temperature = self.tau_opt_

        self.training_time_ = time.time() - overall_start

        if self.verbose:
            print("\n" + "=" * 60)
            print("AMOSOA training completed")
            print("=" * 60)
            print(f"  Selected weights: w1={self.lambda_opt_[0]:.3f}, "
                  f"w2={self.lambda_opt_[1]:.3f}, w3={self.lambda_opt_[2]:.3f}")
            print(f"  Selected thresholds: b12={self.b_opt_[0]:.4f}, "
                  f"b23={self.b_opt_[1]:.4f}")
            print(f"  Selected temperature: tau={self.tau_opt_:.2f}")
            print(f"  OOF time: {self.oof_time_:.2f}s")
            print(f"  Optimization time: {self.optimization_time_:.2f}s")
            print(f"  Total training time: {self.training_time_:.2f}s")

            # Validation performance
            y_pred_val = self.predict(X_fx_val, X_gx_val)
            cm_val = confusion_matrix(y_val, y_pred_val)
            nll_val = self._compute_nll(z_val, y_val, self.b_opt_[0],
                                        self.b_opt_[1], self.tau_opt_)
            brier_val = self._compute_brier(z_val, y_val, self.b_opt_[0],
                                            self.b_opt_[1], self.tau_opt_)

            print("\n  Validation performance:")
            print(f"    Balanced Accuracy: "
                  f"{balanced_accuracy_score(y_val, y_pred_val):.4f}")
            print(f"    Kappa (quadratic): "
                  f"{cohen_kappa_score(y_val, y_pred_val, weights='quadratic'):.4f}")
            print(f"    Kendall's tau: {kendalltau(y_val, y_pred_val)[0]:.4f}")
            print(f"    NLL: {nll_val:.4f}")
            print(f"    Brier Score: {brier_val:.4f}")
            print(f"\n    Confusion matrix:\n    {cm_val}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_latent(self, X_fx, X_gx):
        """Composite latent-risk score z = clip(f(x) + g_bounded(x), 0, 1)."""
        f_pred = self._predict_fx(X_fx)
        g_pred = self._predict_gx(X_gx)
        return np.clip(f_pred + g_pred, 0, 1)

    def predict_proba(self, X_fx, X_gx):
        """Class probabilities from soft ordinal decoding."""
        z = self.predict_latent(X_fx, X_gx)
        if self.b_opt_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        b12, b23 = self.b_opt_
        return self._decode_proba(z, b12, b23, self.tau_opt_)

    def predict(self, X_fx, X_gx):
        """Hard ordinal class labels in {1, 2, 3}."""
        z = self.predict_latent(X_fx, X_gx)
        if self.b_opt_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        b12, b23 = self.b_opt_
        return self._decode_hard(z, b12, b23)

    def predict_with_uncertainty(self, X_fx, X_gx):
        """Predicted labels and 1 - max class probability."""
        proba = self.predict_proba(X_fx, X_gx)
        y_pred = np.argmax(proba, axis=1) + 1
        uncertainty = 1 - np.max(proba, axis=1)
        return y_pred, uncertainty

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, feature_names_fx=None, feature_names_gx=None):
        """Feature importance for f(x) (gain-based) and g(x) (coefficients)."""
        importance = {}

        if self.fx_model is not None:
            fx_imp = self.fx_model.feature_importances_
            if feature_names_fx is not None:
                sorted_idx = np.argsort(fx_imp)[::-1]
                importance["fx"] = {
                    "names": [feature_names_fx[i] for i in sorted_idx],
                    "values": fx_imp[sorted_idx],
                    "dict": dict(zip(feature_names_fx, fx_imp)),
                }
            else:
                importance["fx"] = fx_imp

        if self.gx_model is not None:
            gx_coef = self.gx_model.coef_
            if feature_names_gx is not None:
                sorted_idx = np.argsort(np.abs(gx_coef))[::-1]
                importance["gx"] = {
                    "names": [feature_names_gx[i] for i in sorted_idx],
                    "coefficients": gx_coef[sorted_idx],
                    "importance": np.abs(gx_coef[sorted_idx]),
                    "dict_coef": dict(zip(feature_names_gx, gx_coef)),
                    "dict_imp": dict(zip(feature_names_gx, np.abs(gx_coef))),
                }
            else:
                importance["gx"] = gx_coef

        return importance

    def get_optimization_history(self):
        """Threshold optimization history as a DataFrame."""
        if self.optimization_history_:
            return pd.DataFrame(self.optimization_history_)
        return pd.DataFrame()

    def get_gx_statistics(self):
        """Summary statistics of the bounded correction on the training set."""
        if hasattr(self, "g_pred_train_"):
            return {
                "mean": np.mean(self.g_pred_train_),
                "std": np.std(self.g_pred_train_),
                "min": np.min(self.g_pred_train_),
                "max": np.max(self.g_pred_train_),
                "mean_abs": np.mean(np.abs(self.g_pred_train_)),
            }
        return None


# ---------------------------------------------------------------------------
# Part 3: Model evaluation
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Evaluation of a fitted AMOSOA model (Step 7 of the pipeline)."""

    def __init__(self, model, output_dir=None):
        self.model = model
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, X_fx, X_gx, y_true, dataset_name="Test", verbose=True):
        """Compute the full evaluation metric set."""
        inference_start = time.time()
        y_pred = self.model.predict(X_fx, X_gx)
        y_prob = self.model.predict_proba(X_fx, X_gx)
        z = self.model.predict_latent(X_fx, X_gx)
        inference_time = time.time() - inference_start

        cm = confusion_matrix(y_true, y_pred)
        total = cm.sum()

        # Severe errors: direct cross-level errors (|y_hat - y| >= 2)
        severe_errors = cm[0, 2] + cm[2, 0]
        severe_error_rate = severe_errors / total

        # Adjacent (one-level) errors
        near_errors = cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]
        near_error_rate = near_errors / total

        # Probability quality
        b12, b23 = self.model.b_opt_
        tau = self.model.tau_opt_
        nll = self.model._compute_nll(z, y_true, b12, b23, tau)
        brier = self.model._compute_brier(z, y_true, b12, b23, tau)

        # Per-class metrics
        class_recalls, class_precisions, class_f1s = {}, {}, {}
        for cls in [0, 1, 2]:
            mask = y_true == (cls + 1)
            if mask.sum() > 0:
                recall = cm[cls, cls] / mask.sum()
                precision = cm[cls, cls] / (y_pred[y_pred == (cls + 1)].sum() + EPS)
                f1 = 2 * precision * recall / (precision + recall + EPS)
                class_recalls[f"class_{cls + 1}_recall"] = recall
                class_precisions[f"class_{cls + 1}_precision"] = precision
                class_f1s[f"class_{cls + 1}_f1"] = f1
            else:
                class_recalls[f"class_{cls + 1}_recall"] = 0
                class_precisions[f"class_{cls + 1}_precision"] = 0
                class_f1s[f"class_{cls + 1}_f1"] = 0

        results = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "kappa_quadratic": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "kappa_linear": cohen_kappa_score(y_true, y_pred, weights="linear"),
            "kendall_tau": kendalltau(y_true, y_pred)[0],
            "spearman_rho": spearmanr(y_true, y_pred)[0],
            "severe_error_rate": severe_error_rate,
            "near_error_rate": near_error_rate,
            "nll": nll,
            "brier_score": brier,
            "inference_time": inference_time,
            "inference_samples": len(y_true),
            "inference_time_per_sample": inference_time / len(y_true),
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "z": z,
            **class_recalls,
            **class_precisions,
            **class_f1s,
        }

        try:
            y_bin = label_binarize(y_true, classes=[1, 2, 3])
            if y_prob.shape[1] == 3:
                results["macro_auc"] = roc_auc_score(
                    y_bin, y_prob, average="macro", multi_class="ovr")
                results["weighted_auc"] = roc_auc_score(
                    y_bin, y_prob, average="weighted", multi_class="ovr")
        except Exception:
            pass

        if verbose:
            print("\n" + "=" * 60)
            print(f"AMOSOA evaluation - {dataset_name}")
            print("=" * 60)

            print("  Overall metrics:")
            metrics_display = [
                ("balanced_accuracy", "Balanced Accuracy"),
                ("accuracy", "Accuracy"),
                ("macro_f1", "Macro F1"),
                ("kappa_quadratic", "Kappa (quadratic)"),
                ("kendall_tau", "Kendall's tau"),
                ("macro_auc", "Macro AUC"),
                ("severe_error_rate", "Severe Error Rate"),
                ("near_error_rate", "Near Error Rate"),
                ("nll", "Negative Log-Likelihood"),
                ("brier_score", "Brier Score"),
            ]
            for key, label in metrics_display:
                if key in results:
                    print(f"    {label:>25}: {results[key]:.4f}")

            print("\n  Per-class recall:")
            for cls in [1, 2, 3]:
                print(f"    Class {cls}: {results[f'class_{cls}_recall']:.4f}")

            print("\n  Inference efficiency:")
            print(f"    {'Inference time':>25}: {inference_time:.4f}s")
            print(f"    {'Samples':>25}: {len(y_true)}")
            print(f"    {'Time per sample':>25}: "
                  f"{inference_time / len(y_true):.6f}s")

            print(f"\n  Confusion matrix:\n  {cm}")

        return results

    def save_results(self, results, dataset_name="Test"):
        """Save metrics, confusion matrix, and predictions to CSV files."""
        if self.output_dir is None:
            return

        metrics_to_save = {
            k: v for k, v in results.items()
            if k not in ["confusion_matrix", "y_true", "y_pred", "y_prob", "z"]
        }
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": v} for k, v in metrics_to_save.items()
        ])
        metrics_df.to_csv(
            os.path.join(self.output_dir, f"{dataset_name}_metrics.csv"), index=False)

        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{i}" for i in [1, 2, 3]],
            columns=[f"Pred_{i}" for i in [1, 2, 3]],
        )
        cm_df.to_csv(os.path.join(self.output_dir, f"{dataset_name}_confusion_matrix.csv"))

        pred_df = pd.DataFrame({
            "true_label": results["y_true"],
            "predicted": results["y_pred"],
            "z_score": results["z"],
        })
        proba = results["y_prob"]
        for i in range(proba.shape[1]):
            pred_df[f"prob_class_{i + 1}"] = proba[:, i]
        pred_df.to_csv(
            os.path.join(self.output_dir, f"{dataset_name}_predictions.csv"), index=False)

        print(f"  Results saved to {self.output_dir}")

    def plot_confusion_matrix(self, results, dataset_name="Test"):
        """Plot and save the confusion matrix."""
        cm = results["confusion_matrix"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Active (1)", "At-Risk (2)", "Churned (3)"],
                    yticklabels=["Active (1)", "At-Risk (2)", "Churned (3)"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"AMOSOA Confusion Matrix - {dataset_name}")

        if self.output_dir:
            plt.savefig(
                os.path.join(self.output_dir, f"{dataset_name}_confusion_matrix.png"),
                dpi=300, bbox_inches="tight")
        plt.show()

    def plot_threshold_optimization(self):
        """Plot the threshold optimization history."""
        history = self.model.get_optimization_history()
        if len(history) == 0:
            print("No optimization history available.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["iteration"], history["b12"], "b-o", label="b12")
        axes[0].plot(history["iteration"], history["b23"], "r-o", label="b23")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Threshold Value")
        axes[0].set_title("Threshold Optimization Progress")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["iteration"], history["objective"], "g-o")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Objective Value")
        axes[1].set_title("Multi-Objective Optimization Progress")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, "threshold_optimization.png"),
                        dpi=300, bbox_inches="tight")
        plt.show()
