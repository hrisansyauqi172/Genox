import os
import inspect
import time
import optuna
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization.matplotlib import plot_optimization_history
from Genox import GraphTransform, WWLAnalysis, KernelApproach

# Set device for PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU")

class ManualGPR:
    def __init__(self, noise=1e-4):
        self.noise = noise
        self.alpha = None
        self.L = None
        self.X_train = None

    def fit(self, K_train, y):
        """
        K_train : precomputed kernel matrix (n x n)
        y       : targets (n,)
        """
        n = len(y)
        K = K_train + self.noise * np.eye(n)

        # Cholesky factor
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), y)
        return self

    def predict(self, K_val, K_self=None, return_std=False):
        """
        K_val : (m x n) validation vs train kernel matrix
        K_self : (m x m) validation vs validation kernel matrix OR diagonal vector (m,)
        return_std : whether to return standard deviation
        """
        # Mean
        mean = K_val @ self.alpha
        
        if not return_std:
            return mean
        
        # Solve v = L^-1 K_val^T
        v = solve_triangular(self.L, K_val.T, lower=True)

        # variance = K_self - v^T v
        if K_self is None:
            raise ValueError("K_self required for uncertainty")
        
        if K_self.ndim == 2:
            var = np.diag(K_self) - np.sum(v**2, axis=0)
        else:
            var = K_self - np.sum(v**2, axis=0)

        var = np.maximum(var, 1e-12)  # numerical safety
        std = np.sqrt(var)
        
        return mean, std

    def log_marginal_likelihood(self, K_train, y):
        n = len(y)
        K = K_train + self.noise * np.eye(n)

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), y)

        lml = -0.5 * y @ alpha
        lml -= np.sum(np.log(np.diag(L)))
        lml -= 0.5 * n * np.log(2 * np.pi)
        return lml
    
    def optimize_kernel_hyperparameters(D, y, type_kernel="RBF", CV=None, idx_train=None, random_state=42):
        if CV is not None:
            kf = KFold(n_splits=CV, shuffle=True, random_state=random_state)
        
        median_D = np.median(D[D > 0])
        mean_D = np.mean(D[D > 0])

        sigma_0, sigma_1 = sorted([median_D, mean_D])

        low_sigma_bound = max(0.01 * sigma_0, 1.0)
        high_sigma_bound = sigma_1

        init = [1e-3, sigma_0, 3.0]
        bounds = [(1e-7, 1), (low_sigma_bound, high_sigma_bound), (1e-6, 30)]

        if type_kernel in ["Cauchy", "RationalQuadratic", "IMQ"]:
            init.extend([1.0])
            bounds.extend([(0.1, 10)])

        log_init = np.log(init)
        log_bounds = [(np.log(lo), np.log(hi)) for lo, hi in bounds]

        # track best effective params
        best = {
            "value": np.inf,
            "params": None
        }

        def neg_lml(log_params):
            params = np.exp(log_params)
            fold_lml = 0.0
            noise = params[0]  # start from trial-suggested noise

            def compute_lml(D_sub, y_sub):
                K = KernelApproach._kernel_transform(D_sub, params[1:], type_kernel=type_kernel)
                current_noise = noise
                max_noise = 1.0

                while True:
                    try:
                        gpr = ManualGPR(noise=current_noise)
                        lml = gpr.log_marginal_likelihood(K, y_sub)
                        # sync back the adapted noise   
                        return lml, current_noise
                    except (np.linalg.LinAlgError, ValueError):
                        current_noise *= 10
                        if current_noise > max_noise:                                
                            return None, None  # ❗ failed
            
            if CV is not None:
                used_noise = 0.0
                for fold_idx_train, _ in kf.split(np.arange(len(y))):
                    D_train = D[fold_idx_train][:, fold_idx_train]
                    y_train = y[fold_idx_train]
                    lml, noise = compute_lml(D_train, y_train)
                    if lml is None:
                        return 1e10  # ❗ penalize bad params
                    if noise is not None:
                        used_noise = max(used_noise, noise)
                    fold_lml += lml
            else:
                D_train = D[idx_train][:, idx_train]
                y_train = y[idx_train]
                lml, used_noise = compute_lml(D_train, y_train)
                if lml is None:
                    return 1e10  # ❗ penalize bad params
                fold_lml += lml
            value = -fold_lml
            # track best effective params
            if value < best["value"]:
                best["value"] = value
                best["params"] = params.copy()
                best["params"][0] = used_noise  # update noise to adapted value

            return value

        res = minimize(neg_lml, x0=log_init, bounds=log_bounds)
        if not res.success:
            print(f"[WARNING] Hyperparameter optimization failed: {res.message}")
            return best["params"] if best["params"] is not None else init  # fallback to initial params
        else:
            return best["params"] if best["params"] is not None else np.exp(res.x)  # return adapted noise if tracked, else exp of optimization result

class PredictionModel:
    @staticmethod
    def _evaluate_model(D_kernel, ads_energy_list, noise, idx_train=None, idx_val=None, kf=None, trial=None):
        if kf is not None:
            all_true, all_pred = [], []

            for fold_id, (fold_idx_train, fold_idx_val) in enumerate(kf.split(np.arange(len(ads_energy_list)))):
                gpr = ManualGPR(noise=noise)
                gpr.fit(D_kernel[fold_idx_train][:, fold_idx_train], ads_energy_list[fold_idx_train])
                y_val_pred = gpr.predict(D_kernel[fold_idx_val][:, fold_idx_train])

                fold_mae = mean_absolute_error(ads_energy_list[fold_idx_val], y_val_pred)

                if trial:
                    trial.report(fold_mae, step=fold_id)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                all_true.extend(ads_energy_list[fold_idx_val])
                all_pred.extend(y_val_pred)
            mae = mean_absolute_error(all_true, all_pred)
        else:
            gpr = ManualGPR(noise=noise)
            gpr.fit(D_kernel[idx_train][:, idx_train], ads_energy_list[idx_train])
            y_val_pred = gpr.predict(D_kernel[idx_val][:, idx_train])

            mae = mean_absolute_error(ads_energy_list[idx_val], y_val_pred)

            if trial:
                trial.report(mae, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return mae
    
    @staticmethod
    def _kernel_r2_ceiling(D_kernel_train, D_kernel_val_train, y_train, y_val, noise):
        n_train = len(y_train)
        
        # Regularized inverse
        K = D_kernel_train + noise * np.eye(n_train)
        K_inv = np.linalg.inv(K)
        
        # Best possible kernel regression prediction
        y_pred = D_kernel_val_train @ K_inv @ y_train
        
        # R² calculation
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

        r2_ceiling = 1 - ss_res / ss_tot
        return r2_ceiling
    
    @staticmethod
    def _extract_trials_df(study):
        records = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {}
            # Optuna parameters
            for k, v in t.params.items():
                row[k] = v
            # User attributes (LML outputs)
            for k, v in t.user_attrs.items():
                if v is not None:
                    row[k] = v
            # Objective
            row["objective"] = t.value
            records.append(row)
        
        df = pd.DataFrame(records)
        df = df.select_dtypes(include=[float, int])

        return df
    
    @staticmethod
    def _compute_importance(df):
        X = df.drop(columns=["objective"])
        y = df["objective"]
        model = RandomForestRegressor(n_estimators=200, random_state=0)
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=20, random_state=0)
        importances = dict(zip(X.columns, result.importances_mean))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        return importances

    @staticmethod
    def _plot_optuna_result(study, plot_path, out_prefix=None):
        """
        Plot optimization history, hyperparameter importance from an Optuna study.

        Parameters:
        -----------
        study : optuna.Study
            The Optuna study object (after optimization).
        out_prefix : str or None
            Path prefix for saving PNGs.
        """
        df = PredictionModel._extract_trials_df(study)
        if df.shape[1] <= 1:
            print("[INFO] Not enough parameters for importance plot.")
            return
        
        # Hyperparameter Importance
        importances = PredictionModel._compute_importance(df)
        if len(importances) == 0:
            print("[INFO] No hyperparameters to show importance for.")
            return

        params = list(importances.keys())
        scores = list(importances.values())

        plt.figure(figsize=(6,4))
        plt.barh(params, scores, color='purple', alpha=0.7)
        plt.xlabel("Permutation Importance")
        plt.title(f"Hyperparameter Importance")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{out_prefix}_Hyperparam_Importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Optimization History
        ax = plot_optimization_history(study)
        ax.figure.savefig(os.path.join(plot_path, f"{out_prefix}_history.png"), dpi=300)
        plt.close()

    @staticmethod
    def _gpr_ads_energy(atoms_list, ads_energy_list, n_trials=100, random_state=42, CV=None, tt_split=None, spec_idx_train=None, spec_idx_val=None,
                        lml_act=True, ads_dist_threshold=None, inner_weight=None, outer_weight=None, wl_depth=None,
                        noise=None, sigma=None, amplitude=None, type_kernel="RBF",
                        node_importance=False, act_doub_scale=False, adj_ads_desc=False, gas_db=None, tune_pertub_param=False,
                        node_attr_keys = None, plot_name=None, save_folder=None):
        """
        Train GPR using WL Wasserstein kernel. Hyperparameters (ads_dist_threshold, wl_depth, sigma)
        are tuned using Optuna if not provided. Manual CV using a single train/validation split.

        Returns:
            model: trained GPR
            best_params: dict of best hyperparameters
            best_graphs: list of networkx graphs for each structure
            cv_r2: model performance
            cv_mae: model performance
            y_pred_all: predicted adsorption energy
            uncertainty_all: uncertainty of each predicted value
        """
        ads_energy_list = np.array(ads_energy_list)
        if CV is not None:
            kf = KFold(n_splits=CV, shuffle=True, random_state=random_state)
            idx_train, idx_val = None, None
        elif tt_split is not None:
            idx = np.arange(len(atoms_list))
            idx_train, idx_val = train_test_split(idx, test_size=tt_split, random_state=random_state, shuffle=True)
            kf = None
        elif spec_idx_train is not None and spec_idx_val is not None:
            idx_train = spec_idx_train
            idx_val = spec_idx_val
            kf = None
        
        distance_cache = {}

        def objective(trial):
            # Suggest hyperparameters if not fixed
            # adt = ads_dist_threshold or trial.suggest_float('ads_dist_threshold', 2.5, 4.5)
            adt = ads_dist_threshold or trial.suggest_float('ads_dist_threshold', 0.5, 2.5, step=0.05)
            in_weight = inner_weight or trial.suggest_float('inner_weight', 1, 15, step=0.5)
            out_weight = outer_weight or trial.suggest_float('outer_weight', 0.05, 1, step=0.01)
            depth = wl_depth or trial.suggest_int('wl_depth', 1, 3)

            # ---- Perturbation parameters (OPTIONAL) ----
            if adj_ads_desc:
                if tune_pertub_param:
                    beta = trial.suggest_float("beta", 0.0, 1.5, step=0.01)
                    lambda_corr = trial.suggest_float("lambda_corr", 0.0, 0.6, step=0.01)
                    sigma_d = trial.suggest_float("sigma_d", 0.2, 0.8, step=0.01)
                    sigma_theta = trial.suggest_float("sigma_theta", 0.1, 0.6, step=0.01)
                    alpha = trial.suggest_float("alpha", 0.5, 3.0, step=0.01)
                else:
                    beta = 0.5
                    lambda_corr = 0.3
                    sigma_d = 0.4
                    sigma_theta = 0.2
                    alpha = 1.5
                # store for later retrieval
                trial.set_user_attr("beta", beta)
                trial.set_user_attr("lambda_corr", lambda_corr)
                trial.set_user_attr("sigma_d", sigma_d)
                trial.set_user_attr("sigma_theta", sigma_theta)
                trial.set_user_attr("alpha", alpha)
            else:
                # adsorbate descriptor adjustment OFF → parameters irrelevant
                beta = lambda_corr = sigma_d = sigma_theta = alpha = None

            # Cache key includes all parameters that affect graph construction
            cache_key = (
                round(adt, 4),
                in_weight, out_weight,
                depth,
                beta, lambda_corr, sigma_d, sigma_theta, alpha
            )
            
            # # Build graphs and distance matrix caching
            if cache_key in distance_cache:
                D = distance_cache[cache_key]
            else:
                graphs = [
                    GraphTransform._make_graph(
                        atoms_init=item["contcar"],
                        adsorbate_label=item["ads_label"],
                        ads_dist_threshold=adt,
                        descriptor_df=item["descriptors"],
                        adj_ads_desc=adj_ads_desc,
                        gas_db=gas_db,
                        inner_weight=in_weight,
                        outer_weight=out_weight,
                        beta=beta,
                        lambda_corr=lambda_corr,
                        sigma_d=sigma_d,
                        sigma_theta=sigma_theta,
                        alpha=alpha
                    )
                    for item in atoms_list
                ]

                # Wasserstein distance and kernel
                if node_importance and act_doub_scale:
                    D, _ = WWLAnalysis._pairwise_wasserstein(graphs, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=adt)
                else:
                    D, _ = WWLAnalysis._pairwise_wasserstein(graphs, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance)
                distance_cache[cache_key] = D
            
            # Kernel hyperparameter optimization (LML)
            if lml_act:
                params = ManualGPR.optimize_kernel_hyperparameters(
                    D, ads_energy_list, type_kernel=type_kernel, CV=CV, idx_train=idx_train, random_state=random_state
                )
                nois = noise or params[0]
                sig = sigma or params[1]
                amp = amplitude or params[2]
                trial.set_user_attr("noise", nois)
                trial.set_user_attr("sigma", sig)
                trial.set_user_attr("amplitude", amp)
                extra_param = None
                if len(params) > 3:
                    extra_param = params[3]
                    trial.set_user_attr("extra_param", extra_param)
            else:
                median_D = np.median(D[D > 0])
                mean_D = np.mean(D[D > 0])
                sigma_0, sigma_1 = sorted([median_D, mean_D])
                low_sigma_bound = 0.3 * sigma_0
                high_sigma_bound = 20.0 * sigma_1
                nois = noise or trial.suggest_float('noise', 1e-7, 1e-2, log=True)
                sig = sigma or trial.suggest_float('sigma', low_sigma_bound, high_sigma_bound, log=True)
                amp = amplitude or trial.suggest_float('amplitude', 0.5, 30, log=True)
                params = [nois, sig, amp]
                if type_kernel in ["Cauchy", "RationalQuadratic", "IMQ"]:
                    extra_param = trial.suggest_float('extra_param', 0.1, 10.0, log=True)
                    params.append(extra_param)
            
            D_kernel = KernelApproach._kernel_transform(
                D, params[1:], type_kernel=type_kernel
            )

            global_mae = PredictionModel._evaluate_model(
                D_kernel=D_kernel, ads_energy_list=ads_energy_list,
                noise=nois, idx_train=idx_train, idx_val=idx_val, kf=kf, trial=trial
            )
                
            return global_mae

        # ----------------------------
        # Hyperparameter optimization
        # ----------------------------
        def print_usr_atr(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            # trial.value is the objective value
            noise = trial.user_attrs.get("noise", None)
            sigma = trial.user_attrs.get("sigma", None)
            amplitude = trial.user_attrs.get("amplitude", None)
            extra_param = trial.user_attrs.get("extra_param", None)
            print(f"[Trial {trial.number}] value={trial.value:.3f}, wl_params={trial.params}, gpr_params={{'noise': {noise}, 'sigma': {sigma}, 'amplitude': {amplitude}, 'extra_param': {extra_param}}}")

        if ads_dist_threshold is None or sigma is None or wl_depth is None:
            n_startup = max(40, int(0.25 * n_trials))
            tpe_sampler = TPESampler(
                multivariate=True,
                group=True,
                n_startup_trials=n_startup,
                seed=random_state
            )
            pruner = MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=3,
                interval_steps=1
            )
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{plot_name}_{type_kernel}_{int(time.time())}",
                load_if_exists=True,
                sampler=tpe_sampler,
                pruner=pruner
            )
            study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[print_usr_atr] if lml_act else None
            )

            best_params = study.best_trial.params.copy()
            if lml_act:
                best_params.update(study.best_trial.user_attrs)
                best_params["noise"] = study.best_trial.user_attrs["noise"]
                best_params["sigma"] = study.best_trial.user_attrs["sigma"]
                best_params["amplitude"] = study.best_trial.user_attrs["amplitude"]
                if "extra_param" in study.best_trial.user_attrs:
                    best_params["extra_param"] = study.best_trial.user_attrs["extra_param"]
            
            # ---- Perturbation params (ONLY if used) ----
            if adj_ads_desc:
                for key in ["beta", "lambda_corr", "sigma_d", "sigma_theta", "alpha"]:
                    best_params[key] = study.best_trial.user_attrs[key]

            if ads_dist_threshold is not None:
                best_params["ads_dist_threshold"] = ads_dist_threshold
            if inner_weight is not None:
                best_params["inner_weight"] = inner_weight
            if outer_weight is not None:
                best_params["outer_weight"] = outer_weight
            if wl_depth is not None:
                best_params["wl_depth"] = wl_depth
            if noise is not None:
                best_params["noise"] = noise
            if sigma is not None:
                best_params["sigma"] = sigma
            if amplitude is not None:
                best_params["amplitude"] = amplitude
        else:
            # All hyperparameters fixed, just train once
            best_params = {
                "ads_dist_threshold": ads_dist_threshold,
                "inner_weight": inner_weight,
                "outer_weight": outer_weight,
                "wl_depth": wl_depth,
                "noise": noise,
                "sigma": sigma,
                "amplitude": amplitude
            }

        # ----------------------------
        # Rebuild graphs with best hyperparameters
        # ----------------------------
        best_graphs = [
            GraphTransform._make_graph(
                atoms_init=item["contcar"],
                adsorbate_label=item["ads_label"],
                ads_dist_threshold=best_params["ads_dist_threshold"],
                descriptor_df=item["descriptors"],
                adj_ads_desc=adj_ads_desc,
                gas_db=gas_db,
                inner_weight=best_params["inner_weight"],
                outer_weight=best_params["outer_weight"],
                beta=best_params.get("beta", 0.5),
                lambda_corr=best_params.get("lambda_corr", 0.3),
                sigma_d=best_params.get("sigma_d", 0.4),
                sigma_theta=best_params.get("sigma_theta", 0.2),
                alpha=best_params.get("alpha", 1.5),
            )
            for item in atoms_list
        ]

        # Compute Wasserstein distance and kernel
        if node_importance and act_doub_scale:
            D, _ = WWLAnalysis._pairwise_wasserstein(best_graphs, depth=best_params["wl_depth"], node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=best_params["ads_dist_threshold"])
        else:
            D, _ = WWLAnalysis._pairwise_wasserstein(best_graphs, depth=best_params["wl_depth"], node_attr_keys=node_attr_keys, node_importance=node_importance)
        sigma_val = best_params["sigma"]
        amplitude_val = best_params["amplitude"]
        kernel_params = [sigma_val, amplitude_val]
        if "extra_param" in best_params:
            kernel_params.append(best_params["extra_param"])
        D_kernel = KernelApproach._kernel_transform(D, kernel_params, type_kernel=type_kernel)

        # ------------------------------------------------
        # Fit final GPR using CV with best hyperparameters
        # ------------------------------------------------
        train_r2, train_mae = [], []
        val_r2, val_mae = [], []
        last_idx_train, last_idx_val = [], []
        y_pred_all = np.zeros(len(atoms_list))
        uncertainty_all = np.zeros(len(atoms_list))

        if idx_train is None and idx_val is None:
            fold_mae, tt_fold_mae = [], []
            for fold_idx_train, fold_idx_val in kf.split(np.arange(len(atoms_list))):
                # best_gpr = GaussianProcessRegressor(kernel=None, alpha=1e-4, normalize_y=True)
                best_gpr = ManualGPR(noise=best_params["noise"])
                best_gpr.fit(D_kernel[fold_idx_train][:, fold_idx_train], ads_energy_list[fold_idx_train])

                # --- Train-set prediction ---
                y_train_preds = best_gpr.predict(D_kernel[fold_idx_train][:, fold_idx_train])
                D_kernel_train = D_kernel[fold_idx_train][:, fold_idx_train]
                last_idx_train = fold_idx_train
                train_r2 = r2_score(ads_energy_list[fold_idx_train], y_train_preds)
                train_mae = mean_absolute_error(ads_energy_list[fold_idx_train], y_train_preds)

                # --- Validation-set prediction ---
                D_val = D[fold_idx_val][:, fold_idx_val]
                D_kernel_val = KernelApproach._kernel_transform(D_val, kernel_params, type_kernel=type_kernel)
                D_kernel_val_train = D_kernel[fold_idx_val][:, fold_idx_train]
                r2_ceiling = PredictionModel._kernel_r2_ceiling(D_kernel_train, D_kernel_val_train, ads_energy_list[fold_idx_train], ads_energy_list[fold_idx_val], best_params["noise"])
                y_val_preds, uncertainties = best_gpr.predict(D_kernel[fold_idx_val][:, fold_idx_train], K_self=D_kernel_val, return_std=True)
                y_pred_all[fold_idx_val] = y_val_preds
                uncertainty_all[fold_idx_val] = uncertainties
                last_idx_val = fold_idx_val
                val_r2 = r2_score(ads_energy_list[fold_idx_val], y_val_preds)
                val_mae = mean_absolute_error(ads_energy_list[fold_idx_val], y_val_preds)
                fold_mae.append((train_mae + val_mae) / 2)
                tt_fold_mae.append((train_mae, val_mae))
            
            average_mae = np.mean(fold_mae)
            low_mae = np.min(fold_mae)
            high_mae = np.max(fold_mae)
            std_mae = np.std(fold_mae)

            global_cv_r2 = r2_score(ads_energy_list, y_pred_all)
            global_r2 = global_cv_r2
            global_cv_mae = mean_absolute_error(ads_energy_list, y_pred_all)
            global_mae = global_cv_mae
        else:
            # best_gpr = GaussianProcessRegressor(kernel=None, alpha=1e-4, normalize_y=True)
            best_gpr = ManualGPR(noise=best_params["noise"])
            best_gpr.fit(D_kernel[idx_train][:, idx_train], ads_energy_list[idx_train])

            # --- Train-set prediction ---
            D_train = D[idx_train][:, idx_train]
            D_kernel_train = KernelApproach._kernel_transform(D_train, kernel_params, type_kernel=type_kernel)
            y_train_preds, uncertainties_train = best_gpr.predict(D_kernel[idx_train][:, idx_train], K_self=D_kernel_train, return_std=True)
            y_pred_all[idx_train] = y_train_preds
            uncertainty_all[idx_train] = uncertainties_train
            train_r2 = r2_score(ads_energy_list[idx_train], y_train_preds)
            train_mae = mean_absolute_error(ads_energy_list[idx_train], y_train_preds)
            D_kernel_train = D_kernel[idx_train][:, idx_train]

            # --- Validation-set prediction ---
            D_val = D[idx_val][:, idx_val]
            D_kernel_val = KernelApproach._kernel_transform(D_val, kernel_params, type_kernel=type_kernel)
            D_kernel_val_train = D_kernel[idx_val][:, idx_train]
            r2_ceiling = PredictionModel._kernel_r2_ceiling(D_kernel_train, D_kernel_val_train, ads_energy_list[idx_train], ads_energy_list[idx_val], best_params["noise"])
            y_val_preds, uncertainties_val = best_gpr.predict(D_kernel[idx_val][:, idx_train], K_self=D_kernel_val, return_std=True)
            y_pred_all[idx_val] = y_val_preds
            uncertainty_all[idx_val] = uncertainties_val
            val_r2 = r2_score(ads_energy_list[idx_val], y_val_preds)
            val_mae = mean_absolute_error(ads_energy_list[idx_val], y_val_preds)

            last_idx_train = idx_train
            last_idx_val = idx_val

            global_r2 = r2_score(ads_energy_list, y_pred_all)
            global_mae = mean_absolute_error(ads_energy_list, y_pred_all)
        
        # -------------------------------
        # Parity plot and kernel spectrum
        # -------------------------------
        if plot_name is not None:
            # === Get caller directory ===
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            if save_folder is None:
                plot_path = os.path.join(caller_dir, "model_performance")
            else:
                plot_path = os.path.join(caller_dir, "model_performance", save_folder)
            os.makedirs(plot_path, exist_ok=True)

            if idx_train is None and idx_val is None:
                # CV plot
                plt.figure(figsize=(6,6))
                plt.scatter(ads_energy_list, y_pred_all, color='purple', alpha=0.6)
                plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                     [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
                plt.title("GPR Parity Plot CV")
                plt.xlabel("Actual Adsorption Energy")
                plt.ylabel("Predicted Adsorption Energy")

                plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), f"CV R² (mean) = {global_cv_r2:.3f}\nCV MAE (mean) = {global_cv_mae:.3f}",
                     fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
                plt.tight_layout()

                # Save the plot to where the function was called
                ext_cv = f"_{plot_name}_{type_kernel}_imp_node" if node_importance is True else f"_{plot_name}_{type_kernel}"
                plt.savefig(os.path.join(plot_path, f"GPR_CV_parity_plot{ext_cv}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                # Train validation per each fold error histogram plot
                # Extract train/validation MAE per fold
                train_mae_list = [x[0] for x in tt_fold_mae]
                val_mae_list = [x[1] for x in tt_fold_mae]

                n_folds = len(train_mae_list)
                folds = np.arange(1, n_folds + 1)

                # ---- Plot ----
                width = 0.35  # bar width

                plt.figure(figsize=(10, 6))
                plt.bar(folds - width/2, train_mae_list, width=width, color="blue", label="Train MAE")
                plt.bar(folds + width/2, val_mae_list, width=width, color="orange", label="Validation MAE")

                plt.xlabel("Fold")
                plt.ylabel("MAE")
                plt.title("Train vs Validation MAE per Fold")
                plt.xticks(folds)
                plt.legend()

                # ---- Statistics text ----
                stats_text = (
                    f"Average MAE: {average_mae:.2f}\n"
                    f"Std MAE: {std_mae:.3f}\n"
                    f"Min MAE: {low_mae:.2f}\n"
                    f"Max MAE: {high_mae:.2f}"
                )

                plt.gca().text(
                    0.02, 0.98,
                    stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
            
                plt.tight_layout()
                plt.savefig(os.path.join(plot_path, f"mae_per_fold{ext_cv}.png"), dpi=300)
                plt.close()

                df_results = pd.DataFrame({
                    "Fold": folds,
                    "Train_MAE": train_mae_list,
                    "Validation_MAE": val_mae_list
                })

                df_stats = pd.DataFrame({
                    "Average_MAE": [average_mae],
                    "Std_MAE": [std_mae],
                    "Min_MAE": [low_mae],
                    "Max_MAE": [high_mae]
                })

                # Save results to CSV
                df_results.to_csv(os.path.join(plot_path, f"mae_per_fold_{plot_name}_{type_kernel}.csv"), index=False)
                df_stats.to_csv(os.path.join(plot_path, f"mae_stats_{plot_name}_{type_kernel}.csv"), index=False)
                
                print(f"[INFO] Final GPR model trained: CV R²={global_cv_r2:.3f}, MAE={global_cv_mae:.3f}")

            # Train-validation set
            plt.figure(figsize=(6, 6))
            plt.scatter(ads_energy_list[last_idx_train], y_train_preds, color='green', alpha=0.6, label='Train')
            plt.scatter(ads_energy_list[last_idx_val], y_val_preds, color='orange', alpha=0.6, label='Validation')
            plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
            plt.title("GPR Parity Plot Train-Validation")
            plt.xlabel("Actual Adsorption Energy")
            plt.ylabel("Predicted Adsorption Energy")

            # Add last-fold metrics as text
            textstr = (
                f"Train R²  = {train_r2:.3f}\n"
                f"Train MAE = {train_mae:.3f}\n"
                f"Validation R²   = {val_r2:.3f}\n"
                f"Validation MAE  = {val_mae:.3f}\n"
                f"Validation R² Ceiling = {r2_ceiling:.3f}"
            )

            plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), textstr,
                     fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(loc='center left')
            plt.tight_layout()

            # Save the plot
            ext_tt = f"_{plot_name}_{type_kernel}_imp_node" if node_importance is True else f"_{plot_name}_{type_kernel}"
            plt.savefig(os.path.join(plot_path, f"GPR_tt_parity_plot{ext_tt}.png"),
                    dpi=300, bbox_inches='tight')
            plt.close()

            # Hyperparameter importance
            ext_hyp = f"GPR_{plot_name}_{type_kernel}_imp_node" if node_importance is True else f"GPR_{plot_name}_{type_kernel}"
            PredictionModel._plot_optuna_result(study, plot_path, out_prefix=ext_hyp)

            eigvals = np.linalg.eigvalsh(D_kernel)
            eigvals = np.sort(eigvals[::-1])
            explained_var = np.cumsum(eigvals) / np.sum(eigvals)
            ext_kern = f"GPR_{plot_name}_{type_kernel}_imp_node" if node_importance is True else f"GPR_{plot_name}_{type_kernel}"

            plt.figure(figsize=(6,4))
            plt.plot(eigvals, marker='o', linestyle='-', color='purple')
            plt.yscale('log')
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue (log scale)")
            plt.title(f"Kernel Matrix Eigenvalues - {type_kernel}")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f"{ext_kern}_kernel_spectrum.png"), dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(explained_var, marker='o', linestyle='-', color='purple')
            plt.xlabel("Number of Eigenvalues")
            plt.ylabel("Cumulative Explained Variance")
            plt.title(f"Cumulative Explained Variance - {type_kernel}")
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f"{ext_kern}_explained_variance.png"), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"[INFO] Final GPR model trained: Train-Validation R²={train_r2:.3f} train and {val_r2:.3f} validation, MAE={train_mae:.3f} train and {val_mae:.3f} validation")

        # Save GPR model
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        if save_folder is None:
            model_path = os.path.join(caller_dir, "saved_models")
        else:
            model_path = os.path.join(caller_dir, "saved_models", save_folder)
        os.makedirs(model_path, exist_ok=True)
        model_name = f"GPR_{type_kernel}_{plot_name}_imp_node" if node_importance is True else f"GPR_{type_kernel}_{plot_name}"
        save_path = os.path.join(model_path, model_name + ".pkl")
        data_to_save = {
            "model": best_gpr,
            "type_kernel": type_kernel,
            "best_params": best_params,
            "kernel_params": kernel_params,
            "all_graph": best_graphs,
            "ref_graphs": [best_graphs[i] for i in last_idx_train],
            "train_idx": last_idx_train,
            "node_importance": node_importance,
            "act_doub_scale": act_doub_scale,
            "adj_ads_desc": adj_ads_desc,
            "node_attr_keys": node_attr_keys if node_attr_keys is not None else None,
            "gas_db": gas_db if gas_db is not None else None
        }
        with open(save_path, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"[INFO] GPR model and metadata saved to {save_path}")

        return {
            "model": best_gpr,
            "best_params": best_params,
            "all_graphs": best_graphs,
            "global_r2": global_r2,
            "global_mae": global_mae,
            "y_pred_all": y_pred_all,
            "uncertainty_all": uncertainty_all,
            "train_idx": last_idx_train,
            "validation_idx": last_idx_val
        }
    
    @staticmethod
    def _nn_ads_energy(atoms_list, ads_energy_list, n_trials=20, CV=5,
                       epochs=200, ads_dist_threshold=None, wl_depth=None, lr=None, hidden1=None, hidden2=None,
                       node_importance=False, adj_ads_desc=False, gas_db=None, tune_pertub_param=False,
                       plot_name=None):
        """
        Train NN using WL embedding. Hyperparameters (ads_dist_threshold, learning_rate, hidden_dimension)
        are tuned using Optuna if not provided. Manual CV using a single train/test split.

        Returns:
            model: trained NN
            best_params: dict of best hyperparameters
            best_graphs: list of networkx graphs for each structure
            cv_r2: model performance
            cv_mae: model performance
            y_pred_all: predicted adsorption energy
        """
        ads_energy_list = np.array(ads_energy_list)
        y_tensor = torch.tensor(ads_energy_list, dtype=torch.float32, device=device)
        kf = KFold(n_splits=CV, shuffle=True, random_state=42)

        def objective(trial):
            # Suggest hyperparameters if not fixed
            adt = ads_dist_threshold or trial.suggest_float('ads_dist_threshold', 2.5, 4.5)
            depth = wl_depth or trial.suggest_int('wl_depth', 1, 3)
            hid1 = hidden1 or trial.suggest_int("hidden1", 128, 512)
            hid2 = hidden2 or trial.suggest_int("hidden2", 64, 256)
            lr_trial = lr or trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            
            # ---- Perturbation parameters (OPTIONAL) ----
            if adj_ads_desc:
                if tune_pertub_param:
                    beta = trial.suggest_float("beta", 0.0, 1.5)
                    lambda_corr = trial.suggest_float("lambda_corr", 0.0, 0.6)
                    sigma_d = trial.suggest_float("sigma_d", 0.2, 0.8)
                    sigma_theta = trial.suggest_float("sigma_theta", 0.1, 0.6)
                    alpha = trial.suggest_float("alpha", 0.5, 3.0)
                else:
                    beta = 0.5
                    lambda_corr = 0.3
                    sigma_d = 0.4
                    sigma_theta = 0.2
                    alpha = 1.5
                
                # store for later retrieval
                trial.set_user_attr("beta", beta)
                trial.set_user_attr("lambda_corr", lambda_corr)
                trial.set_user_attr("sigma_d", sigma_d)
                trial.set_user_attr("sigma_theta", sigma_theta)
                trial.set_user_attr("alpha", alpha)
            else:
                # adsorbate descriptor adjustment OFF → parameters irrelevant
                beta = lambda_corr = sigma_d = sigma_theta = alpha = None

            # Build graphs
            graphs = [
                GraphTransform._make_graph(
                    atoms_init=item["contcar"],
                    adsorbate_label=item["ads_label"],
                    ads_dist_threshold=adt,
                    descriptor_df=item["descriptors"],
                    adj_ads_desc=adj_ads_desc,
                    gas_db=gas_db,
                    beta=beta,
                    lambda_corr=lambda_corr,
                    sigma_d=sigma_d,
                    sigma_theta=sigma_theta,
                    alpha=alpha
                )
                for item in atoms_list
            ]
            embedding = [
                torch.tensor(
                    WWLAnalysis._wl_node_embedding(G, depth=depth, node_importance=node_importance),
                    dtype=torch.float32,
                    device=device,
                )
                for G in graphs
            ]

            input_dim = embedding[0].shape[1]
            indices = np.arange(len(embedding))
            preds_global = []
            trues_global = []
            for train_idx, test_idx in kf.split(indices):
                # Define node-wise MLP
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hid1),
                    nn.ReLU(),
                    nn.Linear(hid1, hid2),
                    nn.ReLU(),
                    nn.Linear(hid2, 1),
                ).to(device)

                optimizer = optim.Adam(mlp.parameters(), lr=lr_trial)
                loss_fn = nn.MSELoss()

                # Training
                for _ in range(epochs):
                    mlp.train()
                    optimizer.zero_grad()
                    loss = 0.0
                    for i in train_idx:
                        graph_pred = mlp(embedding[i]).mean()     # scalar
                        loss += loss_fn(graph_pred, y_tensor[i])
                    loss.backward()
                    optimizer.step()

                # Evaluation
                mlp.eval()
                with torch.no_grad():
                    for i in test_idx:
                        preds_global.append(mlp(embedding[i]).mean().item())
                        trues_global.append(y_tensor[i].item())
                        
            global_mae = mean_absolute_error(trues_global, preds_global)
            return float(global_mae)
        
        # ----------------------------
        # Hyperparameter optimization
        # ----------------------------
        if ads_dist_threshold is None or wl_depth is None or lr is None or hidden1 is None or hidden2 is None:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_trial.params.copy()
            
            # ---- Perturbation params (ONLY if used) ----
            if adj_ads_desc:
                for key in ["beta", "lambda_corr", "sigma_d", "sigma_theta", "alpha"]:
                    best_params[key] = study.best_trial.user_attrs[key]

            if ads_dist_threshold is not None:
                best_params["ads_dist_threshold"] = ads_dist_threshold
            if wl_depth is not None:
                best_params["wl_depth"] = wl_depth
            if lr is not None:
                best_params["lr"] = lr
            if hidden1 is not None:
                best_params["hidden1"] = hidden1
            if hidden2 is not None:
                best_params["hidden2"] = hidden2
        else:
            # All hyperparameters fixed, just train once
            best_params = {
                "ads_dist_threshold": ads_dist_threshold,
                "wl_depth": wl_depth,
                "lr": lr,
                "hidden1": hidden1,
                "hidden2": hidden2
            }
        
        # Rebuild graphs + embeddings with best hyperparameters
        best_graphs = [
            GraphTransform._make_graph(
                atoms_init=item["contcar"],
                adsorbate_label=item["ads_label"],
                ads_dist_threshold=best_params["ads_dist_threshold"],
                descriptor_df=item["descriptors"],
                adj_ads_desc=adj_ads_desc,
                gas_db=gas_db,
                beta=best_params.get("beta", 0.5),
                lambda_corr=best_params.get("lambda_corr", 0.3),
                sigma_d=best_params.get("sigma_d", 0.4),
                sigma_theta=best_params.get("sigma_theta", 0.2),
                alpha=best_params.get("alpha", 1.5),
            )
            for item in atoms_list
        ]

        best_embeddings = [
            torch.tensor(
                WWLAnalysis._wl_node_embedding(G, depth=best_params["wl_depth"], node_importance=node_importance),
                dtype=torch.float32,
                device=device,
            )
            for G in best_graphs
        ]

        input_dim = best_embeddings[0].shape[1]
        indices = np.arange(len(best_embeddings))
        y_pred_all = np.zeros(len(best_embeddings))

        last_train_preds, last_test_preds = None, None
        last_train_true, last_test_true = None, None
        last_model = None

        for train_idx, test_idx in kf.split(indices):
            # Define node-wise MLP
            best_model = nn.Sequential(
                nn.Linear(input_dim, best_params["hidden1"]),
                nn.ReLU(),
                nn.Linear(best_params["hidden1"], best_params["hidden2"]),
                nn.ReLU(),
                nn.Linear(best_params["hidden2"], 1),
            ).to(device)

            optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
            loss_fn = nn.MSELoss()

            # Training
            for _ in range(epochs):
                best_model.train()
                optimizer.zero_grad()
                loss = 0.0
                for i in train_idx:
                    graph_pred = best_model(best_embeddings[i]).mean()     # scalar
                    loss += loss_fn(graph_pred, y_tensor[i])
                loss.backward()
                optimizer.step()

            # Evaluation
            best_model.eval()
            train_preds, train_true = [], []
            test_preds, test_true = [], []

            with torch.no_grad():
                # Train predictions (only for last-fold diagnostics)
                for i in train_idx:
                    pred = best_model(best_embeddings[i]).mean().item()
                    train_preds.append(pred)
                    train_true.append(ads_energy_list[i])
                
                # Test predictions (used for CV)
                for i in test_idx:
                    pred = best_model(best_embeddings[i]).mean().item()
                    y_pred_all[i] = pred
                    test_preds.append(pred)
                    test_true.append(ads_energy_list[i])

            last_model = best_model
            last_train_preds, last_test_preds = train_preds, test_preds
            last_train_true, last_test_true = train_true, test_true
        
        cv_mae = mean_absolute_error(ads_energy_list, y_pred_all)
        cv_r2 = r2_score(ads_energy_list, y_pred_all)

        train_mae = mean_absolute_error(last_train_true, last_train_preds)
        train_r2 = r2_score(last_train_true, last_train_preds)

        test_mae = mean_absolute_error(last_test_true, last_test_preds)
        test_r2 = r2_score(last_test_true, last_test_preds)

        # Plots
        if plot_name is not None:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            plot_path = os.path.join(caller_dir, "model_performance")
            os.makedirs(plot_path, exist_ok=True)

            # CV plot
            plt.figure(figsize=(6,6))
            plt.scatter(ads_energy_list, y_pred_all, color='purple', alpha=0.6)
            plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                 [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
            plt.title("Neural Network Parity Plot CV")
            plt.xlabel("Actual Adsorption Energy")
            plt.ylabel("Predicted Adsorption Energy")

            plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), f"CV R² (mean) = {cv_r2:.3f}\nCV MAE (mean) = {cv_mae:.3f}",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
            plt.tight_layout()

            # Save the plot to where the function was called
            ext_cv = f"_{plot_name}_imp_node" if node_importance is True else f"_{plot_name}"
            plt.savefig(os.path.join(plot_path, f"NN_CV_parity_plot{ext_cv}.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # ---- Last fold train/test parity ----
            # Train-test set
            plt.figure(figsize=(6, 6))
            plt.scatter(last_train_true, last_train_preds, color='green', alpha=0.6, label='Train')
            plt.scatter(last_test_true, last_test_preds, color='red', alpha=0.6, label='Test')
            plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
            plt.title("Neural Network Parity Plot Train-test")
            plt.xlabel("Actual Adsorption Energy")
            plt.ylabel("Predicted Adsorption Energy")

            # Add last-fold metrics as text
            textstr = (
                f"Train R²  = {train_r2:.3f}\n"
                f"Train MAE = {train_mae:.3f}\n"
                f"Test R²   = {test_r2:.3f}\n"
                f"Test MAE  = {test_mae:.3f}"
            )

            plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), textstr,
                     fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(loc='center left')
            plt.tight_layout()

            # Save the plot
            ext_tt = f"_{plot_name}_imp_node" if node_importance is True else f"_{plot_name}"
            plt.savefig(os.path.join(plot_path, f"NN_tt_parity_plot{ext_tt}.png"),
                    dpi=300, bbox_inches='tight')
            plt.close()

            # Hyperparameter importance
            ext_hyp = f"NN_{plot_name}_imp_node" if node_importance is True else f"NN_{plot_name}"
            PredictionModel._plot_optuna_result(study, plot_path, out_prefix=ext_hyp)
        
        print(f"[INFO] Final NN model trained: CV R²={cv_r2:.3f}, MAE={cv_mae:.3f}")
        print(f"[INFO] Final NN model trained: Train-Test R²={train_r2:.3f} train and {test_r2:.3f} test, MAE={train_mae:.3f} train and {test_mae:.3f} test")

        # Save NN model
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        model_path = os.path.join(caller_dir, "saved_models")
        os.makedirs(model_path, exist_ok=True)
        model_name = f"NN_{plot_name}_imp_node" if node_importance is True else f"NN_{plot_name}"
        save_path = os.path.join(model_path, model_name + ".pt")
        checkpoint  = {
            "model_state_dict": last_model.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "hidden1": best_params["hidden1"],
                "hidden2": best_params["hidden2"]
            },
            "best_params": best_params,
            "node_importance": node_importance,
            "adj_ads_desc": adj_ads_desc,
            "gas_db": gas_db if gas_db is not None else None
        }
        torch.save(checkpoint, save_path)
        print(f"[INFO] NN model and metadata saved to {save_path}")

        return {
            "model": last_model,
            "best_params": best_params,
            "best_graphs": best_graphs,
            "cv_r2": cv_r2,
            "cv_mae": cv_mae,
            "y_pred_all": y_pred_all,
        }
    
    @staticmethod
    def _run_gpr_model(model_path, atom, ads_label, descriptors):
        """
        Predict adsorption energies using GPR trained model.
    
        Args:
            model_path: trained GPR model (pickle format)
            atom : ASE Atoms object representing the atomic structure of the system
            ads_label : label identifying the adsorbate species in the structure
            descriptors : Precomputed atomic or structural descriptors to be attached to
                nodes during graph construction. Must match the descriptor format
                used during model training.
    
        Returns:
            pred_val: predicted adsorption energy
        """
        # Load saved GPR model and metadata
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        # Extract info
        gpr_model = saved_data["model"]
        ref_graph_list = saved_data["ref_graphs"]
        best_params = saved_data["best_params"]
        type_kernel = saved_data["type_kernel"]
        kernel_params = saved_data["kernel_params"]
        node_importance = saved_data["node_importance"]
        act_doub_scale = saved_data["act_doub_scale"]
        adj_ads_desc=saved_data["adj_ads_desc"]
        node_attr_keys = saved_data.get("node_attr_keys", None)
        gas_db=saved_data["gas_db"]

        # Extract perturbation parameters if they exist, otherwise use defaults
        beta = best_params.get("beta", 0.5)
        lambda_corr = best_params.get("lambda_corr", 0.3)
        sigma_d = best_params.get("sigma_d", 0.4)
        sigma_theta = best_params.get("sigma_theta", 0.2)
        alpha = best_params.get("alpha", 1.5)

        # Build graph for inference
        graph = GraphTransform._make_graph(
            atoms_init=atom,
            adsorbate_label=ads_label,
            ads_dist_threshold=best_params["ads_dist_threshold"],
            descriptor_df=descriptors,
            adj_ads_desc=adj_ads_desc,
            gas_db=gas_db,
            inner_weight=best_params["inner_weight"],
            outer_weight=best_params["outer_weight"],
            beta=beta,
            lambda_corr=lambda_corr,
            sigma_d=sigma_d,
            sigma_theta=sigma_theta,
            alpha=alpha
        )
        
        # Compute Wasserstein distances between the eval graph and reference graphs
        D_opt = WWLAnalysis._wasserstein_calc(
            [graph],
            ref_graph_list,
            depth=best_params["wl_depth"],
            node_importance=node_importance,
            node_attr_keys=node_attr_keys,
            ads_dist_threshold=best_params["ads_dist_threshold"] if act_doub_scale is True else None
        )
        D_kernel_opt = KernelApproach._kernel_transform(D_opt, kernel_params, type_kernel=type_kernel)

        # Compute Wasserstein distances of the eval graph
        D_self = WWLAnalysis._wasserstein_calc(
            [graph],
            [graph],
            depth=best_params["wl_depth"],
            node_importance=node_importance,
            node_attr_keys=node_attr_keys,
            ads_dist_threshold=best_params["ads_dist_threshold"] if act_doub_scale is True else None
        )
        D_kernel_self = KernelApproach._kernel_transform(D_self, kernel_params, type_kernel=type_kernel)

        pred_val, uncertainty = gpr_model.predict(D_kernel_opt, K_self=D_kernel_self, return_std=True)
        
        return float(pred_val[0]), float(uncertainty[0])
    
    @staticmethod
    def _run_nn_model(model_path, atom, ads_label, descriptors):
        """
        Predict adsorption energies using NN trained model.
    
        Args:
            model_path: trained NN model (torch checkpoint format)
            atom : ASE Atoms object representing the atomic structure of the system
            ads_label : label identifying the adsorbate species in the structure
            descriptors : Precomputed atomic or structural descriptors to be attached to
                nodes during graph construction. Must match the descriptor format
                used during model training.
    
        Returns:
            pred_val: predicted adsorption energy
        """
        # Extract info
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        model_config = checkpoint["model_config"]
        input_dim = model_config["input_dim"]
        best_params = checkpoint["best_params"]
        node_importance = checkpoint["node_importance"]
        adj_ads_desc = checkpoint["adj_ads_desc"]
        gas_db = checkpoint["gas_db"]
        
        # Extract perturbation parameters if they exist, otherwise use defaults
        beta = best_params.get("beta", 0.5)
        lambda_corr = best_params.get("lambda_corr", 0.3)
        sigma_d = best_params.get("sigma_d", 0.4)
        sigma_theta = best_params.get("sigma_theta", 0.2)
        alpha = best_params.get("alpha", 1.5)

        # Recreate model architecture
        model = nn.Sequential(
            nn.Linear(input_dim, model_config["hidden1"]),
            nn.ReLU(),
            nn.Linear(model_config["hidden1"], model_config["hidden2"]),
            nn.ReLU(),
            nn.Linear(model_config["hidden2"], 1),
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        graph = GraphTransform._make_graph(
            atoms_init=atom,
            adsorbate_label=ads_label,
            ads_dist_threshold=best_params["ads_dist_threshold"],
            descriptor_df=descriptors,
            adj_ads_desc=adj_ads_desc,
            gas_db=gas_db,
            beta=beta,
            lambda_corr=lambda_corr,
            sigma_d=sigma_d,
            sigma_theta=sigma_theta,
            alpha=alpha
        )

        # Compute WL embedding
        embedding = torch.tensor(
            WWLAnalysis._wl_node_embedding(graph, depth=best_params["wl_depth"], node_importance=node_importance),
            dtype=torch.float32,
            device=device
        )

        # Predict scalar energy
        with torch.no_grad():
            pred_val = model(embedding).mean().item()

        return float(pred_val)