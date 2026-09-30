import os
import inspect
from xml.parsers.expat import model

from numpy.random.mtrand import gamma
import optuna
import numpy as np
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.importance import get_param_importances
from optuna.visualization.matplotlib import plot_optimization_history
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool
from Genox import GraphTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert NetworkX graph to PyTorch Geometric Data object
def nx_to_pyg(graph, target=None, node_importance=False, node_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Parameters
    ----------
    graph : networkx.Graph
    target : float
    node_importance : bool
        If True, only nodes with node_category == 1 are used
    node_attr_keys : list or None
        Explicit node attributes to use. If None, infer automatically.

    Returns
    -------
    torch_geometric.data.Data
    """
    def adjust_features(graph, nodes, n_nodes, node_features=None):
        # Exclude unimportant features
        exclude_keys = {"cell", "pbc", "position", "is_adsorbate", "base_weight", "node_category"}
        
        # Adjust node features
        sample_attrs = graph.nodes[nodes[0]]
        if node_features is None:
            node_features = [
                k for k, v in sample_attrs.items()
                if isinstance(v, (int, float, np.floating))
                and k not in exclude_keys
            ]

        X = np.zeros((n_nodes, len(node_features)), dtype=float)
        for i, v in enumerate(nodes):
            for j, key in enumerate(node_features):
                X[i, j] = graph.nodes[v].get(key, 0.0)

        x = torch.tensor(X, dtype=torch.float32)

        return x, node_features

    # Select nodes (WITHOUT recreating graph)
    if not node_importance:
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        # Adjust node features
        x, selected_features = adjust_features(graph, nodes, n_nodes, node_features=node_features)

        # Edge geometry unchanged
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    else:
        nodes = [v for v, d in graph.nodes(data=True) if d.get("node_category", 0) == 1]
        if len(nodes) == 0:
            raise ValueError("No nodes left after node_importance filtering.")

        node_to_idx = {v: i for i, v in enumerate(nodes)}
        n_nodes = len(nodes)

        # Adjust node features
        x, selected_features = adjust_features(graph, nodes, n_nodes, node_features=node_features)

        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
        if len(edges) == 0:
            raise ValueError("No edges left after node_importance filtering.")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float32)

    return data, selected_features

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Graph Neural Network Model
class GraphNN(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, conv_type="GraphSAGE", depth=1, dropout=0.2):
        """
        Graph Neural Network with flexible depth and hidden layers.

        Parameters
        ----------
        input_dim : int
            Number of input node features.
        hidden1 : int
            Hidden dimension of the first hidden layer after initial embedding.
        hidden2 : int
            Hidden dimension of subsequent hidden layers.
        conv_type : str
            Type of GNN layer: 'GCN', 'GIN', 'GraphSAGE', 'GAT'.
        depth : int
            Number of message-passing steps (neighbor aggregation). If depth=1, only direct neighbors.
        """
        super().__init__()
        conv_types = {"GCN": GCNConv, "GIN": GINConv, "GraphSAGE": SAGEConv, "GAT": GATConv}
        if conv_type not in conv_types:
            raise ValueError(f"conv_type must be one of {list(conv_types.keys())}")
        ConvLayer = conv_types[conv_type]

        # GELs
        self.embed_layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Initial embedding layers (message passing, depth)
        for _ in range(depth):
            if conv_type == "GIN":
                nn_seq = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.ReLU(),
                    nn.Linear(prev_dim, prev_dim)
                )
                self.embed_layers.append(ConvLayer(nn_seq))
            elif conv_type == "GAT":
                self.embed_layers.append(ConvLayer(prev_dim, prev_dim, heads=1))
            else:
                self.embed_layers.append(ConvLayer(prev_dim, prev_dim))
        
        # Hidden layers
        self.hidden1 = nn.Linear(prev_dim, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)

        # Final linear layer for output
        self.lin = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # Initial embedding + message passing
        for conv in self.embed_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)

        # Hidden layers
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Pooling and output
        x = global_mean_pool(x, batch)
        out = self.lin(x)
        return out.view(-1)
    
class GNNPredModel:
    @staticmethod
    def _evaluate_model(dataset, batch_size, hidden1, hidden2, 
                        conv_type, depth, dropout, lr, device, 
                        scheduler_choice, epochs, final_eval=False, return_model=False,
                        idx_train=None, idx_val=None, kf =None,
                        trial=None, scheduler_params=None):
        input_dim = dataset[0].x.shape[1]

        def build_scheduler(optimizer, scheduler_choice, trial=None, epochs=None, scheduler_params=None):
            if scheduler_choice == "StepLR":
                if trial is not None and scheduler_params is None:
                    step_size = trial.suggest_int("step_size", 10, 100)
                    gamma = trial.suggest_float("gamma", 0.1, 0.9)
                    trial.set_user_attr("step_size", step_size)
                    trial.set_user_attr("gamma", gamma)
                else:
                    step_size = scheduler_params.get("step_size")
                    gamma = scheduler_params.get("gamma")
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_choice == "ExponentialLR":
                if trial is not None and scheduler_params is None:
                    gamma = trial.suggest_float("gamma", 0.9, 0.999)
                    trial.set_user_attr("gamma", gamma)
                else:
                    gamma = scheduler_params.get("gamma")
                return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            elif scheduler_choice == "CosineAnnealingLR":
                if trial is not None and scheduler_params is None:
                    T_max = trial.suggest_int("T_max", 10, epochs)
                    trial.set_user_attr("T_max", T_max)
                else:
                    T_max = scheduler_params.get("T_max")
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        def make_loader(indices, shuffle):
            g = torch.Generator()
            g.manual_seed(42)

            return DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g
            )
        
        def train_one_split(train_idx, val_idx, final_eval=False):
            set_seed(42)
            
            train_loader = make_loader(train_idx, True)
            val_loader   = make_loader(val_idx, False)

            model = GraphNN(input_dim, hidden1=hidden1, hidden2=hidden2, conv_type=conv_type, depth=depth, dropout=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = build_scheduler(optimizer=optimizer, scheduler_choice=scheduler_choice, trial=trial, epochs=epochs, scheduler_params=scheduler_params)
            loss_fn = nn.MSELoss()

            for _ in range(epochs):
                # Training
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = loss_fn(out, batch.y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Evaluation
            model.eval()
            if final_eval:
                preds_all = np.zeros(len(dataset))
                trues_all = np.zeros(len(dataset))
                with torch.no_grad():
                    # Fill BOTH train + val
                    for loader in [train_loader, val_loader]:
                        for batch in loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.batch)
                            idx = batch.idx.cpu().numpy()
                            preds_all[idx] = out.cpu().numpy()
                            trues_all[idx] = batch.y.cpu().numpy()
                return model, preds_all, trues_all, None
            else:
                preds_eval, trues_eval, idx_eval = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        preds_eval.extend(out.cpu().numpy())
                        trues_eval.extend(batch.y.cpu().numpy())
                        idx_eval.extend(batch.idx.cpu().numpy())
                return model, np.array(preds_eval), np.array(trues_eval), np.array(idx_eval)
        
        # CV mode
        if kf is not None and idx_train is None and idx_val is None:
            preds_global = np.zeros(len(dataset), dtype=np.float32)
            trues_global = np.zeros(len(dataset), dtype=np.float32)
            indices = np.arange(len(dataset))

            for train_idx, val_idx in kf.split(indices):
                _, preds, trues, corr_idx = train_one_split(train_idx, val_idx)
                preds_global[corr_idx] = preds
                trues_global[corr_idx] = trues
            mae = mean_absolute_error(trues_global, preds_global)
        else:
            model, preds, trues, _ = train_one_split(idx_train, idx_val, final_eval=final_eval)
            mae = mean_absolute_error(trues, preds)
            preds_global, trues_global = preds, trues
        
        if final_eval:
            if return_model:
                return model, mae, preds_global, trues_global
            else:
                return mae, preds_global, trues_global
        else:
            return mae

    @staticmethod
    def _plot_optuna_result(study, plot_path, out_prefix=None):
        """
        Plot hyperparameter importance from an Optuna study.

        Parameters:
        -----------
        study : optuna.Study
            The Optuna study object (after optimization).
        out_prefix : str or None
            Path prefix for saving PNGs.
        """
        # Hyperparameter Importance
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                for k, v in t.user_attrs.items():
                    t.params[k] = v  # temporarily inject into params

        importances = get_param_importances(study)
        if len(importances) == 0:
            print("[INFO] No hyperparameters to show importance for.")
            return

        params = list(importances.keys())
        scores = list(importances.values())

        plt.figure(figsize=(6,4))
        plt.barh(params, scores, color='purple', alpha=0.7)
        plt.xlabel("Importance")
        plt.title(f"Hyperparameter Importance")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{out_prefix}_HI.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Optimization History
        ax = plot_optimization_history(study)
        ax.figure.savefig(os.path.join(plot_path, f"{out_prefix}_history.png"), dpi=300)
        plt.close()

    # GNN Model Creation
    @staticmethod
    def _gnn_ads_energy(atoms_list, ads_energy_list, n_trials=100, random_state=42, CV=None, tt_split=None, spec_idx_train=None, spec_idx_val=None,
                        epochs=200, ads_dist_threshold=None, depth=None, lr=None, hidden1=None, hidden2=None, dropout=None,
                        batch_size=None, scheduler_type=None, scheduler_params=None, conv_type="GCN",
                        node_importance=False, node_features=None, adj_ads_desc=False, gas_db=None, tune_pertub_param=False,
                        plot_name=None, save_folder=None):
        """
        Train GNN to predict adsorption energies. Hyperparameters (ads_dist_threshold, depth, lr, hidden1, hidden2, batch_size, scheduler_type)
        are tuned using Optuna if not provided. K-Fold cross-validation is used for evaluation.
    
        Returns:
            model: trained GNN
            best_params: dict of best hyperparameters
            best_graphs: list of networkx graphs for each structure
            cv_r2: model performance
            cv_mae: model performance
            y_pred_all: predicted adsorption energy
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

        def objective(trial):
            torch.manual_seed(42)
            np.random.seed(42)

            adt = ads_dist_threshold or trial.suggest_float("ads_dist_threshold", 2.8, 3.5)
            depth_emb = depth or trial.suggest_int("depth", 1, 3)
            hid1 = hidden1 or trial.suggest_int("hidden1", 128, 512)
            hid2 = hidden2 or trial.suggest_int("hidden2", 64, 256)
            drop = dropout or trial.suggest_float("dropout", 0.0, 0.5)
            lr_trial = lr or trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            batch_sz = batch_size or trial.suggest_int("batch_size", 8, 64, step=8)
            scheduler_choice = scheduler_type or trial.suggest_categorical(
                "scheduler_type", ["StepLR", "ExponentialLR", "CosineAnnealingLR"]
            )

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
            all_results = [nx_to_pyg(
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
                        ), target=ads_energy_list[i],
                        node_importance=node_importance,
                        node_features=node_features
                      )
                      for i, item in enumerate(atoms_list)]
            
            dataset, _ = zip(*all_results)
            dataset = list(dataset)
            for i, data in enumerate(dataset):
                data.idx = torch.tensor([i], dtype=torch.long)

            global_mae = GNNPredModel._evaluate_model(
                dataset=dataset, batch_size=batch_sz, hidden1=hid1, hidden2=hid2,
                conv_type=conv_type, depth=depth_emb, dropout=drop, lr=lr_trial,
                device=device, scheduler_choice=scheduler_choice, epochs=epochs,
                idx_train=idx_train, idx_val=idx_val, kf=kf, trial=trial
            )

            return float(global_mae)
    
        # ------------------------------
        # Hyperparameter Optimization
        # ------------------------------
        if any(param is None for param in [ads_dist_threshold, depth, lr, hidden1, hidden2, batch_size, scheduler_type]):
            n_startup = max(40, int(0.25 * n_trials))
            tpe_sampler = TPESampler(
                multivariate=True,
                group=True,
                n_startup_trials=n_startup,
                seed=42
            )
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{plot_name}_{int(time.time())}",
                load_if_exists=True,
                sampler=tpe_sampler
            )
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_trial.params.copy()
            
            # ---- Perturbation params (ONLY if used) ----
            if adj_ads_desc:
                for key in ["beta", "lambda_corr", "sigma_d", "sigma_theta", "alpha"]:
                    best_params[key] = study.best_trial.user_attrs[key]

            # ---- Scheduler params recovery ----
            recovered_scheduler_params = {}
            for key in ["step_size", "gamma", "T_max"]:
                if key in study.best_trial.user_attrs:
                    recovered_scheduler_params[key] = study.best_trial.user_attrs[key]

            if ads_dist_threshold is not None:
                best_params["ads_dist_threshold"] = ads_dist_threshold
            if depth is not None:
                best_params["depth"] = depth
            if dropout is not None:
                best_params["dropout"] = dropout
            if lr is not None:
                best_params["lr"] = lr
            if hidden1 is not None:
                best_params["hidden1"] = hidden1
            if hidden2 is not None:
                best_params["hidden2"] = hidden2
            if batch_size is not None:
                best_params["batch_size"] = batch_size
            if scheduler_type is not None:
                best_params["scheduler_type"] = scheduler_type
            if scheduler_params is not None:
                best_params["scheduler_params"] = scheduler_params
            else:
                best_params["scheduler_params"] = recovered_scheduler_params
        else:
            # All hyperparameters fixed, just train once
            best_params = {
                "ads_dist_threshold": ads_dist_threshold,
                "depth": depth,
                "dropout": dropout,
                "lr": lr,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "batch_size": batch_size,
                "scheduler_type": scheduler_type,
                "scheduler_params": scheduler_params
            }

        # Rebuild graphs and model with best hyperparameters
        all_results = [nx_to_pyg(
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
                    ), target=ads_energy_list[i],
                    node_importance=node_importance,
                    node_features=node_features
                  )
                  for i, item in enumerate(atoms_list)]
        
        dataset, features_list = zip(*all_results)
        selected_features = features_list[0]
        dataset = list(dataset)
        for i, data in enumerate(dataset):
            data.idx = torch.tensor([i], dtype=torch.long)

        input_dim = dataset[0].x.shape[1]
        indices = np.arange(len(dataset))
    
        last_train_preds, last_val_preds = None, None
        last_train_true, last_val_true = None, None

        scheduler_choice = best_params.get("scheduler_type")
    
        if idx_train is None and idx_val is None:
            global_mae, preds_global, trues_global = GNNPredModel._evaluate_model(
                dataset=dataset, batch_size=best_params["batch_size"], hidden1=best_params["hidden1"], hidden2=best_params["hidden2"],
                conv_type=conv_type, depth=best_params["depth"], dropout=best_params["dropout"], lr=best_params["lr"],
                device=device, scheduler_choice=scheduler_choice, epochs=epochs, final_eval=True,
                kf=kf, scheduler_params=best_params["scheduler_params"]
            )
            selected_fold = float("inf")
            for train_idx, val_idx in kf.split(indices):
                fold_model, fold_mae, preds_fold, trues_fold = GNNPredModel._evaluate_model(
                    dataset=dataset, batch_size=best_params["batch_size"], hidden1=best_params["hidden1"], hidden2=best_params["hidden2"],
                    conv_type=conv_type, depth=best_params["depth"], dropout=best_params["dropout"], lr=best_params["lr"],
                    device=device, scheduler_choice=scheduler_choice, epochs=epochs, final_eval=True, return_model=True,
                    idx_train=train_idx, idx_val=val_idx, scheduler_params=best_params["scheduler_params"]
                )
                if fold_mae < selected_fold:
                    selected_fold = fold_mae
                    last_train_preds, last_val_preds = preds_fold[train_idx], preds_fold[val_idx]
                    last_train_true, last_val_true = trues_fold[train_idx], trues_fold[val_idx]
                    best_model = fold_model
                    last_idx_train, last_idx_val = train_idx, val_idx
                    
            global_r2 = r2_score(trues_global, preds_global)
            train_mae = mean_absolute_error(last_train_true, last_train_preds)
            train_r2 = r2_score(last_train_true, last_train_preds)
            val_mae = mean_absolute_error(last_val_true, last_val_preds)
            val_r2 = r2_score(last_val_true, last_val_preds)
        else:
            best_model, global_mae, preds_global, trues_global = GNNPredModel._evaluate_model(
                dataset=dataset, batch_size=best_params["batch_size"], hidden1=best_params["hidden1"], hidden2=best_params["hidden2"],
                conv_type=conv_type, depth=best_params["depth"], dropout=best_params["dropout"], lr=best_params["lr"],
                device=device, scheduler_choice=scheduler_choice, epochs=epochs, final_eval=True, return_model=True,
                idx_train=idx_train, idx_val=idx_val, scheduler_params=best_params["scheduler_params"]
            )
            last_train_preds, last_val_preds = preds_global[idx_train], preds_global[idx_val]
            last_train_true, last_val_true = trues_global[idx_train], trues_global[idx_val]
            last_idx_train, last_idx_val = idx_train, idx_val

            global_r2 = r2_score(trues_global, preds_global)
            train_mae = mean_absolute_error(last_train_true, last_train_preds)
            train_r2 = r2_score(last_train_true, last_train_preds)
            val_mae = mean_absolute_error(last_val_true, last_val_preds)
            val_r2 = r2_score(last_val_true, last_val_preds)
        y_pred_all = preds_global

        # Plots
        if plot_name is not None:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            caller_dir = os.path.dirname(os.path.abspath(caller_file))
            if save_folder is None:
                plot_path = os.path.join(caller_dir, "model_performance")
            else:
                plot_path = os.path.join(caller_dir, "model_performance", save_folder)
            os.makedirs(plot_path, exist_ok=True)

            if idx_train is None and idx_val is None:
                # CV Plot
                plt.figure(figsize=(6,6))
                plt.scatter(ads_energy_list, y_pred_all, color='purple', alpha=0.6)
                plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                    [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
                plt.title("Graph Neural Network Parity Plot CV")
                plt.xlabel("Actual Adsorption Energy")
                plt.ylabel("Predicted Adsorption Energy")

                plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), f"CV R² (mean) = {global_r2:.3f}\nCV MAE (mean) = {global_mae:.3f}",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
                plt.tight_layout()

                # Save the plot to where the function was called
                ext_cv = f"_{plot_name}_{conv_type}_imp_node" if node_importance is True else f"_{plot_name}_{conv_type}"
                plt.savefig(os.path.join(plot_path, f"GNN_CV_parity_plot{ext_cv}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                print(f"[INFO] Final GNN model trained: CV R²={global_r2:.3f}, MAE={global_mae:.3f}")

            # ---- Last fold train/val parity ----
            # Train-val set
            plt.figure(figsize=(6, 6))
            plt.scatter(last_train_true, last_train_preds, color='green', alpha=0.6, label='Train')
            plt.scatter(last_val_true, last_val_preds, color='red', alpha=0.6, label='val')
            plt.plot([ads_energy_list.min(), ads_energy_list.max()],
                [ads_energy_list.min(), ads_energy_list.max()], 'k--', lw=1)
            plt.title("Graph Neural Network Parity Plot Train-val")
            plt.xlabel("Actual Adsorption Energy")
            plt.ylabel("Predicted Adsorption Energy")

            # Add last-fold metrics as text
            textstr = (
                f"Train R²  = {train_r2:.3f}\n"
                f"Train MAE = {train_mae:.3f}\n"
                f"val R²   = {val_r2:.3f}\n"
                f"val MAE  = {val_mae:.3f}"
            )

            plt.text(0.95*ads_energy_list.min(), 0.90*ads_energy_list.max(), textstr,
                    fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            plt.legend(loc='center left')
            plt.tight_layout()

            # Save the plot
            ext_tt = f"_{plot_name}_{conv_type}_imp_node" if node_importance is True else f"_{plot_name}_{conv_type}"
            plt.savefig(os.path.join(plot_path, f"GNN_tt_parity_plot{ext_tt}.png"),
                    dpi=300, bbox_inches='tight')
            plt.close()

            # Hyperparameter importance
            ext_hyp = f"GNN_{plot_name}_{conv_type}_imp_node" if node_importance is True else f"GNN_{plot_name}_{conv_type}"
            GNNPredModel._plot_optuna_result(study, plot_path, out_prefix=ext_hyp)

        print(f"[INFO] Final GNN model trained: Train-val R²={train_r2:.3f} train and {val_r2:.3f} val, MAE={train_mae:.3f} train and {val_mae:.3f} val")

        # Save GNN model
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        if save_folder is None:
            model_path = os.path.join(caller_dir, "saved_models")
        else:
            model_path = os.path.join(caller_dir, "saved_models", save_folder)
        os.makedirs(model_path, exist_ok=True)
        model_name = f"GNN_{conv_type}_{plot_name}_imp_node" if node_importance is True else f"GNN_{conv_type}_{plot_name}"
        save_path = os.path.join(model_path, model_name + ".pt")

        checkpoint ={
            # Model architecture config
            "model_state_dict": best_model.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "hidden1": best_params["hidden1"],
                "hidden2": best_params["hidden2"],
                "conv_type": conv_type,
                "depth": best_params["depth"],
                "dropout": best_params["dropout"],
            },
            # Training schema config
            "training_config": {
                "epochs": epochs,
                "random_state": random_state,
                "CV": CV,
                "tt_split": tt_split,
                "lr": best_params["lr"],
                "batch_size": best_params["batch_size"],
                "scheduler_type": best_params["scheduler_type"],
                "scheduler_params": best_params["scheduler_params"]
            },
            # Graph construction config
            "graph_config": {
                "ads_dist_threshold": best_params["ads_dist_threshold"],
                "node_importance": node_importance,
                "selected_features": selected_features,
                "adj_ads_desc": adj_ads_desc,
                "gas_db": gas_db if gas_db is not None else None,
                # perturbation params
                "beta": best_params.get("beta", 0.5),
                "lambda_corr": best_params.get("lambda_corr", 0.3),
                "sigma_d": best_params.get("sigma_d", 0.4),
                "sigma_theta": best_params.get("sigma_theta", 0.2),
                "alpha": best_params.get("alpha", 1.5),
            },
        }
        torch.save(checkpoint, save_path)

        print(f"[INFO] GNN model and metadata saved to {save_path}")


        return {
            "model": best_model,
            "best_params": best_params,
            "all_graphs": [g for g in dataset],
            "global_r2": global_r2,
            "global_mae": global_mae,
            "y_pred_all": y_pred_all,
            "train_idx": last_idx_train,
            "validation_idx": last_idx_val,
            "selected_features": selected_features
        }
    
    @staticmethod
    def _run_gnn_model(model_path, atom, ads_label, descriptors):
        """
        Run inference on a single NetworkX graph using a trained GNN model.

        Parameters
        ----------
        model_path : trained GNN model (torch checkpoint format)
        atom : ASE Atoms object representing the atomic structure of the system
        ads_label : label identifying the adsorbate species in the structure
        descriptors : precomputed atomic or structural descriptors to be attached to
            nodes during graph construction. Must match the descriptor format
            used during model training

        Returns
        -------
        pred_val : float
            Predicted adsorption energy
        """
        # Extract info
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint["model_config"]
        graph_config = checkpoint["graph_config"]

        # Adjust model config for inference (if needed)
        if len(model_config) > 6:  # if there are extra keys, filter them out
            allowed_keys = {"input_dim", "hidden1", "hidden2", "conv_type", "depth", "dropout"}
            model_config = {k: v for k, v in model_config.items() if k in allowed_keys}

        # Recreate model architecture
        model = GraphNN(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        graph = GraphTransform._make_graph(
            atoms_init=atom,
            adsorbate_label=ads_label,
            ads_dist_threshold=graph_config["ads_dist_threshold"],
            descriptor_df=descriptors,
            adj_ads_desc=graph_config["adj_ads_desc"],
            gas_db=graph_config["gas_db"],
            beta=graph_config["beta"],
            lambda_corr=graph_config["lambda_corr"],
            sigma_d=graph_config["sigma_d"],
            sigma_theta=graph_config["sigma_theta"],
            alpha=graph_config["alpha"]
        )

        data, _ = nx_to_pyg(
            graph=graph,
            node_importance=graph_config["node_importance"],
            node_features=graph_config["selected_features"]
        )
        data = data.to(device)
        
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, batch)
        
        pred_val = float(pred.cpu().item())

        return pred_val