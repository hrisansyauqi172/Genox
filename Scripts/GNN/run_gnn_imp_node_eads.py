import os
import json
import argparse
import io
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from Genox import GNNPredModel, GraphTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Train GNN on adsorption energies with specified embedding")
parser.add_argument('--conv_type', type=str, required=True,
                    choices=["GCN", "GraphSAGE", "GIN", "GAT"],
                    help="GNN convolution type")
args = parser.parse_args()
conv_type = args.conv_type
print(f"[INFO] Using GNN embedding: {conv_type}")

db_path = "ML_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)
print(f"✅ Dataset size: {len(db_list)}")

# Deserialize ASE Atoms objects from POSCAR/CONTCAR strings
for entry in db_list:
    entry["poscar"] = read(io.StringIO(entry["poscar"]), format="vasp")
    entry["contcar"] = read(io.StringIO(entry["contcar"]), format="vasp")
    entry["descriptors"] = pd.read_json(entry["descriptors"])

# Split data into training and test sets
np.random.seed(777)
indices = np.arange(len(db_list))
np.random.shuffle(indices)

train_frac = 0.85
train_idx = indices[:int(train_frac*len(db_list))]
train_list = [db_list[i] for i in train_idx]
ads_energy_train = [entry["E_ads"] for entry in train_list]
ids_train = [entry["id"] for entry in train_list]
for i in range(len(train_list)):
    assert train_list[i]["id"] == ids_train[i], \
        f"Misalignment detected at index {i}: {train_list[i]['id']} != {ids_train[i]}"
print(f"✅ Training set size: {len(train_list)}")

test_idx = indices[int(train_frac * len(db_list)):]    
test_list = [db_list[i] for i in test_idx]
ads_energy_test = [entry["E_ads"] for entry in test_list]
ids_test = [entry["id"] for entry in test_list]
for i in range(len(test_list)):
    assert test_list[i]["id"] == ids_test[i], \
        f"Misalignment detected at index {i}: {test_list[i]['id']} != {ids_test[i]}"
print(f"✅ Test set size: {len(test_list)}")

tt_split = 1 - (0.70 / train_frac)

# Train GPR model
all_features = ["d_band_center", "DDEC6_charge", "s_orbital", "p_orbital", "d_orbital", "Work_function", "Pauling_EN", "E-fermi", "HOMO_rel", "LUMO_rel", "p_band_center"]
print(f"\n[INFO] Training GNN with {conv_type} embedding...\n")
gnn_result = GNNPredModel._gnn_ads_energy(
    atoms_list=train_list,
    ads_energy_list=ads_energy_train,
    n_trials=40,
    random_state=777,
    CV=None,
    tt_split=tt_split,
    epochs=400,
    plot_name=f"eads_tt",
    conv_type=conv_type,
    node_importance=True,
    adj_ads_desc=True,
    tune_pertub_param=True,
    node_features=all_features
)
gnn_model = gnn_result["model"]
gnn_ref_graphs = gnn_result["all_graphs"]

# Save predictions
os.makedirs("saved_models", exist_ok=True)
n = len(gnn_result["y_pred_all"])
# Initialize all as unknown (optional safety)
split_labels = np.array(["unknown"] * n)

# Assign train/test
split_labels[gnn_result["train_idx"]] = "train"
split_labels[gnn_result["validation_idx"]] = "validation"

gnn_predictions = pd.DataFrame({
    "structure": ids_train,
    "target_E_ads": ads_energy_train,
    "predicted_E_ads": gnn_result["y_pred_all"],
    "split": split_labels
})
gnn_predictions.to_csv(f"saved_models/GNN_predictions_eads_tt_imp_node_{conv_type}.csv", index=False)

print(f"[INFO] Finished training GNN with {conv_type} embedding.")

explainer = Explainer(
    model=gnn_model,
    algorithm=GNNExplainer(epochs=400),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
)

graphs_nx = [GraphTransform._make_graph(
       atoms_init=item["contcar"],
       adsorbate_label=item["ads_label"],
       ads_dist_threshold=gnn_result["best_params"]["ads_dist_threshold"],
       descriptor_df=item["descriptors"]
    )
    for i, item in enumerate(train_list)]

feature_imp_all = []
feature_imp_surface = []
feature_imp_adsorbate = []

node_imp = []
edge_imp = []

meta = []

for graph, nx_graph in zip(gnn_ref_graphs, graphs_nx):
    data = graph.to(device)

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    )

    node_mask = explanation.node_mask.detach().cpu()  # [N_nodes, N_features]

    for i in range(data.num_nodes):
        attrs = nx_graph.nodes[i]

        is_ads = attrs["is_adsorbate"]
        symbol = attrs["symbol"]

        feature_imp_all.append(node_mask[i])

        if is_ads:
            feature_imp_adsorbate.append(node_mask[i])
        else:
            feature_imp_surface.append(node_mask[i])

# Stack feature importances
feature_imp_all = torch.stack(feature_imp_all)               # [N_nodes_all, N_features]
feature_imp_surface = torch.stack(feature_imp_surface)       # [N_surface_nodes, N_features]
feature_imp_adsorbate = torch.stack(feature_imp_adsorbate)   # [N_ads_nodes, N_features]

# Mean + std (GLOBAL, SHAP-like)
mean_imp_all = feature_imp_all.mean(dim=0)
std_imp_all = feature_imp_all.std(dim=0)

mean_imp_surface = feature_imp_surface.mean(dim=0)
mean_imp_adsorbate = feature_imp_adsorbate.mean(dim=0)

# Descriptor names (ordering must match node feature construction)
feature_names = gnn_result["selected_features"]
print(len(feature_names))

# Sort features by importance
order = torch.argsort(mean_imp_all, descending=True)
print(len(order))

mean_imp_all = mean_imp_all[order]
std_imp_all = std_imp_all[order]

mean_imp_surface = mean_imp_surface[order]
mean_imp_adsorbate = mean_imp_adsorbate[order]

feature_names = [feature_names[i] for i in order]

TOP_K = min(20, len(feature_names))

os.makedirs("gnn_feat_imp", exist_ok=True)
# Plot 1: Global feature importance
plt.figure(figsize=(7, 6))
plt.barh(
    feature_names[:TOP_K],
    mean_imp_all[:TOP_K].cpu().numpy(),
    xerr=std_imp_all[:TOP_K].cpu().numpy()
)
plt.gca().invert_yaxis()
plt.xlabel("Mean Feature Importance")
plt.title(f"Global Feature Importance ({conv_type})")
plt.tight_layout()
plt.savefig(f"gnn_feat_imp/global_feat_imp_node_{conv_type}.png")
plt.close()

# Plot 2: Surface vs adsorbate
y = np.arange(TOP_K)
width = 0.35

plt.figure(figsize=(8, 6))
plt.barh(
    y - width / 2,
    mean_imp_surface[:TOP_K].cpu().numpy(),
    height=width,
    label="Surface"
)
plt.barh(
    y + width / 2,
    mean_imp_adsorbate[:TOP_K].cpu().numpy(),
    height=width,
    label="Adsorbate"
)

plt.yticks(y, feature_names[:TOP_K])
plt.gca().invert_yaxis()
plt.xlabel("Mean Feature Importance")
plt.title(f"Surface vs Adsorbate Feature Importance ({conv_type})")
plt.legend()
plt.tight_layout()
plt.savefig(f"gnn_feat_imp/surf_ads_feat_imp_node_{conv_type}.png")
plt.close()

model_name = f"GNN_{conv_type}_eads_tt_imp_node.pt"
model_path = os.path.join("saved_models", model_name)

y_pred_test = []
for item in test_list:
    y_pred = GNNPredModel._run_gnn_model(
        model_path=model_path,
        atom=item["contcar"],
        ads_label=item["ads_label"],
        descriptors=item["descriptors"]
    )
    y_pred_test.append(y_pred)

test_df = pd.DataFrame({
    "structure": ids_test,
    "target_E_ads": ads_energy_test,
    "predicted_E_ads": y_pred_test
})
test_df.to_csv(f"saved_models/GNN_eads_tt_test_{conv_type}_imp_node.csv", index=False)