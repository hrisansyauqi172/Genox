import os
import json
import io
import argparse
import pandas as pd
import numpy as np
from ase.io import read
from sklearn.model_selection import train_test_split
from Genox import PredictionModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# ==========================
# ✅ ARGPARSE FOR DATASET SIZE
# ==========================
parser = argparse.ArgumentParser(description="Run GPR with variable dataset size")
parser.add_argument(
    "--size",
    type=int,
    default=None,
    help="Number of samples to use (50–665)"
)
args = parser.parse_args()

db_path = "ML_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)
print(f"✅ Dataset size: {len(db_list)}")

# ==========================
# ✅ SUBSAMPLE DATASET
# ==========================
def increment(idx_train, idx_val, train_list, test_list, size):
    train_size = int(0.7 * size)
    val_size = int(0.15 * size)

    sel_idx_train = idx_train[:train_size]
    sel_idx_val = idx_val[:val_size]
    selected_idx = np.concatenate([sel_idx_train, sel_idx_val])
    new_train_list = [train_list[i] for i in selected_idx]
    adj_idx_train = np.arange(len(sel_idx_train))
    adj_idx_val = np.arange(len(sel_idx_train), len(sel_idx_train) + len(sel_idx_val))

    remaining_train = idx_train[train_size:]
    remaining_val = idx_val[val_size:]
    leftover_idx = np.concatenate([remaining_train, remaining_val])
    remain_list = [train_list[i] for i in leftover_idx]
    new_test_list = test_list + remain_list

    return adj_idx_train, adj_idx_val, new_train_list, new_test_list

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
print(f"✅ Initial Test set size: {len(test_list)}")

tt_split = 1 - (0.70 / train_frac)

idx = np.arange(len(train_list))
idx_train, idx_val = train_test_split(idx, test_size=tt_split, random_state=42, shuffle=True)

if args.size is not None:
    if args.size < 50 or args.size > 665:
        raise ValueError("Dataset size must be between 50 and 665")
    if args.size > len(db_list):
        raise ValueError(f"Requested size {args.size} exceeds available data {len(db_list)}")

    sel_idx_train, sel_idx_val, new_train_list, new_test_list = increment(idx_train, idx_val, train_list, test_list, args.size)

new_ads_energy_train = [entry["E_ads"] for entry in new_train_list]
new_ids_train = [entry["id"] for entry in new_train_list]
for i in range(len(new_train_list)):
    assert new_train_list[i]["id"] == new_ids_train[i], \
        f"Misalignment detected at index {i}: {new_train_list[i]['id']} != {new_ids_train[i]}"
print(f"✅ New Training set size: {len(new_train_list)}")

new_ads_energy_test = [entry["E_ads"] for entry in new_test_list]
new_ids_test = [entry["id"] for entry in new_test_list]
for i in range(len(new_test_list)):
    assert new_test_list[i]["id"] == new_ids_test[i], \
        f"Misalignment detected at index {i}: {new_test_list[i]['id']} != {new_ids_test[i]}"
print(f"✅ New Test set size: {len(new_test_list)}")

# Train GPR model
selected_features = ["d_band_center", "DDEC6_charge", "s_orbital", "p_orbital", "d_orbital", "Work_function", "Pauling_EN", "E-fermi", "HOMO_rel", "LUMO_rel", "p_band_center"]
gpr_result = PredictionModel._gpr_ads_energy(
    atoms_list=new_train_list,
    ads_energy_list=new_ads_energy_train,
    n_trials=40,
    wl_depth=1,
    random_state=777,
    spec_idx_train=sel_idx_train,
    spec_idx_val=sel_idx_val,
    plot_name=f"eads_tt_{args.size}",
    type_kernel="Laplacian",
    node_importance=False,
    adj_ads_desc=True,
    tune_pertub_param=True,
    node_attr_keys=selected_features,
    save_folder="Dataset_size"
)

# Save predictions
os.makedirs("saved_models", exist_ok=True)
n = len(gpr_result["y_pred_all"])
# Initialize all as unknown (optional safety)
split_labels = np.array(["unknown"] * n)

# Assign train/test
split_labels[gpr_result["train_idx"]] = "train"
split_labels[gpr_result["validation_idx"]] = "validation"

gpr_predictions = pd.DataFrame({
    "structure": new_ids_train,
    "target_E_ads": new_ads_energy_train,
    "predicted_E_ads": gpr_result["y_pred_all"],
    "uncertainty": gpr_result["uncertainty_all"],
    "split": split_labels
})
gpr_predictions.to_csv(f"saved_models/Dataset_size/GPR_predictions_eads_tt_{args.size}.csv", index=False)

model_name = f"GPR_Laplacian_eads_tt_{args.size}.pkl"
model_path = os.path.join("saved_models", "Dataset_size", model_name)

y_pred_test = []
uncertainty_test = []
for item in test_list:
    y_pred, unc_test = PredictionModel._run_gpr_model(
        model_path=model_path,
        atom=item["contcar"],
        ads_label=item["ads_label"],
        descriptors=item["descriptors"]
    )
    y_pred_test.append(y_pred)
    uncertainty_test.append(unc_test)

test_df = pd.DataFrame({
    "structure": ids_test,
    "target_E_ads": ads_energy_test,
    "predicted_E_ads": y_pred_test,
    "uncertainty": uncertainty_test
})
test_df.to_csv(f"saved_models/Dataset_size/GPR_predictions_eads_tt_{args.size}_test.csv", index=False)

y_pred_new_test = []
uncertainty_new_test = []
for item in new_test_list:
    y_pred, unc_test = PredictionModel._run_gpr_model(
        model_path=model_path,
        atom=item["contcar"],
        ads_label=item["ads_label"],
        descriptors=item["descriptors"]
    )
    y_pred_new_test.append(y_pred)
    uncertainty_new_test.append(unc_test)

test_df = pd.DataFrame({
    "structure": new_ids_test,
    "target_E_ads": new_ads_energy_test,
    "predicted_E_ads": y_pred_new_test,
    "uncertainty": uncertainty_new_test
})
test_df.to_csv(f"saved_models/Dataset_size/GPR_predictions_eads_tt_{args.size}_new_test.csv", index=False)
