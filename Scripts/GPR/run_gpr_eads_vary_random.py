import os
import json
import io
import argparse
import pandas as pd
import numpy as np
from ase.io import read
from Genox import PredictionModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# ==========================
# ✅ ARGPARSE FOR DATASET SIZE
# ==========================
parser = argparse.ArgumentParser(description="Run GPR with variable random seed")
parser.add_argument(
    "--random_seed",
    type=int,
    default=None,
    help="Random seed variations"
)
args = parser.parse_args()

db_path = "ML_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)
print(f"✅ Dataset size: {len(db_list)}", flush=True)


# Deserialize ASE Atoms objects from POSCAR/CONTCAR strings
for entry in db_list:
    entry["poscar"] = read(io.StringIO(entry["poscar"]), format="vasp")
    entry["contcar"] = read(io.StringIO(entry["contcar"]), format="vasp")
    entry["descriptors"] = pd.read_json(entry["descriptors"])

# Split data into training and test sets
print(f"🔍 Using random seed: {args.random_seed}", flush=True)
np.random.seed(args.random_seed)
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
print(f"✅ Training set size: {len(train_list)}", flush=True)

test_idx = indices[int(train_frac * len(db_list)):]    
test_list = [db_list[i] for i in test_idx]
ads_energy_test = [entry["E_ads"] for entry in test_list]
ids_test = [entry["id"] for entry in test_list]
for i in range(len(test_list)):
    assert test_list[i]["id"] == ids_test[i], \
        f"Misalignment detected at index {i}: {test_list[i]['id']} != {ids_test[i]}"
print(f"✅ Test set size: {len(test_list)}", flush=True)

tt_split = 1 - (0.70 / train_frac)

# Train GPR model
selected_features = ["d_band_center", "DDEC6_charge", "s_orbital", "p_orbital", "d_orbital", "Work_function", "Pauling_EN", "E-fermi", "HOMO_rel", "LUMO_rel", "p_band_center"]
gpr_result = PredictionModel._gpr_ads_energy(
    atoms_list=train_list,
    ads_energy_list=ads_energy_train,
    n_trials=40,
    wl_depth=1,
    random_state=args.random_seed,
    CV=None,
    tt_split=tt_split,
    plot_name=f"eads_tt_rs_{args.random_seed}",
    type_kernel="Laplacian",
    node_importance=False,
    adj_ads_desc=True,
    tune_pertub_param=True,
    node_attr_keys=selected_features,
    save_folder="Random_seed"
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
    "structure": ids_train,
    "target_E_ads": ads_energy_train,
    "predicted_E_ads": gpr_result["y_pred_all"],
    "uncertainty": gpr_result["uncertainty_all"],
    "split": split_labels
})
gpr_predictions.to_csv(f"saved_models/Random_seed/GPR_predictions_eads_tt_rs_{args.random_seed}.csv", index=False)

model_name = f"GPR_Laplacian_eads_tt_rs_{args.random_seed}.pkl"
model_path = os.path.join("saved_models", "Random_seed", model_name)

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
test_df.to_csv(f"saved_models/Random_seed/GPR_predictions_eads_tt_rs_{args.random_seed}_test.csv", index=False)
