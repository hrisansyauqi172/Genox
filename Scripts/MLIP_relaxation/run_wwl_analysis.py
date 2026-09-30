import os
import argparse
import json
import io
import pandas as pd
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from Genox import GraphTransform, WWLAnalysis, KernelApproach
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def load_atoms(eid, dct, label):
    if eid not in dct:
        print(f"⚠️ {eid} missing in {label}")
        return None
    return read(io.StringIO(dct[eid]["contcar"]), format="vasp")

def calculate_adwt(atoms, ref_atoms, threshold_limits=(0.1, 0.5), threshold_min=0.01, threshold_step=0.001):
    if len(atoms) != len(ref_atoms):
        raise ValueError(
            f"Atom-count mismatch: evaluated={len(atoms)}, "
            f"reference={len(ref_atoms)}"
        )

    if atoms.get_chemical_symbols() != ref_atoms.get_chemical_symbols():
        raise ValueError(
            "The evaluated and reference structures have different "
            "chemical symbols or atom ordering."
        )

    fixed_indices = set()
    for constraint in ref_atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_indices.update(constraint.get_indices())
    free_indices = np.array([index for index in range(len(ref_atoms)) if index not in fixed_indices], dtype=int)
    if len(free_indices) == 0:
        raise ValueError("No free atoms were found in the reference structure.")

    # Cartesian displacement relative to the DFT reference
    displacement = atoms.positions[free_indices] - ref_atoms.positions[free_indices]

    # Use the reference cell
    cell = ref_atoms.get_cell(complete=True)

    # Convert Cartesian displacement to fractional coordinates
    fractional_displacement = np.linalg.solve(cell.T, displacement.T).T

    # Apply minimum-image convention
    for axis, periodic in enumerate(ref_atoms.pbc):
        if periodic:
            fractional_displacement[:, axis] -= np.round(fractional_displacement[:, axis])

    # Convert back to Cartesian coordinates
    displacement_mic = fractional_displacement @ cell
    atomic_distances = np.linalg.norm(displacement_mic, axis=1)

    # Mean position error for this structure
    position_mae = np.mean(atomic_distances)

    results = {"position_mae": float(position_mae)}

    for upper_threshold in threshold_limits:
        thresholds = np.arange(threshold_min, upper_threshold + threshold_step / 2, threshold_step)

        # CatBench-style structure contribution:
        # 1 if the structure position MAE is below a threshold, otherwise 0
        dwt_values = position_mae < thresholds

        adwt = np.mean(dwt_values) * 100.0

        column_suffix = str(upper_threshold).replace(".", "p")
        results[f"adwt_{column_suffix}"] = float(adwt)

    return results

def load_eads(eid, dct, label):
    if eid not in dct:
        print(f"⚠️ {eid} missing in {label}")
        return None
    return dct[eid]["eads"]

def load_eform(eid, dct, label):
    if eid not in dct:
        print(f"⚠️ {eid} missing in {label}")
        return None
    return dct[eid]["eform"]

def load_force(eid, dct, label):
    if eid not in dct:
        print(f"⚠️ {eid} missing in {label}")
        return None
    return dct[eid]["force"]

def load_step(eid, dct, label):
    if eid not in dct:
        print(f"⚠️ {eid} missing in {label}")
        return None
    return dct[eid]["step"]

def classify_relaxation(atoms, ref_atoms, ads_label, fmax, step,
                        force_threshold=0.01, max_steps=500,
                        slab_displacement_threshold=1.0, bond_change_threshold=0.20,
                        desorption_distance_threshold=3.5, cell_tolerance=0.05):
    gt = GraphTransform(atoms)
    ref_gt = GraphTransform(ref_atoms)
    ads_mask = gt._detect_adsorbates(ads_label)
    ads_indices = np.where(ads_mask)[0]
    slab_indices = np.where(~ads_mask)[0]

    fixed_indices = set()
    for constraint in ref_atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_indices.update(constraint.get_indices())
    free_indices = np.array([index for index in range(len(ref_atoms)) if index not in fixed_indices], dtype=int)
    if len(free_indices) == 0:
        raise ValueError("No free atoms were found in the reference structure.")

    # Identify atoms that are both free and part of the slab
    free_slab_mask = np.isin(free_indices, slab_indices)
    free_ads_mask = np.isin(free_indices, ads_indices)

    if not np.any(free_slab_mask):
        raise ValueError("No free slab atoms were found in the reference structure.")
    if not np.any(free_ads_mask):
        raise ValueError("No free adsorbate atoms were found in the reference structure.")
    
    # Unphysical relaxation check
    # Minimum-image atomic displacement relative to DFT
    displacement = atoms.positions[free_indices] - ref_atoms.positions[free_indices]
    cell = ref_atoms.get_cell(complete=True)
    fractional_displacement = np.linalg.solve(cell.T, displacement.T).T

    for axis, periodic in enumerate(ref_atoms.pbc):
        if periodic:
            fractional_displacement[:, axis] -= np.round(fractional_displacement[:, axis])

    displacement_mic = fractional_displacement @ cell
    atomic_displacement = np.linalg.norm(displacement_mic, axis=1)
    slab_atomic_displacement = atomic_displacement[free_slab_mask]

    max_slab_displacement = np.max(slab_atomic_displacement)

    force_not_converged = fmax > force_threshold
    step_limit_reached = step == max_steps
    slab_moved = max_slab_displacement > slab_displacement_threshold
    unphysical_relaxation = (force_not_converged or step_limit_reached or slab_moved)

    # Check whether adsorbate has desorbed
    final_min_surface_distances = []
    for ads_index in ads_indices:
        distances = atoms.get_distances(ads_index, slab_indices, mic=True)
        final_min_surface_distances.append(np.min(distances))
    final_min_surface_distances = np.asarray(final_min_surface_distances)
    adsorbate_desorbed = np.min(final_min_surface_distances) > desorption_distance_threshold

    # Check whether adsorbate going outside the unit cell
    scaled_positions = atoms.get_scaled_positions(wrap=False)
    ads_scaled_positions = scaled_positions[ads_indices]
    inside_cell = np.ones(len(ads_indices), dtype=bool)

    for axis in range(3):
        coordinate = ads_scaled_positions[:, axis]
        inside_axis = ((coordinate >= -cell_tolerance) & (coordinate < 1.0 + cell_tolerance))
        inside_cell &= inside_axis
    num_adsorbate_atoms = len(ads_indices)
    num_inside_cell = int(np.count_nonzero(inside_cell))
    num_outside_cell = num_adsorbate_atoms - num_inside_cell
    outside_fraction = (num_outside_cell / num_adsorbate_atoms)
    # Classify as outside the unit cell when more than half
    adsorbate_outside_unit_cell = outside_fraction > 0.5

    # Adsorbate migration
    ref_adjacency = ref_gt._build_adjacency()
    mlip_adjacency = gt._build_adjacency()

    max_bond_change = 0.0
    adsorbate_surface_neighbours_changed = False
    adsorbate_coordination_changed = False
    
    for ads_index in ads_indices:
        # Distance from this adsorbate atom to every slab atom
        ref_distances = ref_atoms.get_distances(ads_index, slab_indices, mic=True)
        mlip_distances = atoms.get_distances(ads_index, slab_indices, mic=True)
    
        # Use the closest surface atom in the DFT structure
        closest_position = np.argmin(ref_distances)
    
        ref_distance = ref_distances[closest_position]
        mlip_distance = mlip_distances[closest_position]
    
        relative_change = abs(mlip_distance - ref_distance) / ref_distance
        max_bond_change = max(max_bond_change, relative_change)

        ref_surface_neighbours = {slab_index for slab_index in slab_indices if ref_adjacency[ads_index, slab_index]}
        mlip_surface_neighbours = {slab_index for slab_index in slab_indices if mlip_adjacency[ads_index, slab_index]}
        
        if ref_surface_neighbours != mlip_surface_neighbours:
            adsorbate_surface_neighbours_changed = True
        
        if len(ref_surface_neighbours) != len(mlip_surface_neighbours):
            adsorbate_coordination_changed = True

    adsorbate_close_to_surface = np.min(final_min_surface_distances) <= desorption_distance_threshold
    if adsorbate_close_to_surface and not adsorbate_outside_unit_cell:
        adsorbate_migration = (max_bond_change > bond_change_threshold or adsorbate_surface_neighbours_changed or adsorbate_coordination_changed)

    # Classification
    if unphysical_relaxation:
        classification = "unphysical_relaxation"
    elif adsorbate_desorbed:
        classification = "adsorbate_desorbed"
    elif adsorbate_outside_unit_cell:
        classification = "adsorbate_outside_unit_cell"
    elif adsorbate_migration:
        classification = "adsorbate_migration"
    else:
        classification = "normal"

    return classification


def process_structure(eid, label, atoms, ref_atoms, ads_label, db_dict, compute_graph=False, compute_class=False):
    """
    Compute/create graph, and similarity vs reference for one structure.
    """
    result = {}
    if compute_graph:
        result["graph"] = GraphTransform._make_graph(
            atoms_init=atoms,
            ads_dist_threshold=5.0,
            adsorbate_label=ads_label,
        )

    if compute_class:
        max_force = load_force(eid, db_dict, label)
        max_step = load_step(eid, db_dict, label)
        result["Class"] = classify_relaxation(atoms, ref_atoms, ads_label, fmax=max_force, step=max_step)

    result["Eads"] = load_eads(eid, db_dict, label),
    result["Eform"] = load_eform(eid, db_dict, label),

    return result

db_path = "ML_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)
print(f"✅ Dataset size: {len(db_list)}")

uma_v11_path = "UMA_v11_d3_database.json"
with open(uma_v11_path, "r") as f:
    uma_v11_list = json.load(f)

uma_v12_path = "UMA_v12_d3_database.json"
with open(uma_v12_path, "r") as f:
    uma_v12_list = json.load(f)

gnc_path = "Genoc_d3_database.json"
with open(gnc_path, "r") as f:
    gnc_list = json.load(f)

gncl_path = "Genoc-Clam_d3_database.json"
with open(gncl_path, "r") as f:
    gncl_list = json.load(f)

uma_v11_dict  = {e["id"]: e for e in uma_v11_list}
uma_v12_dict  = {e["id"]: e for e in uma_v12_list}
gnc_dict  = {e["id"]: e for e in gnc_list}
gncl_dict = {e["id"]: e for e in gncl_list}

output_file = "wasserstein_kernel_results.csv"
parser = argparse.ArgumentParser(description="Calculate WWL similarity, ADwT, and relaxation classifications.")
parser.add_argument("--run_mode", choices=["full", "updated_translated", "ADwT", "class"], default="full",
    help=(
        "full: calculate all outputs for all structures; "
        "updated_translated: recalculate all outputs only for entries tagged translate=True; "
        "ADwT: calculate/update only ADwT columns; "
        "class: calculate/update only relaxation-class columns."
    ),
)
args = parser.parse_args()
run_mode = args.run_mode
print(f"Run mode: {run_mode}", flush=True)

if run_mode == "full":
    init_res = None
else:
    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"{output_file} is required for run_mode='{run_mode}'. "
            "Run first with --run_mode full."
        )
    init_res = pd.read_csv(output_file)

translated_ids = set()
if run_mode == "updated_translated":
    for database in [uma_v11_list, uma_v12_list, gnc_list, gncl_list]:
        translated_ids.update(
            entry["id"] for entry in database if entry.get("translate", False)
        )
    print(f"Found {len(translated_ids)} structures tagged translate=True.")
    if not translated_ids:
        print("No translated structures found. Nothing to update.")
        raise SystemExit(0)

mlip_specs = [
    ("UMA_v11", uma_v11_dict, "uma_v11"),
    ("UMA_v12", uma_v12_dict, "uma_v12"),
    ("Genoc", gnc_dict, "gnc"),
    ("Genoc-Clam", gncl_dict, "gncl"),
]

compute_graph = run_mode in ["full", "updated_translated"]
compute_adwt = run_mode in {"full", "updated_translated", "ADwT"}
compute_class = run_mode in {"full", "updated_translated", "class"}

entries_to_process = [
    entry for entry in db_list
    if run_mode != "updated_translated" or entry["id"] in translated_ids
]

results = []
i = 0
x = 0
for entry in entries_to_process:
    eid = entry['id']
    ads = entry['ads_label']

    db_eads = entry['E_ads']
    db_eform = entry['E_form']

    db_contcar = read(io.StringIO(entry["contcar"]), format="vasp")
    if compute_graph:
        db_graph = GraphTransform._make_graph(
            atoms_init=db_contcar,
            adsorbate_label=ads,
            ads_dist_threshold=5.0,
        )

    # UMA, Genoc, and Genoc-Clam
    uma_v11_atoms  = load_atoms(eid, uma_v11_dict, "UMA_v11")
    uma_v12_atoms  = load_atoms(eid, uma_v12_dict, "UMA_v12")
    gnc_atoms  = load_atoms(eid, gnc_dict, "Genoc")
    gncl_atoms = load_atoms(eid, gncl_dict, "Genoc-Clam")

    # Check that all four MLIP structures are available
    if not all(atoms is not None for atoms in (uma_v11_atoms, uma_v12_atoms, gnc_atoms, gncl_atoms)):
        x += 1
        print(f"{eid} analysis failed (failure number = {x})", flush=True)
        continue

    uma_v11  = process_structure(eid, "UMA_v11", uma_v11_atoms, db_contcar, ads, uma_v11_dict, compute_graph, compute_class)
    uma_v12  = process_structure(eid, "UMA_v12", uma_v12_atoms, db_contcar, ads, uma_v12_dict, compute_graph, compute_class)
    gnc  = process_structure(eid, "Genoc", gnc_atoms, db_contcar, ads, gnc_dict, compute_graph, compute_class)
    gncl = process_structure(eid, "Genoc-Clam", gncl_atoms, db_contcar, ads, gncl_dict, compute_graph, compute_class)

    new_row = {
        "id": eid,
        "DFT_Eads": db_eads,
        "UMA_v11_Eads": uma_v11["Eads"],
        "UMA_v12_Eads": uma_v12["Eads"],
        "Gnc_Eads": gnc["Eads"],
        "Gncl_Eads": gncl["Eads"],
        "DFT_Eform": db_eform,
        "UMA_v11_Eform": uma_v11["Eform"],
        "UMA_v12_Eform": uma_v12["Eform"],
        "Gnc_Eform": gnc["Eform"],
        "Gncl_Eform": gncl["Eform"],
    }

    if compute_class:
        new_row.update({
            "Class_UMA_v11": uma_v11["Class"],
            "Class_UMA_v12": uma_v12["Class"],
            "Class_gnc": gnc["Class"],
            "Class_gncl": gncl["Class"],
        })

    if compute_adwt:
        uma_v11_adwt = calculate_adwt(uma_v11_atoms, db_contcar)
        uma_v12_adwt = calculate_adwt(uma_v12_atoms, db_contcar)
        gnc_adwt = calculate_adwt(gnc_atoms, db_contcar)
        gncl_adwt = calculate_adwt(gncl_atoms, db_contcar)

        new_row.update({
            "ADwT_0p1_uma_v11": uma_v11_adwt["adwt_0p1"],
            "ADwT_0p5_uma_v11": uma_v11_adwt["adwt_0p5"],
            "ADwT_0p1_uma_v12": uma_v12_adwt["adwt_0p1"],
            "ADwT_0p5_uma_v12": uma_v12_adwt["adwt_0p5"],
            "ADwT_0p1_gnc": gnc_adwt["adwt_0p1"],
            "ADwT_0p5_gnc": gnc_adwt["adwt_0p5"],
            "ADwT_0p1_gncl": gncl_adwt["adwt_0p1"],
            "ADwT_0p5_gncl": gncl_adwt["adwt_0p5"],
        })

    if compute_graph:
        graphs = [uma_v11["graph"], uma_v12["graph"], gnc["graph"], gncl["graph"]]
        # selected_features = ["d_band_center", "d_orbital", "E-fermi", "p_orbital", "Work_function", "d_band_filling", "DDEC6_charge"]
        D = WWLAnalysis._wasserstein_calc(graphs, [db_graph], depth=1, node_importance=True)
        D_uma_v11, D_uma_v12, D_gnc, D_gncl = D[:, 0]
        D_tot = np.array([[0.0, *D[:, 0]]])
        K = KernelApproach._kernel_transform(D_tot, hyperparameters=[None], type_kernel="Laplacian")
        K_uma_v11, K_uma_v12, K_gnc, K_gncl = K[0, 1:]

        new_row.update({
            "D_uma_v11": D_uma_v11,
            "D_uma_v12": D_uma_v12,
            "D_gnc": D_gnc,
            "D_gncl": D_gncl,
            "K_uma_v11": K_uma_v11,
            "K_uma_v12": K_uma_v12,
            "K_gnc": K_gnc,
            "K_gncl": K_gncl,
        })

    if run_mode == "full":
        results.append(new_row)
    else:
        mask = init_res["id"] == eid
        if mask.any():
            for key, value in new_row.items():
                init_res.loc[mask, key] = value
        else:
            init_res = pd.concat([init_res, pd.DataFrame([new_row])], ignore_index=True)

    i += 1
    if run_mode == "full":
        print(f"Structure {i}/{len(entries_to_process)}. {eid} analysis success", flush=True)
    else:
        print(f"Updated {i}/{len(entries_to_process)}. {eid}", flush=True)

if run_mode == "full":
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
else:
    init_res.to_csv(output_file, index=False)