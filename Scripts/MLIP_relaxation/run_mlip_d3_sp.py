import os
import json
import io
import numpy as np
from ase.io import read, write
from fairchem.core import OCPCalculator
from catbench.dispersion import DispersionCorrection
from Genox import GraphTransform

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

db_path = "ML_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)
print(f"✅ Dataset size: {len(db_list)}")

current_directory = os.getcwd()

fm_dirs = ['Genoc_d3', 'Genoc-Clam_d3']
for fm in fm_dirs:
    fm_path = os.path.join(current_directory, fm)
    if fm == 'Genoc':
        model_path = "/home/helmi.risansyauqi/Foundation_models/gnoc_oc22_oc20_all_s2ef/gnoc_oc22_oc20_all_s2ef.pt"
    else:
        model_path = "/home/helmi.risansyauqi/Foundation_models/gnoc_clam/gnoc_clam.pt"
    # Load MLIP foundation model
    calc = OCPCalculator(checkpoint_path=model_path, cpu=False)

    # Load D3 correction
    d3 = DispersionCorrection()
    calc_d3  = d3.apply(calc)

    for entry in db_list:
        eid = entry['id']
        parts = eid.split("_")
        if len(parts) == 4:
            phase, facet, dopant, ads = parts
            site = None
        elif len(parts) == 5:
            if parts[2] == 'sym':
                phase, facet, _, dopant, ads = parts
                facet = f"{facet}_sym"
            else:
                phase, facet, dopant, ads, extra = parts
                ads = f"{ads}_{extra}"
            site = None
        elif len(parts) >= 6:
            phase, facet, dopant, ads, site_tag, site_id = parts
            site = f"{site_tag}_{site_id}"
        else:
            raise ValueError(f"Unexpected ID format: {eid}")
        contcar = read(io.StringIO(entry["contcar"]), format="vasp")
        
        # --- Assign tags based on whether atom is fixed, surface, or adsorbate ---
        gt = GraphTransform(contcar)
        ads_base = ads.split("_")[0]
        if ads_base in ["2H", "Ho", "H", "Hm"]:
            ads_label = "H" if ads_base != "2H" else "2H"
        elif ads_base in ["bCO2", "oCO2", "ZnCO2"]:
            ads_label = "CO2"
        else:
            ads_label = ads_base
        ads_mask = gt._detect_adsorbates(ads_label)
        ads_indices = np.where(ads_mask)[0]    # get indices of adsorbate atoms

        tags = []
        for i in range(len(contcar)):
            if i in ads_indices:
                tags.append(2)
            else:
                tags.append(1)

        contcar.set_tags(tags)

        output_dir = os.path.join(fm_path, phase, facet, dopant, ads)
        if site is not None:
            output_dir = os.path.join(output_dir, site)
        
        energy_file = os.path.join(output_dir, "energy.txt")
        if os.path.exists(energy_file):
            continue
        else:
            os.makedirs(output_dir, exist_ok=True)
            write(os.path.join(output_dir, "CONTCAR"), contcar, format='vasp', direct=True)
            contcar.set_calculator(calc_d3)
            energy = contcar.get_potential_energy()
            forces = contcar.get_forces()
            total_force_vec = forces.sum(axis=0)
            total_force_mag = np.linalg.norm(total_force_vec)
            sum_atomic_forces = np.linalg.norm(forces, axis=1).sum()
            max_force = np.linalg.norm(forces, axis=1).max()
            with open(energy_file, "w") as f:
                f.write(f"Energy (eV): {energy}\n")
                f.write(f"Net force vector (eV/Å): {total_force_vec}\n")
                f.write(f"Net force magnitude (eV/Å): {total_force_mag}\n")
                f.write(f"Sum of atomic forces (eV/Å): {sum_atomic_forces}\n")
                f.write(f"Max atomic force (eV/Å): {max_force}\n")