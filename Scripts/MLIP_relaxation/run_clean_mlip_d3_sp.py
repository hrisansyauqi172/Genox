import os
import json
import io
import numpy as np
from ase.io import read, write
from fairchem.core import OCPCalculator
from catbench.dispersion import DispersionCorrection

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

db_path = "clean_surf_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)

current_directory = os.getcwd()

fm_dirs = ['Genoc_d3', 'Genoc-Clam_d3']
for fm in fm_dirs:
    fm_path = os.path.join(current_directory, fm)
    if fm == 'Genoc':
        model_path = "/home/helmi.risansyauqi/Foundation_models/gnoc_oc22_oc20_all_s2ef/gnoc_oc22_oc20_all_s2ef.pt"
    else:
        model_path = "/home/helmi.risansyauqi/Foundation_models/gnoc_clam/gnoc_clam.pt"
    # Load MLIP foundation model
    calc = OCPCalculator(checkpoint_path=model_path)
    
    # Load D3 correction
    d3 = DispersionCorrection()
    calc_d3  = d3.apply(calc)

    for entry in db_list:
        eid = entry['id']
        parts = eid.split("_")
        if len(parts) == 3:
            phase, facet, dopant = parts
            sym = None
        else:
            phase, facet, _, dopant = parts
            facet = f"{facet}_sym"
        contcar = read(io.StringIO(entry["structure"]), format="vasp")
        
        # --- Assign tags based on whether atom is fixed ---
        tags = []
        for i in range(len(contcar)):
            tags.append(1)  # free-to-move atom

        contcar.set_tags(tags)
        output_dir = os.path.join(fm_path, phase, facet, dopant, "Clean")

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