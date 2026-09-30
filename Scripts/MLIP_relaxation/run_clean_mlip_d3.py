import os
import json
import io
from ase.io import read, write
from ase.optimize import BFGS
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
        poscar = read(io.StringIO(entry["structure"]), format="vasp")
        
        # --- Assign tags based on whether atom is fixed ---
        tags = []
        for i in range(len(poscar)):
            tags.append(1)  # free-to-move atom

        poscar.set_tags(tags)
        output_dir = os.path.join(fm_path, phase, facet, dopant, "Clean")

        if os.path.exists(os.path.join(output_dir, "CONTCAR")):
            continue
        else:
            os.makedirs(output_dir, exist_ok=True)
            write(os.path.join(output_dir, "POSCAR"), poscar, format='vasp', direct=True)
            poscar.set_calculator(calc_d3)
            log_file = os.path.join(output_dir, "relaxation.log")
            traj_file = os.path.join(output_dir, "relaxed_structure.traj")
        
            opt = BFGS(poscar, logfile=log_file, trajectory=traj_file)
            opt.run(fmax=0.01, steps=500)

            write(os.path.join(output_dir, "CONTCAR"), poscar, format='vasp', direct=True)