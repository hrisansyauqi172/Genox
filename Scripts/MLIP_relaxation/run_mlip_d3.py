import os
import json
import io
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from fairchem.core import OCPCalculator
from Genox import GraphTransform
from catbench.dispersion import DispersionCorrection

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
        poscar = read(io.StringIO(entry["poscar"]), format="vasp")
        
        # --- Assign tags based on whether atom is fixed, surface, or adsorbate ---
        gt = GraphTransform(poscar)
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
        for i in range(len(poscar)):
            if i in ads_indices:
                tags.append(2)
            else:
                tags.append(1)

        poscar.set_tags(tags)

        output_dir = os.path.join(fm_path, phase, facet, dopant, ads)
        if site is not None:
            output_dir = os.path.join(output_dir, site)
        
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