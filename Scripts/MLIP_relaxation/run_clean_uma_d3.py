import os
import json
import io
from ase.io import read, write
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from catbench.dispersion import DispersionCorrection

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

db_path = "clean_surf_database.json"
with open(db_path, "r") as f:
    db_list = json.load(f)

current_directory = os.getcwd()

fm_uma = ['UMA_v11_d3', 'UMA_v12_d3']

for uma_path in fm_uma:
    fm_path = os.path.join(current_directory, uma_path)
    # Load MLIP foundation model
    if uma_path == 'UMA_v11':
        take_uma = 'uma-s-1p1'
        predictor = pretrained_mlip.get_predict_unit(take_uma, device="cuda")
        calc = FAIRChemCalculator(predictor, task_name="oc20")
    else:
        take_uma = 'uma-s-1p2'
        predictor = pretrained_mlip.get_predict_unit(take_uma, device="cuda")
        calc = FAIRChemCalculator(predictor, task_name="oc22")

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