from ase import Atoms
from pymatgen.analysis.local_env import CrystalNN
from ase.data import covalent_radii
from pymatgen.io.ase import AseAtomsAdaptor
from collections import Counter
from itertools import combinations
import numpy as np

class AdsorptionSite:
    @staticmethod
    def _top_layer(atoms: Atoms, z_tail_fraction=0.2):
        z_positions = atoms.get_positions()[:, 2]
        z_base = min(z_positions)
        z_range = max(z_positions) - z_base
        z_cutoff = z_base + (z_range * (1 - z_tail_fraction))  # e.g., 80th percentile if z_tail_fraction=0.2
        z_low = z_base + (z_range * z_tail_fraction)
        z_evaluated = np.where(z_positions > z_cutoff)[0]
        z_bulk = np.where((z_positions >= z_low) & (z_positions <= z_cutoff))[0]
        
        struct = AseAtomsAdaptor.get_structure(atoms)
        symbols = atoms.get_chemical_symbols()
        cnn = CrystalNN()
        n_atoms = len(atoms)
        cn = np.array([len(cnn.get_nn_info(struct, i)) for i in range(n_atoms)])
        bulk_cn_ref = {}
        for sym in set(symbols):
            bulk_atoms = [i for i in z_bulk if symbols[i] == sym]
            if bulk_atoms:
                bulk_cn_ref[sym] = np.max([cn[i] for i in bulk_atoms])
        
        top_layer_mask = []
        for i in z_evaluated:
            sym = symbols[i]
            if sym not in bulk_cn_ref:
                # Element not present in bulk → likely adsorbate
                top_layer_mask.append(int(i))
            else:
                if bulk_cn_ref[sym] - cn[i] >= 1:
                    top_layer_mask.append(int(i))

        # --- Height-based inclusion ---
        organic_elements = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
        top_layer_mask = sorted(set(top_layer_mask))
        top_layer_mask_by_element = {}
        up_height_tol = round(max(covalent_radii[a.number] for a in atoms), 1)
        bot_height_tol = round(min(covalent_radii[a.number] for a in atoms), 1)
        for i in top_layer_mask:
            sym = symbols[i]
            if sym not in top_layer_mask_by_element:
                top_layer_mask_by_element[sym] = []
            top_layer_mask_by_element[sym].append(i)

        for i in z_evaluated:
            sym = symbols[i]
            if sym not in top_layer_mask_by_element:
                continue
            z_ref = np.median([z_positions[j] for j in top_layer_mask_by_element[sym]])
            if sym in organic_elements:
                z_look = z_positions[i] - z_ref
            else:
                z_look = abs(z_positions[i] - z_ref)
            if -bot_height_tol < z_look <= up_height_tol:
                if i not in top_layer_mask:
                    top_layer_mask.append(int(i))
            else:
                if i in top_layer_mask:
                    top_layer_mask.remove(int(i))

        # --- Add rare elements ---
        symbol_counts = Counter(symbols[i] for i in z_evaluated)
        top_layer_mask += [int(i) for i in z_evaluated if symbol_counts[symbols[i]] <= 2 and int(i) not in top_layer_mask]

        return sorted(set(top_layer_mask))
    
    @staticmethod
    def _ads_site_detect(atoms: Atoms, z_tail_fraction=0.2, site_type=[1, 2, 3], tol=0.7):
        top_layer_mask = AdsorptionSite._top_layer(atoms, z_tail_fraction=z_tail_fraction)
        symbols = atoms.get_chemical_symbols()
        organic_elements = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}

        metal_top_indices = [i for i in top_layer_mask if symbols[i] not in organic_elements]
        metal_atoms = atoms[metal_top_indices]

        struct = AseAtomsAdaptor.get_structure(metal_atoms)
        cnn = CrystalNN(weighted_cn=True, distance_cutoffs=(1.2, 3.0))
        metal_neighbors = {}
        for i in range(len(metal_atoms)):
            nn_data = cnn.get_nn_data(struct, i)  # returns all CN environments
            max_cn = max(nn_data.cn_nninfo.keys())
            # get all neighbors corresponding to max CN
            neighbors_max_cn = [n['site_index'] for n in nn_data.cn_nninfo[max_cn]]
            metal_neighbors[i] = set(neighbors_max_cn)

        sites = []
        existing_positions = []

        def fully_connected(group, neighbor_map):            
            for a, b in combinations(group, 2):
                if b not in neighbor_map.get(a, []):
                    return False
                
                pos_a = metal_atoms.positions[a]
                pos_b = metal_atoms.positions[b]
                d_linalg = np.linalg.norm(pos_a - pos_b)
                d_mic = metal_atoms.get_distance(a, b, mic=True)
                if abs(d_linalg - d_mic) / d_linalg > 0.1:
                    return False
            return True
        
        for coord in site_type:   # 1=top, 2=bridge, 3=hollow
            if coord == 1:
                # Top sites: all top-layer atoms
                for i in top_layer_mask:
                    sites.append({
                        'coordination': 1,
                        'indices': [i],
                        'position': atoms.positions[i].copy()
                    })
                    existing_positions.append(atoms.positions[i][:2].copy())
                continue
            
            for sub_group in combinations(range(len(metal_atoms)), coord):
                if not fully_connected(sub_group, metal_neighbors):
                    continue

                group = [metal_top_indices[i] for i in sub_group]
                pos = np.mean(atoms.positions[group], axis=0)
                to_remove = []

                # --- Adjust position if too close to organic atoms ---
                adjusted = True
                remove_site = False
                while adjusted:
                    adjusted = False
                    for idx in [i for i in top_layer_mask if symbols[i] in organic_elements]:
                        org_pos = atoms.positions[idx]
                        cov_dist = covalent_radii[atoms[idx].number]
                        vector = pos[:2] - org_pos[:2]  # vector from organic atom → site
                        distance = np.linalg.norm(vector)
                        if distance <= cov_dist:
                            remove_site = True
                            break
                        elif cov_dist < distance < cov_dist * 1.5:
                            direction = vector / (distance if distance > 1e-8 else 1e-8)
                            move_dist = 0.5 * (cov_dist - distance)
                            pos[:2] -= move_dist * direction
                            adjusted = True
                            break

                    if remove_site:
                        break
                if remove_site:
                    continue
                
                # --- Avoid duplicate/close sites ---
                keep_new = True  # flag for whether to keep the new site

                for j, ep in enumerate(existing_positions):
                    if np.linalg.norm(pos[:2] - ep) < tol:
                    # There’s a close existing site
                        existing_coord = sites[j]['coordination']

                        if existing_coord == 1 and coord > 1:
                            # Case: existing is top site (coord=1), current is multi-bonded site
                            # → keep the top site, discard the new one
                            keep_new = False
                            break
                        else:
                            # Otherwise, remove the old one and keep the new
                            to_remove.append(j)

                # Remove older duplicates if keeping the new site
                if keep_new:
                    for j in sorted(to_remove, reverse=True):
                        existing_positions.pop(j)
                        sites.pop(j)
                else:
                    continue

                sites.append({
                    'coordination': coord,
                    'indices': group,
                    'position': pos
                })
                existing_positions.append(pos[:2].copy())

        return sites