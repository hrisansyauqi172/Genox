import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.build.tools import sort
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor

class AddAdsorbate:
    def __init__(self, slab: Atoms, surface_desc: pd.DataFrame = None, ads_desc: pd.DataFrame = None):
        """
        Initialize AddAdsorbate with a slab and optional descriptors.

        Args:
            slab: ASE Atoms object of the clean surface.
            surface_desc: pd.DataFrame or CSV path of surface descriptors (optional)
            ads_desc: pd.DataFrame or CSV path of adsorbate descriptors (optional)
        """
        self.slab = slab.copy()
        self.symbols = slab.get_chemical_symbols()

        if isinstance(surface_desc, str):
            surface_desc = pd.read_csv(surface_desc)
        if isinstance(ads_desc, str):
            ads_desc = pd.read_csv(ads_desc)
        self.surface_desc = surface_desc
        self.ads_desc = ads_desc

    @staticmethod
    def _combine_descriptors(slab: Atoms, adsorbate: Atoms, combined: Atoms, surface_desc: pd.DataFrame, ads_desc: pd.DataFrame):
        """
        Combine slab and adsorbate descriptors for a new structure.
        Slab atoms are matched by position+symbol.
        Adsorbate descriptors are assigned based on symbol if atom is not in slab.
        """

        slab_positions = slab.get_positions()
        slab_symbols = np.array(slab.get_chemical_symbols())
        combined_symbols = np.array(combined.get_chemical_symbols())
        ads_symbols = np.array(adsorbate.get_chemical_symbols())
        ads_length = len(adsorbate)

        # ---- ALIGN COMBINED TO SLAB (remove rigid fractional shift) ----
        slab_frac = slab.get_scaled_positions()
        comb_frac = combined.get_scaled_positions()

        # Define light/organic atoms to exclude
        organic_symbols = {"H", "C", "N", "S", "F", "Cl", "Br", "I"}

        # Reference atoms: anything that is not organic OR is oxygen in the slab
        ref_idx = [i for i, s in enumerate(combined_symbols) if (s not in organic_symbols) or s == "O"]
        o_idx = [i for i, s in enumerate(combined_symbols) if s == "O"]
        o_slab_idx = [i for i, s in enumerate(slab_symbols) if s == "O"]
        if len(o_idx) > len(o_slab_idx):
            extra = len(o_idx) - len(o_slab_idx)
            ref_idx = [i for i in ref_idx if i not in o_idx[-extra:]]

        # Compute fractional shift only using non-organic atoms
        shift = comb_frac[ref_idx] - slab_frac
        shift -= np.round(shift)  # wrap into [-0.5, 0.5] to respect PBC
        global_shift = np.mean(shift, axis=0)
        global_shift[2] = 0.0

        # Remove global shift
        comb_frac_aligned = comb_frac - global_shift
        comb_frac_aligned %= 1.0
        combined.set_scaled_positions(comb_frac_aligned)

        # Update cartesian positions after alignment
        combined_positions = np.array(combined.get_positions())

        # Drop Atom_index and Symbol from original descriptor DataFrames
        surface_cols = [c for c in surface_desc.columns if c.lower() not in ['atom_index', 'symbol']]
        ads_cols = [c for c in ads_desc.columns if c.lower() not in ['atom_index', 'symbol']]

        # Initialize combined dataframe with all columns
        combined_cols = []
        seen = set()
        for col in surface_cols + ads_cols:
            if col not in seen:
                combined_cols.append(col)
                seen.add(col)
        n_combined = len(combined)
        combined_df = pd.DataFrame(index=range(n_combined))
        combined_df['atom_index'] = range(n_combined)
        combined_df['symbol'] = combined_symbols
        for col in combined_cols:
            combined_df[col] = None

        # --- Assign slab descriptors based on position + symbol ---
        slab_taken = set()
        ads_taken = set()
        miss = 0
        cell = slab.get_cell()
        for i, (pos, sym) in enumerate(zip(combined_positions, combined_symbols)):
            same_sym_idx = np.where(slab_symbols == sym)[0]
            best_idx = None
            best_dist = np.inf
            for j in same_sym_idx:
                if j in slab_taken:
                    continue

                pos_j = slab_positions[j]
                _, d = find_mic(pos - pos_j, cell, pbc=True)
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            # print(f"Atom {sym} with index {i} have a best match with {best_idx} with symbol of {slab_symbols[best_idx]}")
            if best_idx is not None and best_dist < 2.0 and slab_symbols[best_idx] == sym and best_idx not in slab_taken:
                # matched slab atom
                slab_taken.add(best_idx)
                desc_row = surface_desc[surface_desc['Atom_index'] == best_idx]
                combined_df.loc[i, surface_desc.columns] = desc_row.iloc[0][surface_cols]
            else:
                if sym in ads_symbols and len(ads_taken) < ads_length:
                    ads_taken.add(i)
                    ave_dist = []
                    for k in ads_taken:
                        ave_dist.append(combined.get_distance(i, k, mic=True))
                    ave_dist = np.mean(np.array(ave_dist))
                    if ave_dist < 2.5:  # if this unmatched atom is close to the adsorbate, assign adsorbate descriptor
                        ads_match_idx = np.where(ads_symbols == sym)[0]
                        if len(ads_match_idx) > 0:
                            desc_row = ads_desc[ads_desc['Atom_index'] == ads_match_idx[0]]
                            combined_df.loc[i, ads_desc.columns] = desc_row.iloc[0][ads_cols]
                    else:
                        ads_taken.remove(i)
                        miss += 1
                else:
                    miss += 1
        if miss > 0:
            print(f"Total {miss} atoms in combined structure were not matched to slab or adsorbate descriptors.")
        
        final_cols = ['atom_index', 'symbol'] + surface_cols + [c for c in ads_cols if c not in surface_cols]
        combined_df = combined_df[final_cols]

        return combined_df

    def plane_aware_placement(self, ads: Atoms, site: dict, tolerance=0.2):
        """
        Generate plane-aware, site-type-aware adsorbate placements for a single site.
        Returns a list of ASE Atoms objects combined with the slab.
        """
        valence_electrons = {
            # Period 1
            "H": 1, "He": 2,
            # Period 2
            "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
            # Period 3
            "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
        }

        def stability_score(symbol):
            ve = valence_electrons[symbol]
            return min(abs(ve - 8), abs(ve))

        placed_adsorbates = []

        site_pos = site["position"]
        surface_indices = site["indices"]
        site_type = site["coordination"]
        if site_type == 1:
            site_type = 'top'
        elif site_type == 2:
            site_type = 'bridge'
        else:
            site_type = 'hollow'
        site_atoms_pos = self.slab[surface_indices].get_positions()
        non_site_atom_pos = np.delete(self.slab.get_positions(), surface_indices, axis=0)
        surf_atom_idx = surface_indices[0]
        surf_symbol = self.symbols[surf_atom_idx]

        # Identify and rotate adsorbate
        def id_rotate_adsorbate(ads: Atoms):
            def rotate(ads: Atoms, bottom_index, top_index=None, step=10, max_steps=36):
                pos = ads.get_positions()

                def orientation_ok(pos, bottom_index, top_index=None):
                    z_bottom = pos[bottom_index, 2]
                    z_diff = z_bottom.max() - z_bottom.min()

                    bottom_flat = z_diff < 5e-2

                    if top_index is None:
                        return bottom_flat

                    z_top = pos[top_index, 2]
                    return bottom_flat and (z_top.mean() > z_bottom.mean())

                if orientation_ok(pos, bottom_index, top_index):
                    return ads

                z_min_prev = pos[bottom_index, 2].mean()
                z_max_prev = pos[top_index, 2].mean() if top_index is not None else None
                
                for _ in range(max_steps):
                    ads.rotate(step, 'x', center=(0,0,0))
                    pos = ads.get_positions()
                    
                    z_min_new = pos[bottom_index, 2].mean()
                    z_diff_new = pos[bottom_index, 2].max() - pos[bottom_index, 2].min() if len(bottom_index) > 1 else 0.0
                    if top_index is not None:
                        z_max_new = pos[top_index, 2].mean()
                    
                    improve_bottom = (z_diff_new < 1e-6) and (z_min_new < z_min_prev)
                    improve_top = True if top_index is None else (z_max_new > z_max_prev)

                    if len(bottom_index) > 1:
                        if improve_bottom and improve_top:
                            break
                        else:
                            z_min_prev = z_min_new
                            z_max_prev = z_max_new if top_index is not None else None
                    else:
                        if not (improve_bottom and improve_top):
                            ads.rotate(-step, 'x', center=(0,0,0))
                            break
                        else:
                            z_min_prev = z_min_new
                            z_max_prev = z_max_new if top_index is not None else None
                return ads
            
            ads_length = len(ads)
            if ads_length == 1:
                bidentate = False
                ads_symbol = ads.get_chemical_symbols()[0]
                bottom_index = [0]
            else:
                ads_symbols_all = np.array(ads.get_chemical_symbols())
                if ads_length == 2 and ads_symbols_all[0] == ads_symbols_all[1]:
                    bidentate = True
                    ads_symbol = ads_symbols_all[0]
                    bottom_index = [0, 1]
                    ads = rotate(ads, bottom_index=bottom_index, step=10, max_steps=36)
                elif ads_length == 2 and ads_symbols_all[0] != ads_symbols_all[1]:
                    bidentate = False
                    # Identify atoms outer orbital stability
                    scores = [stability_score(s) for s in ads_symbols_all]
                    bottom_index = [int(np.argmax(scores))]
                    top_index = [i for i in range(ads_length) if i not in bottom_index]
                    ads_symbol = ads_symbols_all[bottom_index[0]]
                    ads = rotate(ads, bottom_index=bottom_index, top_index=top_index, step=10, max_steps=36)
                else:
                    cnn = CrystalNN()
                    neighbor_map = []
                    struct = AseAtomsAdaptor.get_structure(ads)
                    for i in range(ads_length):
                        nn_data = cnn.get_nn_info(struct, i)
                        neighbor_map.append(len(nn_data))
                    scores = np.array([stability_score(s) for s in ads_symbols_all], dtype=int)
                    neighbor_map = np.array(neighbor_map, dtype=int)
                    while True:
                        # Check if subtracting CN from ALL atoms would keep scores >= 0
                        if np.all(scores - neighbor_map >= 0):
                            scores -= neighbor_map
                        else:
                            break
                    bottom_idx_cand = [i for i, sc in enumerate(scores) if sc == max(scores)]
                    if len(bottom_idx_cand) > 1:
                        bottom_idx_cand = [i for i in bottom_idx_cand if ads_symbols_all[i] != "H"]
                        min_cn = min(neighbor_map[i] for i in bottom_idx_cand)
                        bottom_index = [i for i in bottom_idx_cand if neighbor_map[i] == min_cn]
                    else:
                        bottom_index = bottom_idx_cand

                    if len(bottom_index) == 1:
                        bidentate = False
                    else:
                        bidentate = True
                    ads_symbol = ads_symbols_all[bottom_index[0]]
                    ads = rotate(ads, bottom_index=bottom_index, step=10, max_steps=36)
                    
            return ads, ads_symbol, bidentate, bottom_index
        
        new_ads, ads_symbol, bidentate, bottom_index = id_rotate_adsorbate(ads)

        # Prevent same-atom bonding
        # if surf_symbol == ads_symbol and not bidentate:
        #     return []
        if surf_symbol == ads_symbol:
            return []

        # Z rotations
        def place_on_slab(new_ads, rot_z, tolerance=0.2, site_type="top"):
            rotate_ads = new_ads.copy()
            rotate_ads.rotate(rot_z, "z", center=(0, 0, 0))
            ads_positions = rotate_ads.get_positions()
            # Bond height
            height = covalent_radii[atomic_numbers[surf_symbol]] + covalent_radii[atomic_numbers[ads_symbol]] + tolerance
            # Translation
            ads_xy_mean = ads_positions[:, :2].mean(axis=0)
            ads_bottom_z = ads_positions[bottom_index, 2].min()
            dx = site_pos[0] - ads_xy_mean[0]
            dy = site_pos[1] - ads_xy_mean[1]
            dz = (site_pos[2] + height) - ads_bottom_z
            rotate_ads.translate([dx, dy, dz])

            # Check distances of translated adsorbate to surface atoms
            ads_positions = rotate_ads.get_positions()
            min_z = ads_positions[:, 2].min()
            bottom_mask = np.isclose(ads_positions[:, 2], min_z, atol=1e-6)
            bottom_pos = ads_positions[bottom_mask]
            
            # Pairwise distances: site atoms ↔ bottom adsorbate atoms
            diffs = site_atoms_pos[:, None, :] - bottom_pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[i, j]
            r_vec = diffs[i, j]
            cos_theta = abs(r_vec[2]) / max(min_dist, 1e-8)

            # Z-only correction
            if abs(min_dist - height) > tolerance:
                z_center = rotate_ads.get_center_of_mass()[2]
                dz_corr = ((height - min_dist) / cos_theta) - (z_center - ads_positions[:, 2].min())
                rotate_ads.translate([0.0, 0.0, dz_corr])

            # --- distance score using actual placed geometry ---
            placed_p = rotate_ads.get_positions()
            # heavy → nearest surface atom distance
            score = 0.0
            for h in bottom_index:
                d = np.linalg.norm(site_atoms_pos[:] - placed_p[h], axis=1)
                d_non = np.linalg.norm(non_site_atom_pos[:] - placed_p[h], axis=1)
                if site_type != "top" and not bidentate:
                    score += np.mean(d)
                else:
                    score += np.min(d)
                score -= np.min(d_non)
                # Distance from other adsorbate atoms to surface atoms
                for ah in range(len(placed_p)):
                    if ah in bottom_index:
                        continue
                    if np.linalg.norm(placed_p[ah, 2] - placed_p[h, 2]) < 2*covalent_radii[atomic_numbers[ads_symbol]]:
                        d_non_ads = np.linalg.norm(non_site_atom_pos[:] - placed_p[ah], axis=1)
                        score -= np.min(d_non_ads)

            # --- Crowding adjustment for non-heavy atoms ---
            non_heavy_indices = [i for i in range(len(placed_p)) if i not in bottom_index]
            for nh in non_heavy_indices:
                # Vector from each surface atom to this non-heavy atom
                vecs_xy = placed_p[nh, :2] - site_atoms_pos[:, :2]
                dists_xy = np.linalg.norm(vecs_xy, axis=1)
                # Indices of surface atoms that are too close
                close_idx = np.where(dists_xy < 1.5)[0]

                for ci in close_idx:
                    # Skip if this is bonded heavy atom (along Z)
                    if np.abs(placed_p[nh, 2] - site_atoms_pos[ci, 2]) < 0.5:
                        continue

                    # Compute rotation axis: perpendicular to XY vector toward surface atom
                    axis = np.array([vecs_xy[ci, 1], -vecs_xy[ci, 0], 0.0])
                    if np.linalg.norm(axis) < 1e-8:
                        continue
                    axis /= np.linalg.norm(axis)
                    # Rotate small step (15°) to push atom away
                    rotate_ads.rotate(15.0, axis, center=placed_p[nh])
                    placed_p = rotate_ads.get_positions()  # update positions for next check

            # Combine with slab
            combined = self.slab + rotate_ads
            combined.pbc = (True, True, True)
            combined.wrap()

            # Post-process: adsorbate–surface distance check ---
            n_ads = len(rotate_ads)
            n_tot = len(combined)
            ads_indices = []
            surf_indices = []
            ads_indices = np.arange(n_tot - n_ads, n_tot)
            surf_indices = np.arange(0, n_tot - n_ads)

            ads_pos = combined.get_positions()[ads_indices]
            surf_pos = combined.get_positions()[surf_indices]

            ads_syms = np.array(combined.get_chemical_symbols())[ads_indices]
            surf_syms = np.array(combined.get_chemical_symbols())[surf_indices]

            # Pairwise distances: adsorbate ↔ surface
            diffs = ads_pos[:, None, :] - surf_pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)

            # Find closest pair
            i_ads, i_surf = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[i_ads, i_surf]

            ads_sym_close = ads_syms[i_ads]
            surf_sym_close = surf_syms[i_surf]
            r_ads = covalent_radii[atomic_numbers[ads_sym_close]]
            r_surf = covalent_radii[atomic_numbers[surf_sym_close]]

            # Bond threshold
            height_thresh = (r_surf + r_ads + tolerance)

            # --- Reject if same-element clash and distances closser than covalen radii ---
            if ads_sym_close == surf_sym_close and min_dist < (r_ads + r_surf):
                return None, np.inf, None   # drop this adsorption config

            # --- Lift adsorbate in +z until distance is acceptable ---
            dz = 0.0
            if min_dist < height_thresh:
                dz = height_thresh - min_dist

            # Translate ONLY adsorbate atoms
            shift = np.zeros((len(combined), 3))
            shift[ads_indices, 2] += dz
            combined.set_positions(
                combined.get_positions() + shift
            )
            placed_p = combined.get_positions()[ads_indices]
            combined_sorted = sort(combined)

            return combined_sorted, score, placed_p

        if not bidentate or site_type == "top":
            best_geom = None
            best_score = np.inf
            for deg in range(0, 360, 10):
                combined, score, _ = place_on_slab(new_ads, deg, tolerance=tolerance, site_type=site_type)
                if score < best_score:
                    best_score = score
                    best_geom = combined
            placed_adsorbates.append(best_geom)
        elif site_type == "bridge":
            # Bidentate on bridge
            best_geom = None
            best_score = np.inf
            for deg in range(0, 360, 10):   # coarse scan every 10°
                combined, score, _ = place_on_slab(new_ads, deg, tolerance=tolerance, site_type=site_type)
                if score < best_score:
                    best_score = score
                    best_geom = combined
            placed_adsorbates.append(best_geom)
        elif site_type == "hollow":
            # Bidentate on hollow
            for i in range(len(site_atoms_pos)):   # loop over all surface atoms that make the hollow site
                best_geom_i = None
                best_score_i = np.inf
                for deg in range(0, 360, 10):  # 10° scan
                    combined, _, placed_p = place_on_slab(new_ads, deg, tolerance=tolerance, site_type=site_type)
                    if combined is None or placed_p is None:
                        continue
                    h0 = bottom_index[0]
                    h1 = bottom_index[1]
                    # scoring attraction to site atom i
                    v1 = np.linalg.norm(site_atoms_pos[i][:2] - placed_p[h0][:2])
                    v2 = np.linalg.norm(placed_p[h1][:2] - site_atoms_pos[i][:2])
                    score = v1 + v2
                    if score < best_score_i:
                        best_score_i = score
                        best_geom_i = combined
                placed_adsorbates.append(best_geom_i)

        return placed_adsorbates

    def _add_adsorbate(self, adsorbate: Atoms, sites: list, tolerance=0.2):
        """
        Place an adsorbate on all specified adsorption sites.

        Args:
            adsorbate: ASE Atoms object of the adsorbate
            sites: list of dicts (output of AdsorptionSite._ads_site_detect)
            tolerance: extra spacing added to bond distance

        Returns:
            all_structures: list of ASE Atoms objects with adsorbate
            all_descriptors: list of DataFrames corresponding to each structure (optional)
        """
        all_structures = []
        all_descriptors = []

        ads = adsorbate.copy()
        ads.translate(-ads.get_center_of_mass()) # Center it around origin

        for site in sites:
            placed_ads_list = self.plane_aware_placement(ads, site, tolerance=tolerance)
            
            for placed_ads in placed_ads_list:
                if placed_ads is None:
                    continue
                all_structures.append(placed_ads)

                # Update descriptors if available
                if self.surface_desc is not None or self.ads_desc is not None:
                    combined_desc = self._combine_descriptors(self.slab, ads, placed_ads, self.surface_desc, self.ads_desc)
                    all_descriptors.append(combined_desc)

        if self.surface_desc is None and self.ads_desc is None:
            return all_structures
        else:
            return all_structures, all_descriptors