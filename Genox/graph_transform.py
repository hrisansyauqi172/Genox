import re
import itertools
import os
import json
import io
import pandas as pd
import numpy as np
import networkx as nx
from ase import Atoms, Atom
from ase.io import read
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_distances, find_mic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.bonds import CovalentBond
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# --- Load gas-phase database on import ---
current_dir = os.path.dirname(__file__)
gas_db_path = os.path.join(current_dir, "ads_database.json")  # database in the same folder as Genox
gas_phase_db = {}

if os.path.isfile(gas_db_path):
    with open(gas_db_path, "r") as f:
        db_list = json.load(f)
    for entry in db_list:
        label = entry["ads_label"]
        ads_str = entry["structure"]
        atoms_obj = read(io.StringIO(ads_str), format="vasp")
        gas_phase_db[label] = atoms_obj
else:
    print(f"[Genox] Warning: gas-phase database not found at {gas_db_path}.")

class GraphTransform:
    def __init__(self, atoms, gas_db=None):
        self.atoms = atoms
        self.struct = AseAtomsAdaptor.get_structure(atoms)
        self.positions = atoms.get_positions()
        self.symbols = atoms.get_chemical_symbols()
        self.n_atoms = len(atoms)
        self.cnn = CrystalNN()
        self.nn_info = []
        for i in range(self.n_atoms):
            try:
                neigh = self.cnn.get_nn_info(self.struct, i)
            except ValueError:
                # No Voronoi neighbors → isolated atom
                neigh = []
            self.nn_info.append(neigh)
        self.cn = [len(self.nn_info[i]) for i in range(self.n_atoms)]
        self.gas_phase_db = gas_db if gas_db is not None else gas_phase_db

    # --------------------------------------------------------------
    #  Adjacency matrix builder
    # --------------------------------------------------------------
    def _build_adjacency(self):
        """
        Build adjacency matrix using Voronoi-based CrystalNN (not simple cutoffs) 
        or fallback to using covalent-radii.
        Returns True if atoms are bonded according to CrystalNN.
        """
        struct = self.struct
        n_atoms = self.n_atoms
        adjacency = np.zeros((n_atoms, n_atoms), dtype=bool)

        # --- CrystalNN neighbors ---
        zero_cn_indices = []
        for i in range(n_atoms):
            neighs = self.nn_info[i]
            if len(neighs) == 0:
                zero_cn_indices.append(i)
            for n in neighs:
                j = n['site_index']
                adjacency[i, j] = adjacency[j, i] = True

        # -- Handle zero-CN atoms by checking covalent bonds ---
        for i in zero_cn_indices:
            if len(zero_cn_indices) == 1:
                for j in range(n_atoms):
                    if i == j or adjacency[i, j]:
                        continue
                    
                    try:
                        if CovalentBond.is_bonded(struct[i], struct[j], tol=0.3):
                            adjacency[i, j] = adjacency[j, i] = True
                            continue
                    except Exception:
                        pass
                    
                    d = self.atoms.get_distance(i, j, mic=True)
                    r_i = covalent_radii[self.atoms[i].number]
                    r_j = covalent_radii[self.atoms[j].number]

                    # allow small tolerance (1.15 - 1.3)
                    if d < ((r_i + r_j) * 1.15):
                        adjacency[i, j] = adjacency[j, i] = True
            else:
                for j in zero_cn_indices:
                    if i >= j:
                        continue
                    try:
                        if CovalentBond.is_bonded(struct[i], struct[j], tol=0.3):
                            adjacency[i, j] = adjacency[j, i] = True
                            continue
                    except Exception:
                        pass

                    d = self.atoms.get_distance(i, j, mic=True)
                    r_i = covalent_radii[self.atoms[i].number]
                    r_j = covalent_radii[self.atoms[j].number]

                    # allow small tolerance (1.15 - 1.3)
                    if d < ((r_i + r_j) * 1.15):
                        adjacency[i, j] = adjacency[j, i] = True

        return adjacency
    
    # --------------------------------------------------------------
    #  Simple clustering helper
    # --------------------------------------------------------------
    def _cluster_ads(self, indices):
        """
        Cluster atoms based on a distance threshold.
        Returns a list of clusters, each as a list of atom indices.
        """
        clusters = []
        unassigned = set(indices)
        atoms = self.atoms
        
        while unassigned:
            current = unassigned.pop()
            cluster = [current]
            added = True
            while added:
                added = False
                cluster_positions = atoms.positions[cluster]
                cluster_center = cluster_positions.mean(axis=0)
                
                for i in list(unassigned):
                    dist_to_any = np.min([atoms.get_distance(i, j, mic=True) for j in cluster])
                    dist_to_center = np.linalg.norm(atoms.positions[i] - cluster_center)

                    if dist_to_any < 1.6 and dist_to_center < 1.8:
                        cluster.append(i)
                        unassigned.remove(i)
                        added = True
            clusters.append(cluster)
        return clusters
    
    # --------------------------------------------------------------
    #  Detect likely adsorbate atoms based on CN and height
    # --------------------------------------------------------------
    def _top_candidate(self):
        # --- Compute z-range statistics ---
        z_positions = self.positions[:, 2]
        z_base = min(z_positions)
        z_range = max(z_positions) - z_base
        z_cutoff = z_base + (z_range * 0.75)  # e.g., 80th percentile if z_tail_fraction=0.2
        z_low = z_base + (z_range * 0.25)
        z_evaluated = np.where(z_positions > z_cutoff)[0]
        z_bulk = np.where((z_positions >= z_low) & (z_positions <= z_cutoff))[0]
        
        # --- Compute coordination numbers using CrystalNN ---
        cn = np.array(self.cn)
        symbols = self.symbols

        # --- Compute reference (bulk) CN per element ---
        bulk_cn_ref = {}
        for sym in set(symbols):
            bulk_atoms = [i for i in z_bulk if symbols[i] == sym]
            if bulk_atoms:
                bulk_cn_ref[sym] = np.max([cn[i] for i in bulk_atoms])

        # --- Compare surface CN vs bulk CN (start from the heighest atom) ---
        low_cn_indices = []
        for i in z_evaluated[::-1]:  # start from highest z
            sym = symbols[i]
            if sym not in bulk_cn_ref:
                # Element not present in bulk → likely adsorbate
                low_cn_indices.append(int(i))
            else:
                if bulk_cn_ref[sym] - cn[i] >= 1:
                    low_cn_indices.append(int(i))

        # --- Add rare elements (isolated adsorbates like single H, C, O, etc. or dopant atoms) ---
        symbol_counts = Counter(symbols[i] for i in z_evaluated)
        low_cn_indices += [int(i) for i in z_evaluated if symbol_counts[symbols[i]] <= 2 and int(i) not in low_cn_indices]

        return low_cn_indices
    
    # --------------------------------------------------------------
    #  Detect adsorbate by matching formula
    # --------------------------------------------------------------
    def _detect_adsorbates(self, adsorbate_label):
        """Detect adsorbate atoms via connected components."""

        # Parse label, e.g., '2H' -> ['H','H'], 'HCOO' -> ['H','C','O','O']
        parsed_symbols = []
        pattern = r'(\d*)([A-Z][a-z]?)(\d*)'
        for match in re.finditer(pattern, adsorbate_label.strip()):
            count1, sym, count2 = match.groups()
            count = count1 + count2
            if sym:  # skip empty matches
                parsed_symbols.extend([sym] * (int(count) if count else 1))
        label_counts = Counter(parsed_symbols)
        
        # Candidate atoms based on height and coordination
        candidate_indices = self._top_candidate()
        new_label_counts = Counter()
        store = []
        for i in candidate_indices:
            sym = self.symbols[i]
            if sym not in label_counts:
                continue
            if new_label_counts[sym] < label_counts[sym]:
                new_label_counts[sym] += 1
                store.append(int(i))
            if all(new_label_counts[s] == label_counts[s] for s in label_counts):
                break
        candidate_indices = store

        # Initial clustering
        clusters = self._cluster_ads(candidate_indices)
        clusters = sorted([list(set(c)) for c in clusters], key=lambda c: -len(c))
        
        ads_mask = np.zeros(self.n_atoms, dtype=bool)
        for c in clusters:
            ads_mask[c] = True
        best_ads_mask = ads_mask.copy()
        best_num_clusters = len(clusters)

        def total_composition(mask):
            return Counter(self.symbols[i] for i in np.where(mask)[0])
        
        # Cluster refinement loop
        while len(clusters) >= 1 and np.any(ads_mask):
            updated = False
            needed = label_counts.copy()
            for iter, cluster in enumerate(clusters):
                cluster_counts = Counter(self.symbols[i] for i in cluster)
                if cluster_counts == label_counts:
                    # Perfect match, delete other clusters from ads_mask
                    ads_mask[:] = False
                    ads_mask[cluster] = True
                    updated = True
                    break  # already matches label

                # If cluster has less atoms than label, try to add neighboring atoms
                if any(cluster_counts.get(s, 0) < label_counts[s] for s in label_counts):
                    needed = needed - Counter(self.symbols[i] for i in cluster)
                    for needed_sym in needed:
                        if needed[needed_sym] <= 0:
                            continue
                        for i in cluster:
                            neighbors_info = self.nn_info[i]
                            neighbor_candidates = [
                                n['site_index'] for n in neighbors_info
                                if self.symbols[n['site_index']] == needed_sym and not ads_mask[n['site_index']]
                            ]
                            if not neighbor_candidates:
                                continue
                            
                            # Only consider if more neighbors than needed
                            if len(neighbor_candidates) > needed[needed_sym]:
                                # Compute CN and uniqueness score
                                candidate_scores = []
                                for j in neighbor_candidates:
                                    cn_j = self.cn[j]
                                    unique_score = len(set(self.symbols[n['site_index']] for n in self.nn_info[j]))
                                    candidate_scores.append((j, cn_j, unique_score))
                                candidate_scores.sort(key=lambda x: (x[1], -x[2]))
                                selected_candidates = [x[0] for x in candidate_scores[:needed[needed_sym]]]
                            else:
                                selected_candidates = neighbor_candidates
                            
                            for j in selected_candidates:
                                ads_mask[j] = True
                                updated = True
                                needed[needed_sym] -= 1
                                if needed[needed_sym] <= 0:
                                    break
                
                # Remove excess atoms
                else:
                    for sym in cluster_counts:
                        if cluster_counts[sym] > label_counts.get(sym, 0):
                            excess = cluster_counts[sym] - label_counts.get(sym, 0)
                            sym_atoms = [i for i in cluster if self.symbols[i] == sym]

                            def uniqueness_score(i):
                                return len(set(self.symbols[n['site_index']] for n in self.nn_info[i]))

                            # Sort by z (ascending), then CN descending, then uniqueness ascending
                            sym_atoms_sorted = sorted(
                                sym_atoms,
                                key=lambda x: (
                                    self.atoms.positions[x][2],              # lower z first
                                    -self.cn[x],   # higher CN first
                                    uniqueness_score(x)                 # less unique first
                                )
                            )

                            for i in sym_atoms_sorted:
                                if excess <= 0:
                                    break
                                ads_mask[i] = False
                                updated = True
                                excess -= 1
                
                # Check if current clusters already satisfy label_counts
                selected_indices = [i for c in clusters[:iter] for i in c]
                if total_composition(ads_mask[selected_indices]) == label_counts:
                    ads_mask[:] = False
                    for c in clusters[:iter]:
                        ads_mask[c] = True
                    break
            
            # --- Recompute clusters ---
            new_ads_indices = np.where(ads_mask)[0]
            clusters = self._cluster_ads(new_ads_indices)

            # --- Store best state if improvement achieved ---
            if 0 < len(clusters) < best_num_clusters and total_composition(ads_mask) == label_counts:
                best_num_clusters = len(clusters)
                best_ads_mask = ads_mask.copy()

            if not updated or not np.any(ads_mask) or (len(clusters) == best_num_clusters and total_composition(ads_mask) == label_counts):
                break

        # --- Final fallback ---
        if len(clusters) > 1 and np.any(best_ads_mask) and total_composition(best_ads_mask) == label_counts:
            ads_mask = best_ads_mask

        return ads_mask
    
    def _adjust_adsorbate_descriptors(self, adsorbate_label, ads_mask, descriptor_df, 
                                      beta=0.5, lambda_corr=0.3, sigma_d=0.4, sigma_theta=0.2, alpha=1.5, return_scale=False):
        # --- Safety check ---
        ads_indices = np.where(ads_mask)[0]
        if len(ads_indices) == 0:
            return descriptor_df.copy()

        ads_atoms = self.atoms[ads_indices].copy()
        ref_pos = ads_atoms.positions[0].copy()
        unwrapped = [ref_pos]
        for i in range(1, len(ads_atoms)):
            diff = ads_atoms.positions[i] - ref_pos
            mic_diff, _ = find_mic(diff.reshape(1, 3), ads_atoms.cell, ads_atoms.pbc)
            unwrapped.append(ref_pos + mic_diff[0])
        ads_atoms.positions = np.array(unwrapped)
        ads_atoms.center()
        new_df = descriptor_df.copy()

        # --- Helper: select reference ASE atoms ---
        def select_reference(adsorbate_label):
            # Default: use specific adsorbate
            if adsorbate_label in ["2H"]:
                adsorbate_label = "H2"
            else:
                adsorbate_label = adsorbate_label
            return self.gas_phase_db.get(adsorbate_label)
        
        # --- Helper: SOAP-like geometric similarity ---
        def geom_similarity(gas_atoms: Atoms, ads_atoms: Atoms, sigma_d=0.4, sigma_theta=0.2, return_scale=False):
            # --- Element-aware Hungarian assignment ---
            gas_syms = np.array(gas_atoms.get_chemical_symbols())
            ads_syms = np.array(ads_atoms.get_chemical_symbols())
    
            # Match atoms element by element
            gas_order = []
            ads_order = []
            for element in np.unique(gas_syms):
                gas_idx = np.where(gas_syms == element)[0]
                ads_idx = np.where(ads_syms == element)[0]     
                if len(ads_idx) == 0:
                    print(f"WARNING: element {element} in gas but not in ads")
                    continue
                n_match = min(len(gas_idx), len(ads_idx))
                # Hungarian within same element
                cost = cdist(gas_atoms.positions[gas_idx], ads_atoms.positions[ads_idx])
                row_ind, col_ind = linear_sum_assignment(cost)
                gas_order.extend(gas_idx[row_ind[:n_match]])
                ads_order.extend(ads_idx[col_ind[:n_match]])
            gas_order = np.array(gas_order)
            ads_order = np.array(ads_order)
            n = len(gas_order)
    
            gas_positions = gas_atoms.positions[gas_order]
            ads_positions = ads_atoms.positions[ads_order]

            # Build adjacency matrices and reorder to matched atom ordering
            gas_graph = GraphTransform(gas_atoms)
            gas_adj = gas_graph._build_adjacency()
            gas_adj = gas_adj[np.ix_(gas_order, gas_order)]

            # Distance similarity
            bond_pairs = np.argwhere(np.triu(gas_adj, k=1))
            bond_change = 0.0
            for i, j in bond_pairs:
                dg = np.linalg.norm(gas_positions[i] - gas_positions[j])
                da = np.linalg.norm(ads_positions[i] - ads_positions[j])
                bond_change += (dg - da) ** 2
            S_d = np.exp(-bond_change / sigma_d**2)

            # Angle similarity
            theta_change = 0.0
            for j in range(n):
                neighbors = np.where(gas_adj[j])[0]
                if len(neighbors) < 2:
                    continue
                for i, k in itertools.combinations(neighbors, 2):
                    # Gas-phase vectors
                    v_ij_g = gas_positions[i] - gas_positions[j]
                    v_kj_g = gas_positions[k] - gas_positions[j]
                    theta_g = np.arccos(np.clip(np.dot(v_ij_g, v_kj_g) /
                                        (np.linalg.norm(v_ij_g) * np.linalg.norm(v_kj_g) + 1e-8), -1.0, 1.0))

                    # Adsorbed vectors
                    v_ij_a = ads_positions[i] - ads_positions[j]
                    v_kj_a = ads_positions[k] - ads_positions[j]
                    theta_a = np.arccos(np.clip(np.dot(v_ij_a, v_kj_a) /
                                        (np.linalg.norm(v_ij_a) * np.linalg.norm(v_kj_a) + 1e-8), -1.0, 1.0))
                    theta_change += (theta_a - theta_g) ** 2

            S_theta = np.exp(-theta_change / sigma_theta**2)

            if return_scale:
                return float(S_d * S_theta), S_d, S_theta
            else:
                return float(S_d * S_theta)

        # --- Helper: inertia-based topological change weight ---
        def inertia_weight(gas_atoms: Atoms, ads_atoms: Atoms, alpha=1.5):
            Ig = np.sort(gas_atoms.get_moments_of_inertia(vectors=False))
            Ia = np.sort(ads_atoms.get_moments_of_inertia(vectors=False))
            n = min(len(Ig), len(Ia))
            delta = np.linalg.norm(Ia[:n] - Ig[:n]) / (np.linalg.norm(Ig[:n]) + 1e-8)
            return float(np.tanh(alpha * delta))
        
        # --- Loop over relevant descriptors ---
        for col in descriptor_df.columns:
            if col not in ["HOMO_rel", "LUMO_rel", "p_orbital", "s_orbital", "E-fermi"]:
                continue

            # 1. Get reference structure
            gas_ref = select_reference(adsorbate_label)

            # 2. Compute similarity metrics
            if return_scale:
                S_geom, S_d, S_theta = geom_similarity(gas_ref, ads_atoms, sigma_d=sigma_d, sigma_theta=sigma_theta, return_scale=True)
            else:
                S_geom = geom_similarity(gas_ref, ads_atoms, sigma_d=sigma_d, sigma_theta=sigma_theta)
            w_inertia = inertia_weight(gas_ref, ads_atoms, alpha=alpha)

            # 3. Compute scaling factor
            scale = max(0.0, S_geom * np.exp(-beta * w_inertia))

            # 5. Apply scaling only to adsorbate atoms
            for i in ads_indices:
                val = descriptor_df.loc[i, col]
                if pd.notna(val):
                    delta = lambda_corr * (scale - 1.0) * abs(val)
                    new_df.loc[i, col] = float(val + delta)

        if return_scale:
            var_scale = [delta, scale, w_inertia, S_geom, S_d, S_theta]
            return new_df, var_scale
        else:
            return new_df
    
    def _artif_node_feat(self, artificial_idx, adjacency, feature_values, rbf_sigma=0.8):
        """
        Compute SOAP-like effective features for an artificial adsorbate node.
        Produces a normalized scalar Z_eff using a neighbor-density kernel.

        Inputs:
        self.atoms : ASE Atoms
            Needed to compute PBC-correct distances via atoms.get_distance().
        artificial_idx : int
            Index of the artificial node.
        adjacency : (N,N) bool array
            Adjacency matrix determining neighbors.
        feature_values : array-like (length N)
            Feature values for all real atoms. Should include a placeholder for artificial node.
            - For atomic number: feature_values = atomic_numbers_list
            - For descriptor columns: feature_values = descriptor_df[col].values
        rbf_sigma : float
            Width of the Gaussian RBF kernel.

        Returns
        -------
        feat_eff : float
            SOAP-normalized effective feature value for artificial node.
        """

        # ---- Select neighbors of the artificial node ----
        neighbor_indices = np.where(adjacency[artificial_idx])[0]
        if len(neighbor_indices) == 0:
            # fallback: no neighbors → assign mean Z
            return float(np.mean([v for v in feature_values if np.isfinite(v)]))
        
        # ---- Compute weights w ----
        w = np.array([feature_values[j] if np.isfinite(feature_values[j]) else 0.0 
                      for j in neighbor_indices], dtype=float)
        
        # ---- Compute RBF distances ----
        rbf_list = []
        for j in neighbor_indices:
            d = self.atoms.get_distance(artificial_idx, j, mic=True)
            rbf_list.append(np.exp(-(d**2) / (2.0 * rbf_sigma**2)))
        rbf = np.array(rbf_list, dtype=float)
        norm = np.linalg.norm(rbf)
        if norm < 1e-14:
            return float(np.sum(w))  # fallback
        rbf_norm = rbf / norm

        # ---- SOAP-like local density ----
        feat_eff = float(np.sum(w * rbf_norm))
        return feat_eff
    
    @staticmethod
    def _make_graph(atoms_init, adsorbate_label=None, ads_dist_threshold=3.0, descriptor_df: pd.DataFrame = None,
                    ads_site=None, adj_ads_desc=False, gas_db=None, inner_weight=10, outer_weight=0.5,
                    rbf_sigma=0.8, beta=0.5, lambda_corr=0.3, sigma_d=0.4, sigma_theta=0.2, alpha=1.5):
        """
        Build a Networkx graph for oxide or mixed surfaces.
        Automatically identifies adsorbate vs. surface atoms using iterative connectivity (cluster) detection, not element type or height.

        If adsorbate label is not given then create an artificial node for adsorbate based on site information.

        Node attributes:
          - symbol: atomic symbol
          - is_adsorbate: classified whether the node is surface or adsorbate
          - base_weight: proximity of the nodes to the adsorbate weighted using harmonic average of atomic numbers
          - descriptors attributes
        
        Edge attributes:
          - bond_type: surface-surface, surface-adsorbate, adsorbate-adsorbate
          - length: interatomic distance (Å)
          - weight: proximity of the nodes to the adsorbate weighted using harmonic average of atomic numbers
          - category: classified whether the edge within adsorbate sphere cutoff
        """
        # --- Check adsorbate or create artificial adsorbate
        atoms = atoms_init.copy()
        artificial_adsorbate = False

        if adsorbate_label is None:
            ads_pos = np.array(ads_site)

            if ads_pos.ndim == 2 and ads_pos.shape[0] == 1:
                ads_pos = ads_pos[0]

            # find nearest real atom in xy-plane
            diffs = atoms.positions - ads_pos
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_Z = atoms[nearest_idx].number

            required_dist = covalent_radii[nearest_Z] + covalent_radii[atomic_numbers["N"]]
            current_dist = dists[nearest_idx]

            # final adsorption position
            if current_dist < required_dist:
                dx, dy = diffs[nearest_idx][:2]
                xy_sq = dx*dx + dy*dy
                dz = np.sqrt(max(required_dist**2 - xy_sq, 0.0))
                ads_pos[2] += dz

            # --- Add artifical adsorbate to ASE ---
            X_atom = Atom("N", position=ads_pos)
            atoms += X_atom
            artificial_adsorbate = True
        
        gt = GraphTransform(atoms, gas_db=gas_db)
        symbols = gt.symbols
        positions = gt.positions
        n_atoms = gt.n_atoms

        # --- Create adsorbate maps ---
        ads_mask = np.zeros(n_atoms, dtype=bool)
        if adsorbate_label:
            ads_mask = gt._detect_adsorbates(adsorbate_label)
        else:
            ads_mask[n_atoms-1] = True

        # --- Build adjacency ---
        adjacency = gt._build_adjacency()

        # --- Prepare node weight and edge info ---
        atomic_numbers_list = [atoms[i].number for i in range(n_atoms)]
        if artificial_adsorbate:
            artificial_idx = n_atoms - 1
            # Call SOAP feature builder
            Z_eff = gt._artif_node_feat(
                artificial_idx=artificial_idx,
                adjacency=adjacency,
                feature_values=np.array(atomic_numbers_list, dtype=float),
                rbf_sigma=rbf_sigma
            )
            # Overwrite artificial node atomic number
            atomic_numbers_list[artificial_idx] = Z_eff
        node_weights = []
        node_categories = []
        ads_distances = []
        edges = []
        edge_attrs = []
        ads_indices = np.where(ads_mask)[0]
        
        for i in range(n_atoms):
            ads_dists = []
            for ads_idx in ads_indices:
                _, ads_d = get_distances(positions[i], positions[ads_idx],
                         cell=atoms.cell, pbc=atoms.pbc)
                ads_dists.append(ads_d[0][0])
            ads_dist = np.min(ads_dists)

            # --- Compute node base weight using harmonic mean with neighbors ---
            neighbor_indices = np.where(adjacency[i])[0]
            Zi = atomic_numbers_list[i]

            if len(neighbor_indices) > 0:
                Z_neighbors = [atomic_numbers_list[j] for j in neighbor_indices]
                Zavg_neighbors = np.mean(Z_neighbors)
            else:
                Zavg_neighbors = Zi  # fallback for isolated atom

            base_node_weight = 2.0 / ((1.0 / Zi) + (1.0 / Zavg_neighbors))

            # Distance-dependent modulation (same scheme as edges)
            if ads_dist < ads_dist_threshold:
                node_weight = inner_weight * base_node_weight
                node_category = 1       # "important"
            else:
                node_weight = outer_weight**ads_dist * base_node_weight
                node_category = 0       # "nothing"

            node_weights.append(node_weight)
            node_categories.append(node_category)
            ads_distances.append(ads_dist)

            # --- Compute edge info for this node ---
            for j in range(i + 1, n_atoms):
                if not adjacency[i, j]:
                    continue

                _, d = get_distances(positions[i], positions[j],
                                     cell=atoms.cell, pbc=atoms.pbc)
                dist = d[0][0]

                # Determine bond type
                if ads_mask[i] and ads_mask[j]:
                    bond_type = "adsorbate-adsorbate"
                elif ads_mask[i] or ads_mask[j]:
                    bond_type = "surface-adsorbate"
                else:
                    bond_type = "surface-surface"

                # --- Compute edge weight ---
                Zi, Zj = atomic_numbers_list[i], atomic_numbers_list[j]
                
                base_weight = 2.0 / (1.0 / Zi + 1.0 / Zj)

                # Adsorbate distance dependence
                if ads_dist < ads_dist_threshold:
                    weight = inner_weight * base_weight
                    category = "important"
                else:
                    weight = outer_weight**ads_dist * base_weight
                    category = "nothing"

                edges.append((i, j))
                edge_attrs.append({
                    "bond_type": bond_type,
                    "length": dist,
                    "weight": weight,
                    "category": category
                })
        
        # Convert artificial adsorbate symbol to "X"
        if artificial_adsorbate:
            symbols[n_atoms-1] = "X"
        
        # Graph transformation
        G_nx = nx.Graph()
        G_nx.graph['cell'] = atoms.cell.tolist()
        G_nx.graph['pbc']  = atoms.pbc.tolist()
        descriptor_cols = []
        # --- Add node descriptors ---
        if descriptor_df is not None:
            # Ensure descriptor_df matches original slab length
            expected_len = n_atoms if not artificial_adsorbate else n_atoms - 1
            if len(descriptor_df) != expected_len:
                raise ValueError("Descriptor DataFrame length must match number of atoms.")
            skip_cols = {'atom_index', 'symbol'}
            descriptor_cols = [c for c in descriptor_df.columns if c.lower() not in skip_cols]
            if adj_ads_desc and not artificial_adsorbate:
                descriptor_df = gt._adjust_adsorbate_descriptors(adsorbate_label, ads_mask, descriptor_df, 
                                                                 beta=beta, lambda_corr=lambda_corr, sigma_d=sigma_d, sigma_theta=sigma_theta, alpha=alpha)
        else:
            descriptor_cols = ["atomic_number"]

        for i, sym in enumerate(symbols):
            G_nx.add_node(i,
                          symbol=sym,
                          is_adsorbate=bool(ads_mask[i]),
                          node_category=node_categories[i],
                          ads_distance=ads_distances[i],
                          position=str(positions[i].tolist()),
                          base_weight=node_weights[i])
            # Add node-level descriptors (excluding atom_index and symbol)
            # Artificial adsorbate case: assign NaN descriptors
            for col in descriptor_cols:
                if artificial_adsorbate and i == n_atoms - 1:
                    # Use SOAP-like weighting to generate artificial descriptor value
                    feat_eff = gt._artif_node_feat(
                        artificial_idx=i,
                        adjacency=adjacency,
                        feature_values=np.array(descriptor_df[col].values, dtype=float)
                    )
                    G_nx.nodes[i][col] = float(feat_eff)
                else:
                    val = descriptor_df.loc[i, col] if (descriptor_df is not None and i in descriptor_df.index) else atomic_numbers_list[i]
                    if pd.isna(val) or val is None:
                        val = 0.0
                    G_nx.nodes[i][col] = float(val)
        
        for (u, v), attr in zip(edges, edge_attrs):
            G_nx.add_edge(u, v, **attr)

        return G_nx
    
    @staticmethod
    def _graph_to_atoms(G_nx: nx.Graph):
        """
        Convert a NetworkX graph back to ASE Atoms object.
        
        Each node in G has:
            - 'symbol': chemical symbol
            - 'position': 3D coordinates (numpy array or list)
            - optionally other node attributes like 'embedding'

        Optionally, the graph can store periodic cell information in G.graph['cell']

        Returns:
            atoms : ASE Atoms object
        """
        symbols = []
        positions = []

        for node in G_nx.nodes:
            node_data = G_nx.nodes[node]
            symbols.append(node_data.get("symbol", "X"))  # default 'X' if element missing
            positions.append(node_data.get("position", np.zeros(3)))

        atoms = Atoms(symbols=symbols, positions=np.array(positions))

        # Set periodic cell if available
        if "cell" in G_nx.graph:
            atoms.set_cell(G_nx.graph["cell"])
            atoms.set_pbc(G_nx.graph['pbc'])

        return atoms