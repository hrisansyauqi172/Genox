import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from ase.data import atomic_numbers
from ase.geometry import get_distances
from dscribe.descriptors import SOAP, ACSF
from dscribe.kernels import REMatchKernel, AverageKernel
from Genox import GraphTransform

class SimilarityAnalysis:
    @staticmethod
    def _create_descriptors(atoms, soap_r_cut=6.0, soap_n_max=8, soap_l_max=6,
                            acsf_r_cut=6.0, species=None, g2_params=None, g4_params=None):
        """
        Compute SOAP and ACSF descriptors for each atom in an ASE Atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        soap_r_cut, soap_n_max, soap_l_max : float, int
            SOAP parameters
        acsf_r_cut : float
            ACSF cutoff
        g2_params, g4_params : list of lists
            ACSF parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Atom_index, Symbol, Atomic_number, SOAP, ACSF
        """
        if species is None:
            species = list(set(atoms.get_chemical_symbols()))

        soap = SOAP(
            species=species,
            r_cut=soap_r_cut,
            n_max=soap_n_max,
            l_max=soap_l_max,
            periodic=True,
            sparse=False,
        )
        soap_vectors = soap.create(atoms)

        if g2_params is None:
            # Example default g2 parameters
            g2_params = [[1, 1], [1, 2], [1, 3]]
        if g4_params is None:
            # Example default g4 parameters
            g4_params = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

        acsf = ACSF(
            species=species,
            r_cut=acsf_r_cut,
            g2_params=g2_params,
            g4_params=g4_params,
            periodic=True,
            sparse=False,
        )
        acsf_vectors = acsf.create(atoms)

        df = pd.DataFrame({
            "Atom_index": np.arange(len(atoms)),
            "Symbol": atoms.get_chemical_symbols(),
            "Atomic_number": [atomic_numbers[s] for s in atoms.get_chemical_symbols()],
            "SOAP": list(soap_vectors),
            "ACSF": list(acsf_vectors)
        })

        return df
    
    @staticmethod
    def _get_local_feature_matrix(atoms, adsorbate_label, ads_dist_threshold, soap=True,
                                  soap_r_cut=6.0, soap_n_max=8, soap_l_max=6,
                                  acsf_r_cut=6.0, species=None, g2_params=None, g4_params=None):
        """
        Returns feature matrix (SOAP or ACSF) for atoms near adsorbate.
        """
        # Create descriptors
        descriptors = SimilarityAnalysis._create_descriptors(
            atoms, soap_r_cut=soap_r_cut, soap_n_max=soap_n_max, soap_l_max=soap_l_max,
            acsf_r_cut=acsf_r_cut, species=species, g2_params=g2_params, g4_params=g4_params,
        )

        # Graph transform
        gt = GraphTransform(atoms)
        ads_mask = gt._detect_adsorbates(adsorbate_label)
        ads_indices = np.where(ads_mask)[0]
        positions = gt.positions
        n_atoms = gt.n_atoms

        selected_indices = []
        if ads_dist_threshold is None:
            selected_indices = list(range(n_atoms))
        else:
            for i in range(n_atoms):
                min_dist = np.min([
                    get_distances(
                        positions[i],
                        positions[ads_idx],
                        cell=atoms.cell,
                        pbc=atoms.pbc,
                    )[1][0][0]
                    for ads_idx in ads_indices
                ])
                if min_dist < ads_dist_threshold:
                    selected_indices.append(i)

        # Extract features
        selected = descriptors.iloc[selected_indices]
        if soap:
            feature_matrix = np.vstack(selected["SOAP"].values)
        else:
            feature_matrix = np.vstack(selected["ACSF"].values)

        return feature_matrix
    
    @staticmethod
    def _compare_similarity(eval_atoms, ref_atoms, adsorbate_label, ads_dist_threshold=3.0, soap=True, rematch=True,
                            soap_r_cut=6.0, soap_n_max=8, soap_l_max=6,
                            acsf_r_cut=6.0, g2_params=None, g4_params=None):
        
        """
        Compare similarity between two structures near adsorbate.
        """
        eval_features = SimilarityAnalysis._get_local_feature_matrix(
            eval_atoms, adsorbate_label, ads_dist_threshold, soap,
            soap_r_cut, soap_n_max, soap_l_max, acsf_r_cut, g2_params, g4_params,
        )

        ref_features = SimilarityAnalysis._get_local_feature_matrix(
            ref_atoms, adsorbate_label, ads_dist_threshold, soap,
            soap_r_cut, soap_n_max, soap_l_max, acsf_r_cut, g2_params, g4_params,
        )

        gamma = 1 / ref_features.shape[1]
        if rematch:
            eval_features = normalize(eval_features)
            ref_features = normalize(ref_features)
            kernel = REMatchKernel(metric="rbf", gamma=gamma, alpha=0.5, threshold=1e-6)
            similarity = kernel.create([eval_features, ref_features])
        else:
            kernel = AverageKernel(metric="rbf", gamma=gamma)
            similarity = kernel.create([eval_features, ref_features])

        return similarity[0,1]