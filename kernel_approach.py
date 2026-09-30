import hdbscan
import numpy as np
from sklearn.decomposition import KernelPCA

class KernelApproach:
    @staticmethod
    def _kernel_transform(D, hyperparameters, type_kernel="RBF"):
        """
        Convert a distance matrix D into a kernel (similarity) matrix.
        hyperparameters: list/tuple of kernel parameters (e.g. [sigma, alpha])
        """
        sigma = hyperparameters[0] if hyperparameters[0] is not None else np.median(D[D > 0])
        amplitude = hyperparameters[1] if len(hyperparameters) > 1 else 1.0
        
        if type_kernel == "RBF":
            K = np.exp(-D**2 / (2 * sigma**2))
        elif type_kernel == "Laplacian":
            K = np.exp(-D / sigma)
        elif type_kernel == "Cauchy":
            p = hyperparameters[2] if len(hyperparameters) > 2 else 1.0
            K = 1.0 / (1.0 + (D**2 / sigma**2))**p
        elif type_kernel == "RationalQuadratic":
            alpha = hyperparameters[2] if len(hyperparameters) > 2 else 1.0
            K = (1.0 + (D**2 / (2 * alpha * sigma**2))) ** (-alpha)
        elif type_kernel == "IMQ":  # inverse multiquadric
            c = sigma
            b = hyperparameters[2] if len(hyperparameters) > 2 else 1.0
            K = 1.0 / np.sqrt(D**2 + c**2)**b
        else:
            raise ValueError(f"Unknown kernel type: {type_kernel}")

        # scale, enforce symmetry, and PSD projection (if necessary)
        K *= amplitude
        # Adjustment for square matrices to ensure symmetry and PSD
        if K.shape[0] == K.shape[1]:
            K = 0.5 * (K + K.T)
            eigvals, eigvecs = np.linalg.eigh(K)
            if np.min(eigvals) < 0:
                eigvals[eigvals < 1e-8] = 1e-8
                K = eigvecs @ np.diag(eigvals) @ eigvecs.T
                K = 0.5 * (K + K.T)
        return K
    
    @staticmethod
    def _combined_kernel(D, hyperparameters_1, hyperparameters_2, type_kernel_1="RBF", type_kernel_2="RBF", combine="sum", wk1=None, wk2=None):
        """
        Combine two kernels into a single kernel matrix.

        Parameters
        ----------
        D : np.ndarray
            Distance matrix (N x N)
        hyperparameters_1 : list
            Hyperparameters for kernel 1 (sigma, amplitude, etc.)
        hyperparameters_2 : list
            Hyperparameters for kernel 2
        type_kernel_1 : str
            Type of kernel 1 ("RBF", "Laplacian", "Linear", etc.)
        type_kernel_2 : str
            Type of kernel 2
        combine : str
            "sum" or "product"
        wk1, wk2 : float or None
            Optional positive weights for each kernel
        eps : float
            Small value for PSD projection
        """
        # Create the base kernels
        K1 = KernelApproach._kernel_transform(D, hyperparameters_1, type_kernel=type_kernel_1)
        K2 = KernelApproach._kernel_transform(D, hyperparameters_2, type_kernel=type_kernel_2)

        # Apply optional weights
        wk1 = 1.0 if wk1 is None else wk1
        wk2 = 1.0 if wk2 is None else wk2
        K1 *= wk1
        K2 *= wk2

        # Combine kernels
        if combine == "sum":
            K = K1 + K2
        elif combine == "product":
            K = K1 * K2
        else:
            raise ValueError("combine must be 'sum' or 'product'")

        # Ensure symmetry
        K = 0.5 * (K + K.T)

        # PSD projection only if needed
        eigvals, eigvecs = np.linalg.eigh(K)
        if np.min(eigvals) < 0:
            eigvals[eigvals < 1e-8] = 1e-8
            K = eigvecs @ np.diag(eigvals) @ eigvecs.T
            K = 0.5 * (K + K.T)

        return K

    @staticmethod
    def _kernel_pca(D, hyperparameters, type_kernel="RBF"):
        # --- Convert distances to RBF kernel ---
        N = D.shape[0]
        D_kernel = KernelApproach._kernel_transform(D, hyperparameters, type_kernel=type_kernel)

        # clip tiny eigenvalues negatives to 0 due calculations effect from the script
        eigvals, eigvecs = np.linalg.eigh(D_kernel)
        eigvals[eigvals < 0] = 0.0
        D_kernel = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Kernel PCA embedding ---
        kpca = KernelPCA(kernel="precomputed", n_components=N)
        X_kpca = kpca.fit_transform(D_kernel)

        return X_kpca
    
class SiteCluster:
    @staticmethod
    def _filter_sites(D, cluster_dict, num_representatives=5):
        """
        Select representative adsorption sites per cluster using medoids
        and farthest-point sampling in distance space.

        Args:
            D : np.ndarray (N, N)
                Pairwise distance matrix
            cluster_dict : dict
                {cluster_label: [indices]}
            num_representatives : int
                Number of representatives per cluster

        Returns:
            filtered_sites : list
                Representative site indices
        """

        filtered_sites = []

        for label, cluster_indices in cluster_dict.items():
            # Keep noise
            if label == -1:
                filtered_sites.extend(cluster_indices)
                continue

            # Small cluster → keep all
            if len(cluster_indices) <= num_representatives:
                filtered_sites.extend(cluster_indices)
                continue

            # --- Medoid ---
            subD = D[np.ix_(cluster_indices, cluster_indices)]
            medoid_local_idx = np.argmin(subD.sum(axis=1))
            selected = [cluster_indices[medoid_local_idx]]

            # --- Farthest-point sampling ---
            while len(selected) < num_representatives:
                remaining = [i for i in cluster_indices if i not in selected]

                dists = [
                    min(D[i, j] for j in selected)
                    for i in remaining
                ]

                next_idx = remaining[np.argmax(dists)]
                selected.append(next_idx)

            filtered_sites.extend(selected)

        return filtered_sites     

    @staticmethod
    def _cluster_sites_hdbscan(D, min_cluster_size=2, min_samples=None, num_representatives=5):
        """
        Hierarchical DBSCAN directly on a precomputed distance matrix.

        Args:
            D : np.ndarray (N, N)
                Pairwise distance matrix (Euclidean / Wasserstein / combined)
            min_cluster_size : int
                Smallest cluster size to consider
            min_samples : int or None
                Density robustness parameter (None = min_cluster_size)

        Returns:
            labels : np.ndarray
                Cluster labels (-1 = noise)
            cluster_dict : dict
                {cluster_label: [indices]}
        """

        clusterer = hdbscan.HDBSCAN(
            metric='precomputed',
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(D)

        # --- Build cluster dictionary ---
        cluster_dict = {}
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)[0].tolist()
            cluster_dict[label] = cluster_indices
        
        # --- Sites filtering ---
        filtered_sites = SiteCluster._filter_sites(D, cluster_dict, num_representatives)

        return cluster_dict, filtered_sites