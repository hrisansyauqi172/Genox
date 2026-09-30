import numpy as np
import ot
import networkx as nx

class WWLAnalysis:
    @staticmethod
    def _wl_node_embedding(G_nx: nx.Graph, depth: int = 3, node_attr_keys: list = None, node_importance: bool = False, ads_dist_threshold: float = None):
        """
        Compute WL-style node embeddings for a networkx graph with node and edge features.

        Args:
            G_nx : networkx.Graph
                Graph with node attributes (numerical descriptors) and edge weights.
            depth : int
                Number of WL iterations.
            node_attr_keys : list of str
                Node attribute keys to use as initial features. If None, uses all numeric attributes.
            node_importance : True or False
                Only take into account nodes that within ads_dist_threshold
        Returns:
            h_concat : np.ndarray, shape (n_nodes, feature_dim)
        """
        if node_importance:
            nodes = [v for v, d in G_nx.nodes(data=True) if d.get("node_category", 0) == 1]
            if ads_dist_threshold is not None:
                for v, d in G_nx.nodes(data=True):
                    if d.get("node_category", 0) == 0:
                        dist_to_ads = d.get("ads_distance", np.inf)
                        if dist_to_ads <= 2 * ads_dist_threshold:
                            nodes.append(v)
        else:
            nodes = list(G_nx.nodes())
        
        n_nodes = len(nodes)
        node_to_idx = {v: i for i, v in enumerate(nodes)}

        # --- Build initial node feature matrix ---
        sample_attrs = G_nx.nodes[nodes[0]]
        exclude_keys = {"cell", "pbc", "position", "is_adsorbate", "base_weight", "node_category"}
        if node_attr_keys is None:
            # automatically use numeric attributes
            node_attr_keys = [k for k,v in sample_attrs.items() if isinstance(v, (int, float, np.float32, np.float64)) and k not in exclude_keys]
        else:
            selected_desc = node_attr_keys
            node_attr_keys = [k for k in selected_desc if k in sample_attrs and isinstance(sample_attrs[k], (int, float, np.float32, np.float64)) and k not in exclude_keys]

        X = np.zeros((n_nodes, len(node_attr_keys)), dtype=float)
        for i, v in enumerate(nodes):
            for j, key in enumerate(node_attr_keys):
                X[i, j] = G_nx.nodes[v].get(key, 0.0)

        # --- WL iterations ---
        h_prev = X.copy()
        h_accum = h_prev.copy()  # k = 0 contribution

        for _ in range(1, depth+1):
            h_new = np.zeros_like(h_prev)

            for i, v in enumerate(nodes):
                neighbors = list(G_nx.neighbors(v))
                if node_importance:
                    neighbors = [u for u in neighbors if G_nx.nodes[u].get("node_category", 0) == 1]
                deg = max(1, len(neighbors))
                agg = np.zeros_like(h_prev[i])
                for u in neighbors:
                    j = node_to_idx[u]
                    w = G_nx[v][u].get('weight', 1.0) * G_nx[v][u].get('length', 1.0)
                    agg += w * h_prev[j]
                h_new[i] = 0.5 * (h_prev[i] + agg / deg)  # factor 0.5
            h_accum += h_new
            h_prev = h_new

        return h_accum

    @staticmethod
    def _pairwise_wasserstein(graphs, depth=3, node_attr_keys=None, p=1, node_importance=False, ads_dist_threshold=None):
        """
        Fully directed pairwise Wasserstein distances between graphs using WL embeddings.

        Args:
            graphs : list of networkx.Graph
            depth : WL iterations
            node_attr_keys : list of str, which node attributes to use
            p : ground distance exponent (p=1 is standard)
        Returns:
            D : np.ndarray, (N,N) Wasserstein distance matrix
            metadata : dict with embeddings and node orders
        """
        embeddings = []
        node_orders = []

        for i, G in enumerate(graphs):
            h = WWLAnalysis._wl_node_embedding(G, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=ads_dist_threshold)
            embeddings.append(h)
            if node_importance:
                filtered_nodes = [v for v in G.nodes() if G.nodes[v].get("node_category", 0) == 1]
                node_orders.append(filtered_nodes)
                if ads_dist_threshold is not None:
                    for v in G.nodes():
                        if G.nodes[v].get("node_category", 0) == 0:
                            dist_to_ads = G.nodes[v].get("ads_distance", np.inf)
                            if dist_to_ads <= 2 * ads_dist_threshold:
                                node_orders[-1].append(v)
            else:
                node_orders.append(list(G.nodes()))

        N = len(graphs)
        D = np.zeros((N, N), dtype=float)

        for i in range(N):
            G_i = graphs[i]
            Xi = embeddings[i]
            # --- use base_weight of the node as mass ---
            ai = np.array([G_i.nodes[v].get("base_weight", 1.0) for v in node_orders[i]])
            ai = ai / ai.sum()
            for j in range(N):
                G_j = graphs[j]
                Xj = embeddings[j]
                bj = np.array([G_j.nodes[v].get("base_weight", 1.0) for v in node_orders[j]])
                bj = bj / bj.sum()
                # ground cost matrix: pairwise Euclidean^p
                M = ot.dist(Xi, Xj, metric='euclidean') ** p
                cost = ot.emd2(ai, bj, M)
                D[i, j] = cost ** (1.0 / p)
        
        metadata = {"embeddings": embeddings, "node_orders": node_orders}
        return D, metadata
    
    @staticmethod
    def _wasserstein_calc(graphs, ref_graphs, depth=3, node_attr_keys=None, p=1, node_importance=False, ads_dist_threshold=None):
        """
        Fully directed pairwise Wasserstein distances between graphs using WL embeddings.

        Args:
            graphs : list of networkx.Graph
            ref_graphs : list of networkx.Graph of the reference graphs
            depth : WL iterations
            node_attr_keys : list of str, which node attributes to use
            p : ground distance exponent (p=1 is standard)
        Returns:
            D : np.ndarray, (N,N) Wasserstein distance matrix
        """
        embeddings = []
        node_orders = []
        for i, G in enumerate(graphs):
            h = WWLAnalysis._wl_node_embedding(G, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=ads_dist_threshold)
            embeddings.append(h)
            if node_importance:
                filtered_nodes = [v for v in G.nodes() if G.nodes[v].get("node_category", 0) == 1]
                node_orders.append(filtered_nodes)
                if ads_dist_threshold is not None:
                    for v in G.nodes():
                        if G.nodes[v].get("node_category", 0) == 0:
                            dist_to_ads = G.nodes[v].get("ads_distance", np.inf)
                            if dist_to_ads <= 2 * ads_dist_threshold:
                                node_orders[-1].append(v)
            else:
                node_orders.append(list(G.nodes()))

        ref_embeddings = []
        ref_node_orders = []
        for i, G in enumerate(ref_graphs):
            ref_h_concat = WWLAnalysis._wl_node_embedding(G, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=ads_dist_threshold)
            ref_embeddings.append(ref_h_concat)
            if node_importance:
                filtered_nodes = [v for v in G.nodes() if G.nodes[v].get("node_category", 0) == 1]
                ref_node_orders.append(filtered_nodes)
                if ads_dist_threshold is not None:
                    for v in G.nodes():
                        if G.nodes[v].get("node_category", 0) == 0:
                            dist_to_ads = G.nodes[v].get("ads_distance", np.inf)
                            if dist_to_ads <= 2 * ads_dist_threshold:
                                ref_node_orders[-1].append(v)
            else:
                ref_node_orders.append(list(G.nodes()))

        N = len(graphs)
        N_ref = len(ref_graphs)
        D = np.zeros((N, N_ref), dtype=float)

        for i in range(N):
            G_i = graphs[i]
            Xi = embeddings[i]
            # --- use base_weight of the node as mass ---
            ai = np.array([G_i.nodes[v].get("base_weight", 1.0) for v in node_orders[i]])
            ai = ai / ai.sum()
            for j in range(N_ref):
                G_j = ref_graphs[j]
                Xj = ref_embeddings[j]
                bj = np.array([G_j.nodes[v].get("base_weight", 1.0) for v in ref_node_orders[j]])
                bj = bj / bj.sum()
                # ground cost matrix: pairwise Euclidean^p
                M = ot.dist(Xi, Xj, metric='euclidean') ** p
                cost = ot.emd2(ai, bj, M)
                D[i, j] = cost ** (1.0 / p)
        
        return D
    
    @staticmethod
    def _pairwise_ads_site_dist(graphs, depth=3, node_attr_keys=None, p=1, node_importance=False, ads_dist_threshold=None, dist_type='both'):
        """
        Fully directed pairwise Wasserstein distances between graphs using WL embeddings.

        Args:
            graphs : list of networkx.Graph
            depth : WL iterations
            node_attr_keys : list of str, which node attributes to use
            p : ground distance exponent (p=1 is standard)
        Returns:
            D : np.ndarray, (N,N) Wasserstein distance matrix
            metadata : dict with embeddings and node orders
        """
        def _ads_embedding(embeddings, node_order, ads_node):
            """
            Extract WL embedding of adsorbate nodes.
            """
            if len(ads_node) == 0:
                raise ValueError("No adsorbate nodes found for Euclidean comparison.")
            
            ads_indices = node_order.index(ads_node[0])

            return embeddings[ads_indices]

        embeddings = []
        node_orders = []
        ads_nodes = []

        for i, G in enumerate(graphs):
            h = WWLAnalysis._wl_node_embedding(G, depth=depth, node_attr_keys=node_attr_keys, node_importance=node_importance, ads_dist_threshold=ads_dist_threshold)
            embeddings.append(h)
            if node_importance:
                filtered_nodes = [v for v in G.nodes() if G.nodes[v].get("node_category", 0) == 1]
                node_orders.append(filtered_nodes)
                if ads_dist_threshold is not None:
                    for v in G.nodes():
                        if G.nodes[v].get("node_category", 0) == 0:
                            dist_to_ads = G.nodes[v].get("ads_distance", np.inf)
                            if dist_to_ads <= 2 * ads_dist_threshold:
                                node_orders[-1].append(v)
            else:
                node_orders.append(list(G.nodes()))
            find_ads_node = [v for v in G.nodes() if G.nodes[v].get("is_adsorbate", False) is True]
            ads_nodes.append(find_ads_node)

        N = len(graphs)
        D = np.zeros((N, N), dtype=float)

        for i in range(N):
            G_i = graphs[i]
            Xi = embeddings[i]
            # --- use base_weight of the node as mass ---
            ai = np.array([G_i.nodes[v].get("base_weight", 1.0) for v in node_orders[i]])
            ai = ai / ai.sum()
            # adsorbate embedding for Euclidean
            ads_i = _ads_embedding(Xi, node_orders[i], ads_nodes[i])
            
            for j in range(N):
                G_j = graphs[j]
                Xj = embeddings[j]
                
                # ads node embedding euclidean dist. calc
                ads_j = _ads_embedding(Xj, node_orders[j], ads_nodes[j])
                eucl_dist = np.linalg.norm(ads_i - ads_j)

                # wasserstein distance calc. - ground cost matrix: pairwise Euclidean^p
                bj = np.array([G_j.nodes[v].get("base_weight", 1.0) for v in node_orders[j]])
                bj = bj / bj.sum()
                M = ot.dist(Xi, Xj, metric='euclidean') ** p
                cost = ot.emd2(ai, bj, M)
                wass_dist = cost ** (1.0 / p)

                if dist_type == 'both':
                    D[i, j] = 0.75*eucl_dist + 0.25*wass_dist
                elif dist_type == 'euclidean':
                    D[i, j] = eucl_dist
                else:
                    D[i, j] = wass_dist
        
        metadata = {"embeddings": embeddings, "node_orders": node_orders}
        return D, metadata