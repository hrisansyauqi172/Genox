"""
feature_provenance.py
=====================
Provides the FeatureProvenance class, intended to be imported as part of
the Genox library (alongside GraphTransform, WWLAnalysis, etc.).

Design
------
build()            — operates on ONE graph + ONE structure_id.
                     Call in a for-loop to accumulate a full dataset registry.

from_pkl()         — post-training convenience: opens a saved GPR/NN pickle,
                     iterates the stored graphs with a for-loop, and returns
                     the complete registry in one call.

to_flat_dataframe()        — flatten nested registry → one row per atom.
summary_by_element_role()  — mean ± std per (element, role) group.
describe_structure()       — pretty-print tree for one structure.

Registry schema
---------------
{
    "<structure_id>": {
        "atom_<node_index>": {
            "symbol":        "Zr",
            "role":          "surface",    # "surface" | "adsorbate"
            "ads_distance":  2.134,        # Å to nearest adsorbate atom
            "node_category": 1,            # 0 = outer, 1 = important
            "features": {
                "d_band_center": -1.432,
                "DDEC6_charge":   0.881,
                ...
            }
        },
        ...
    },
    "<structure_id>": { ... },
    ...
}

Typical usage (post-training, from saved pkl)
---------------------------------------------
    from feature_provenance import FeatureProvenance

    registry = FeatureProvenance.from_pkl(
        model_path="saved_models/GPR_Laplacian_eads_tt.pkl",
        feature_names=None,   # None → auto-detect from graph
    )

    # Or build one graph at a time in your own loop:
    registry = {}
    for item, G in zip(db_list, graph_list):
        entry = FeatureProvenance.build(G, structure_id=item["id"])
        registry.update(entry)

    flat_df = FeatureProvenance.to_flat_dataframe(registry)
"""

from __future__ import annotations

import pickle
import warnings
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Node attribute keys that are metadata, not descriptors
_META_KEYS = frozenset({
    "symbol", "is_adsorbate", "node_category",
    "ads_distance", "position", "base_weight",
})


def _auto_feature_keys(G: nx.Graph) -> List[str]:
    """
    Peek at the first node of G and return all numeric non-meta attribute keys.
    All nodes in a Genox graph share the same schema, so one peek is enough.
    """
    for _node, ndata in G.nodes(data=True):
        return [
            k for k, v in ndata.items()
            if k not in _META_KEYS
            and isinstance(v, (int, float, np.floating))
            and not np.isnan(float(v))
        ]
    return []


class FeatureProvenance:
    """
    Atom-specific feature provenance registry extracted from NetworkX graphs.

    All methods are static — no instantiation needed.
    """

    # ------------------------------------------------------------------
    # Core builder  — ONE graph, ONE structure
    # ------------------------------------------------------------------

    @staticmethod
    def build(graph: nx.Graph, structure_id: str, feature_names: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Build the atom-specific registry entry for a single structure.

        Parameters
        ----------
        graph : networkx.Graph
            A graph produced by GraphTransform._make_graph().
            Nodes must carry: symbol, is_adsorbate, ads_distance,
            node_category, and all descriptor columns.
        structure_id : str
            Unique identifier for this structure (e.g. entry["id"]).
        feature_names : list[str] | None
            Which node attributes to record as features.
            None → auto-detect all numeric non-meta attributes.

        Returns
        -------
        registry : dict
            { "<structure_id>": { "atom_0": {...}, "atom_1": {...}, ... } }

            Merge many of these with registry.update(entry) to build the
            full dataset registry.
        """
        if feature_names is None:
            feature_names = _auto_feature_keys(graph)

        struct_entry: Dict[str, dict] = {}

        for node_idx, ndata in graph.nodes(data=True):
            # --- metadata ---
            symbol   = ndata.get("symbol", "?")
            is_ads   = bool(ndata.get("is_adsorbate", False))
            role     = "adsorbate" if is_ads else "surface"
            ads_dist = ndata.get("ads_distance", np.nan)
            node_cat = ndata.get("node_category", -1)

            try:
                ads_dist_f = float(ads_dist)
                ads_dist_out = None if np.isnan(ads_dist_f) else ads_dist_f
            except (TypeError, ValueError):
                ads_dist_out = None

            # --- descriptors ---
            features: Dict[str, float] = {}
            for feat in feature_names:
                val = ndata.get(feat, None)
                if val is None:
                    continue
                try:
                    features[feat] = float(val)
                except (TypeError, ValueError):
                    features[feat] = np.nan

            struct_entry[f"atom_{node_idx}"] = {
                "symbol":        symbol,
                "role":          role,
                "ads_distance":  ads_dist_out,
                "node_category": int(node_cat),
                "features":      features,
            }

        return {structure_id: struct_entry}

    # ------------------------------------------------------------------
    # Post-training convenience — read graphs straight from saved pkl
    # ------------------------------------------------------------------

    @staticmethod
    def from_pkl(model_path: str, structure_ids: Optional[List[str]] = None, feature_names: Optional[List[str]] = None,
                 graph_key: str = "all_graph", verbose: bool = True) -> Dict[str, dict]:
        """
        Build the full dataset registry from a saved GPR/NN model pickle.

        The pickle already stores the graphs (under key ``all_graph`` by
        default) so no graph reconstruction or atoms files are needed.

        Parameters
        ----------
        model_path : str
            Path to the saved .pkl file produced by
            PredictionModel._gpr_ads_energy().
        structure_ids : list[str] | None
            Labels to assign to each graph in order.  If None, uses
            "structure_0", "structure_1", … as fallback labels.
        feature_names : list[str] | None
            Features to record.  None → auto-detect from first graph.
        graph_key : str
            Key inside the pickle dict that holds the list of graphs.
            Default is "all_graph" (the key used by _gpr_ads_energy).
        verbose : bool

        Returns
        -------
        registry : dict   (complete dataset registry)
        """
        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        graphs: List[nx.Graph] = saved.get(graph_key, [])
        if not graphs:
            raise KeyError(
                f"No graphs found under key '{graph_key}' in {model_path}. "
                f"Available keys: {list(saved.keys())}"
            )

        # Feature names: prefer those stored in the pkl (training used these)
        if feature_names is None:
            feature_names = saved.get("node_attr_keys") or _auto_feature_keys(graphs[0])

        if structure_ids is None:
            structure_ids = [f"structure_{i}" for i in range(len(graphs))]

        if len(structure_ids) != len(graphs):
            raise ValueError(
                f"structure_ids has {len(structure_ids)} entries but "
                f"the pkl contains {len(graphs)} graphs."
            )

        if verbose:
            print(f"[FeatureProvenance] Building registry from {model_path}")
            print(f"  Graphs  : {len(graphs)}")
            print(f"  Features: {feature_names}")

        registry: Dict[str, dict] = {}
        for i, (struct_id, G) in enumerate(zip(structure_ids, graphs)):
            entry = FeatureProvenance.build(G, struct_id, feature_names=feature_names)
            registry.update(entry)
            if verbose and (i + 1) % max(1, len(graphs) // 10) == 0:
                print(f"  [{i+1}/{len(graphs)}] {struct_id}")

        if verbose:
            print(f"[FeatureProvenance] Done. {len(registry)} structures in registry.")

        return registry

    # ------------------------------------------------------------------
    # Flatten to a tidy DataFrame
    # ------------------------------------------------------------------

    @staticmethod
    def to_flat_dataframe(registry: Dict[str, dict]) -> pd.DataFrame:
        """
        Flatten the nested registry to one row per atom.

        Columns
        -------
        structure_id, atom_key, symbol, role, ads_distance,
        node_category, <feature_1>, <feature_2>, …

        This is the format for downstream analysis, ablation joins,
        or symbolic-regression input.
        """
        rows = []
        for struct_id, atoms in registry.items():
            for atom_key, info in atoms.items():
                row = {
                    "structure_id":  struct_id,
                    "atom_key":      atom_key,
                    "symbol":        info["symbol"],
                    "role":          info["role"],
                    "ads_distance":  info["ads_distance"],
                    "node_category": info["node_category"],
                }
                row.update(info["features"])
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        meta_cols = ["structure_id", "atom_key", "symbol", "role",
                     "ads_distance", "node_category"]
        feat_cols = sorted(c for c in df.columns if c not in meta_cols)
        return df[meta_cols + feat_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Group-by summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary_by_element_role(registry: Dict[str, dict],
                                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Mean ± std of each feature, grouped by (element symbol, role).

        Useful sanity check: "what is the typical d_band_center of
        surface Zr atoms across the whole dataset?"

        Returns
        -------
        pd.DataFrame  multi-index (symbol, role), multi-column (feature, stat)
        """
        flat = FeatureProvenance.to_flat_dataframe(registry)
        if flat.empty:
            return pd.DataFrame()

        meta_cols = {"structure_id", "atom_key", "symbol", "role",
                     "ads_distance", "node_category"}
        all_feat_cols = [c for c in flat.columns if c not in meta_cols]
        feat_cols = [f for f in (feature_names or all_feat_cols) if f in flat.columns]

        return (
            flat.groupby(["symbol", "role"])[feat_cols]
            .agg(["mean", "std"])
        )

    # ------------------------------------------------------------------
    # Pretty-print one structure
    # ------------------------------------------------------------------

    @staticmethod
    def describe_structure(registry: Dict[str, dict], structure_id: str, max_atoms: int = 20) -> None:
        """
        Print a human-readable tree for one structure.

            t-ZrO2_101_Zr_CO_site_1
            └─ atom_0   [Zr | surface  | dist=2.134 Å | cat=1]
               ├─ d_band_center        : -1.4320
               ├─ DDEC6_charge         :  0.8810
               └─ ...
            └─ atom_42  [C  | adsorbate | dist=0.000 Å | cat=1]
               └─ ...
        """
        entry = registry.get(structure_id)
        if entry is None:
            print(f"Structure '{structure_id}' not found in registry.")
            return

        print(f"\n{structure_id}")
        atoms = list(entry.items())
        shown = atoms[:max_atoms]
        if len(atoms) > max_atoms:
            print(f"  (showing first {max_atoms} of {len(atoms)} atoms)")

        for atom_key, info in shown:
            dist_str = (f"{info['ads_distance']:.3f}"
                        if info["ads_distance"] is not None else "n/a")
            print(f"  └─ {atom_key:<10}  "
                  f"[{info['symbol']:<3}| {info['role']:<10}| "
                  f"dist={dist_str} Å | cat={info['node_category']}]")
            feats = list(info.get("features", {}).items())
            for k, v in feats[:-1]:
                print(f"     ├─ {k:<22} : {v:.4f}")
            if feats:
                k, v = feats[-1]
                print(f"     └─ {k:<22} : {v:.4f}")
