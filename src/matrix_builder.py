"""
ShopSense User-Item Sparse Matrix Builder
=========================================
Builds, indexes, and normalizes sparse user-item interaction matrices
from interaction DataFrames.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp


class UserItemMatrix:
    """
    Encapsulates sparse matrix representations and bidirectional index mappings
    for users and items.
    """

    def __init__(self):
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: List[str] = []
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: List[str] = []

        self.R: Optional[sp.csr_matrix] = None          # Rating matrix (CSR)
        self.R_csc: Optional[sp.csc_matrix] = None      # Rating matrix (CSC) for column operations
        self.R_norm: Optional[sp.csr_matrix] = None     # Mean-centered, L2-normalized matrix for fast cosine sim

        self.user_means: Optional[np.ndarray] = None
        self.item_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

        self.n_users: int = 0
        self.n_items: int = 0
        self.n_interactions: int = 0

    def fit(self, df: pd.DataFrame) -> "UserItemMatrix":
        """
        Fits ID mappings and builds sparse matrices from interaction DataFrame.
        """
        # 1. Establish deterministic ID mappings
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["parent_asin"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = list(unique_users)
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = list(unique_items)

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        # 2. Aggregate any repeated interactions by taking the mean rating per (user, item)
        agg_df = df.groupby(["user_id", "parent_asin"])["rating"].mean().reset_index()
        self.n_interactions = len(agg_df)

        u_idx = agg_df["user_id"].map(self.user_to_idx).values
        i_idx = agg_df["parent_asin"].map(self.item_to_idx).values
        ratings = agg_df["rating"].values.astype(np.float32)

        # 3. Construct CSR and CSC sparse matrices
        self.R = sp.csr_matrix((ratings, (u_idx, i_idx)), shape=(self.n_users, self.n_items))
        self.R_csc = self.R.tocsc()

        # 4. Compute statistics
        self.global_mean = float(ratings.mean())

        # Per-user means
        u_means = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            row = self.R.data[self.R.indptr[u]:self.R.indptr[u + 1]]
            u_means[u] = row.mean() if len(row) > 0 else self.global_mean
        self.user_means = u_means

        # Per-item means
        i_means = np.zeros(self.n_items, dtype=np.float32)
        for i in range(self.n_items):
            col = self.R_csc.data[self.R_csc.indptr[i]:self.R_csc.indptr[i + 1]]
            i_means[i] = col.mean() if len(col) > 0 else self.global_mean
        self.item_means = i_means

        # 5. Build mean-centered and row-L2-normalized matrix for fast cosine similarity
        centered_data = self.R.data - np.repeat(self.user_means, np.diff(self.R.indptr))
        R_centered = sp.csr_matrix((centered_data, self.R.indices, self.R.indptr), shape=self.R.shape)

        # Row L2 normalization
        norms = sp.linalg.norm(R_centered, axis=1)
        norms[norms == 0] = 1e-10
        self.R_norm = sp.diags(1.0 / norms) @ R_centered

        return self

    def get_user_vector(self, user_idx: int) -> sp.csr_matrix:
        """Returns sparse row for a user index."""
        return self.R[user_idx]

    def get_item_column(self, item_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (user_indices, ratings) for an item index from CSC matrix."""
        start = self.R_csc.indptr[item_idx]
        end = self.R_csc.indptr[item_idx + 1]
        return self.R_csc.indices[start:end], self.R_csc.data[start:end]
