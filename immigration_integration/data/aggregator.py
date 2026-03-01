# ============================================================
# aggregator.py  –  Aggregate Data by Country / Region
# ============================================================

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class RegionAggregator:
    """Aggregate integration outcomes by birth region."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def aggregate_by_region(self,
                            outcome_cols: List[str],
                            agg_funcs: Optional[Dict[str, str]] = None
                            ) -> pd.DataFrame:
        """Compute summary statistics for each birth region."""
        if agg_funcs is None:
            agg_funcs = {col: 'mean' for col in outcome_cols}

        result = (
            self._df
            .groupby('birth_region_standardized')
            .agg(agg_funcs)
            .reset_index()
        )
        return result

    def aggregate_by_region_and_year(self,
                                     outcome_cols: List[str]
                                     ) -> pd.DataFrame:
        """Compute yearly averages for each birth region."""
        result = (
            self._df
            .groupby(['birth_region_standardized', 'year'])
            [outcome_cols]
            .mean()
            .reset_index()
        )
        return result

    def compute_integration_score(self,
                                   outcome_cols: List[str],
                                   weights: Optional[Dict[str, float]] = None
                                   ) -> pd.DataFrame:
        """Produce a composite integration score per region."""
        agg = self.aggregate_by_region(outcome_cols)

        if weights is None:
            weights = {col: 1.0 / len(outcome_cols) for col in outcome_cols}

        for col in outcome_cols:
            col_min = agg[col].min()
            col_max = agg[col].max()
            if col_max - col_min > 0:
                agg[f'{col}_norm'] = (agg[col] - col_min) / (col_max - col_min)
            else:
                agg[f'{col}_norm'] = 0.0

        norm_cols = [f'{col}_norm' for col in outcome_cols]
        weight_vals = [weights[col] for col in outcome_cols]
        agg['integration_score'] = sum(
            agg[nc] * w for nc, w in zip(norm_cols, weight_vals)
        )

        return agg

    def __repr__(self) -> str:
        return f"RegionAggregator(rows={len(self._df)})"
