# ============================================================
# merger.py  –  Merge All Data Sources
# ============================================================

import pandas as pd
from typing import List, Optional


class DataMerger:
    """Merges multiple source DataFrames into one unified dataset with proxy broadcasting."""

    def __init__(self):
        self._sources: List[pd.DataFrame] = []
        self._merged: Optional[pd.DataFrame] = None

    def add_source(self, df: pd.DataFrame, name: str = "") -> 'DataMerger':
        if not df.empty:
            self._sources.append(df)
        return self

    def merge_on_keys(self,
                      keys: List[str] = None,
                      how: str = 'outer') -> pd.DataFrame:
        if not self._sources:
            raise RuntimeError("No sources added.")

        keys = keys or ['birth_region_standardized', 'year']
        
        # 1. Base merge
        merged = self._sources[0]
        for src in self._sources[1:]:
            # Clean common columns before merge to avoid _x _y bloat
            common_cols = [c for c in src.columns if c in merged.columns and c not in keys]
            if common_cols:
                src = src.drop(columns=common_cols)
            
            merged = pd.merge(merged, src, on=keys, how=how)

        # 2. BROADCASTING: 
        # If a specific region (e.g. 'Africa') is missing an outcome that IS present 
        # for 'Foreign Born (Total)', use the total as a proxy.
        if 'birth_region_standardized' in merged.columns:
            outcome_cols = [c for c in merged.columns if '_rate' in c or '_avg' in c or '_income' in c]
            
            for outcome in outcome_cols:
                # Get the 'Foreign Born' values per year
                fb_mask = merged['birth_region_standardized'] == 'Foreign Born (Total)'
                if any(fb_mask):
                    proxies = merged[fb_mask][['year', outcome]].dropna().set_index('year')[outcome]
                    
                    def apply_proxy(row):
                        if pd.isna(row[outcome]) and row['birth_region_standardized'] not in ['Sweden', 'Total']:
                            return proxies.get(row['year'], row[outcome])
                        return row[outcome]
                    
                    merged[outcome] = merged.apply(apply_proxy, axis=1)

        self._merged = merged
        return merged

    def save(self, filepath: str) -> str:
        if self._merged is None:
            raise RuntimeError("Nothing merged yet.")
        self._merged.to_csv(filepath, index=False)
        return filepath

    @property
    def merged_data(self) -> pd.DataFrame:
        return self._merged.copy() if self._merged is not None else None
