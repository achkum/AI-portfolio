# ============================================================
# scb_loader.py  –  SCB (Statistics Sweden) Data Loader
# ============================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from data.base_loader import BaseDataLoader


class SCBLoader(BaseDataLoader):
    """SCB-specific data loader with improved observation targeting."""

    def __init__(self, filepath: str, config: Dict[str, Any],
                 indicator: str = 'employment'):
        super().__init__(filepath, config)
        self._indicator = indicator
        self._birth_region_mapping = config.get('birth_region_mapping', {})

    def _read_file(self) -> pd.DataFrame:
        chunksize = 100000
        encoding = self._config.get('scb_encoding', 'ISO-8859-1')
        sep = self._config.get('scb_separator', ',')
        
        mapping_lower = {k.lower(): v for k, v in self._birth_region_mapping.items()}
        target_name = {
            'employment': 'employment_rate',
            'income': 'median_income',
            'self_sufficiency': 'self_sufficiency_rate'
        }.get(self._indicator, 'value')

        all_chunks = []
        
        try:
            reader = pd.read_csv(
                self._filepath,
                encoding=encoding,
                sep=sep,
                chunksize=chunksize,
                low_memory=False
            )
            
            for chunk in reader:
                chunk.columns = [c.strip().replace(' ', '_').lower().strip('"') for c in chunk.columns]
                
                # National Filter
                if 'region' in chunk.columns:
                    chunk = chunk[chunk['region'].str.contains('Sweden|Riket', case=False, na=False)]
                
                # Totals (Sex/Age)
                if 'sex' in chunk.columns:
                    chunk = chunk[chunk['sex'].str.contains('total', case=False, na=False)]
                if 'age' in chunk.columns:
                    chunk = chunk[chunk['age'].str.contains('total|15-74|20-64', case=False, na=False)]
                if 'number_of_years_in_sweden' in chunk.columns:
                    chunk = chunk[chunk['number_of_years_in_sweden'].str.contains('total', case=False, na=False)]

                if chunk.empty: continue

                # Value column
                val_col = chunk.columns[-1]
                chunk[val_col] = chunk[val_col].astype(str).str.replace(' ', '').replace('..', np.nan)
                chunk[val_col] = pd.to_numeric(chunk[val_col], errors='coerce')

                # Observation Filter
                if 'observations' in chunk.columns:
                    obs_col = 'observations'
                    if self._indicator == 'employment':
                        mask = chunk[obs_col].str.contains('employment rate', case=False, na=False)
                        if not any(mask): mask = chunk[obs_col].str.contains('number of employed', case=False, na=False)
                        chunk = chunk[mask].copy()
                    elif self._indicator == 'income':
                        mask = chunk[obs_col].str.contains('Median value', case=False, na=False)
                        chunk = chunk[mask].copy()
                    elif self._indicator == 'self_sufficiency':
                        mask = chunk[obs_col].str.contains('rate|Självförsörjandegrad', case=False, na=False)
                        if not any(mask): mask = chunk[obs_col].str.contains('Number of self-sufficient', case=False, na=False)
                        chunk = chunk[mask].copy()

                if chunk.empty: continue

                chunk = chunk.rename(columns={val_col: target_name})
                
                # Standardise birth regions
                chunk['birth_region_standardized'] = (
                    chunk['region_of_birth']
                    .str.lower()
                    .map(mapping_lower)
                    .fillna(chunk['region_of_birth'])
                )
                
                # Filter out junk
                chunk = chunk[~chunk['region_of_birth'].str.lower().isin(['total', 'all regions', 'all regions of birth'])]
                chunk = chunk.dropna(subset=[target_name, 'year'])
                
                all_chunks.append(chunk[['year', 'birth_region_standardized', target_name]])
                
            if not all_chunks:
                return pd.DataFrame()
                
            df = pd.concat(all_chunks, ignore_index=True)
            # Aggregate to remove any duplicates from across chunks
            df = df.groupby(['year', 'birth_region_standardized'], as_index=False).mean()
            return df
            
        except Exception as e:
            print(f"Error loading {self._filepath}: {e}")
            return pd.DataFrame()

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        return not df.empty

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df['year'] = df['year'].astype(int)
        return df

    def filter_years(self, start: int, end: int) -> 'SCBLoader':
        if self._is_loaded and not self._data.empty:
            self._data = self._data[
                (self._data['year'] >= start) &
                (self._data['year'] <= end)
            ]
        return self
