# ============================================================
# socialstyrelsen_loader.py  –  Socialstyrelsen Data Loader
# ============================================================

import pandas as pd
import numpy as np
from typing import Dict, Any
from data.base_loader import BaseDataLoader


class SocialstyrelsenLoader(BaseDataLoader):
    """Socialstyrelsen welfare data loader (CSV/SDB format) - Aggregated for transparency."""

    def _read_file(self) -> pd.DataFrame:
        if self._filepath.endswith('.csv'):
            df = pd.read_csv(
                self._filepath, 
                sep=self._config.get('soc_separator', ';'),
                encoding='ISO-8859-1',
                skiprows=0
            )
            if not any(isinstance(c, int) or str(c).isdigit() for c in df.columns):
                df = pd.read_csv(
                    self._filepath, 
                    sep=self._config.get('soc_separator', ';'),
                    encoding='ISO-8859-1',
                    skiprows=1
                )
        else:
            df = pd.read_excel(self._filepath)
        
        return df

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        year_cols = [c for c in df.columns if str(c).isdigit()]
        return len(year_cols) > 0 or 'year' in df.columns

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override: Melt and Aggregate to ensure one row per Year/broad-region."""
        
        # Clean column names
        df.columns = [c.strip().replace(' ', '_').lower().replace('-', '_') for c in df.columns]
        
        # Identify Year columns (wide to long)
        year_cols = [c for c in df.columns if str(c).isdigit()]
        id_cols = [c for c in df.columns if c not in year_cols]
        
        if year_cols:
            df = df.melt(
                id_vars=id_cols, 
                value_vars=year_cols, 
                var_name='year', 
                value_name='welfare_amount_total'
            )
        
        # Standardise birth region column
        # Look for the column that contains 'född' or 'inrikes'
        region_col = next((c for c in df.columns if 'föd' in c or 'born' in c or 'inrikes' in c), None)
        if region_col:
            def map_soc_region(val):
                val = str(val).lower()
                if 'inrikes' in val or 'sweden' in val: return 'Sweden'
                if 'utrikes' in val or 'foreign' in val: return 'Foreign Born (Total)'
                return 'Other'
            df['birth_region_standardized'] = df[region_col].apply(map_soc_region)
        else:
            df['birth_region_standardized'] = 'Foreign Born (Total)' # General fallback
            
        # Clean numeric data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['welfare_amount'] = pd.to_numeric(
            df['welfare_amount_total'].astype(str).str.replace(' ', '').replace('--', 'NaN').replace('-', 'NaN'), 
            errors='coerce'
        )
        
        # AGGREGATE: Take the average of ALL rows for a given (Year, Standardized Region)
        # This averages across all municipalities AND household types at the national level
        if 'birth_region_standardized' in df.columns:
            df = df.groupby(['year', 'birth_region_standardized'], as_index=False)['welfare_amount'].mean()
            # Rename resulting column correctly for the merge
            df = df.rename(columns={'welfare_amount': 'welfare_amount_avg'})
            
        return df
