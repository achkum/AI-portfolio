# ============================================================
# migrationsverket_loader.py  –  Migrationsverket Data Loader
# ============================================================

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from data.base_loader import BaseDataLoader


class MigrationsverketLoader(BaseDataLoader):
    """Migrationsverket-specific data loader supporting filtered file discovery."""

    def __init__(self, filepath: str, config: Dict[str, Any],
                 data_type: str = 'asylum'):
        super().__init__(filepath, config)
        self._data_type = data_type

    def _read_file(self) -> pd.DataFrame:
        """Override: Load files matching the data_type pattern to distinguish Asylum from Work/Residence."""
        
        target_path = self._filepath
        if os.path.isdir(target_path):
            files = glob.glob(os.path.join(target_path, "*.xlsx")) + \
                    glob.glob(os.path.join(target_path, "*.xls"))
            
            # Pattern matching for data type
            patterns = {
                'asylum': ['asyl', 'avgjorda'],
                'permits': ['uppehåll', 'arbet', 'beviljade']
            }
            
            target_patterns = patterns.get(self._data_type, [])
            filtered_files = []
            for f in files:
                f_lower = os.path.basename(f).lower()
                if any(p in f_lower for p in target_patterns):
                    # For permits, if it's asylum, skip it
                    if self._data_type == 'permits' and 'asyl' in f_lower:
                        continue
                    filtered_files.append(f)
            
            if not filtered_files:
                return pd.DataFrame()
            
            all_dfs = []
            for f in filtered_files:
                year_from_filename = None
                for word in f.replace('_', ' ').replace('.', ' ').replace('-', ' ').split():
                    if word.isdigit() and len(word) == 4:
                        year_from_filename = int(word)
                        break
                
                try:
                    xl = pd.ExcelFile(f)
                    target_sheet = None
                    # Sheet Selection: prefer 'Medborgarskap'
                    for s in xl.sheet_names:
                        s_l = s.lower()
                        if 'medborgar' in s_l or 'nationality' in s_l:
                             if 'första' in s_l or 'frsta' in s_l:
                                 target_sheet = s
                                 break
                             target_sheet = s
                    
                    if not target_sheet:
                        target_sheet = xl.sheet_names[0]
                        
                    df_temp = pd.read_excel(xl, sheet_name=target_sheet, header=None)
                    
                    # Header detection
                    header_idx = 0
                    for i, row in df_temp.iterrows():
                        row_vals = [str(x).lower().strip() for x in row.values if pd.notna(x)]
                        if any('medborgar' in x or 'nation' in x for x in row_vals) and len(row_vals) > 1:
                            if not all('medborgar' in x or 'nation' in x for x in row_vals):
                                header_idx = i
                                break
                    
                    df = pd.read_excel(xl, sheet_name=target_sheet, skiprows=header_idx)
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    
                    if year_from_filename and 'year' not in df.columns:
                        df['year'] = year_from_filename
                    
                    all_dfs.append(df)
                    
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
            
            result = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
            return result
        else:
            return pd.read_excel(self._filepath)

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        if df.empty: return False
        cols = [c.lower() for c in df.columns]
        return any('nation' in c or 'medborgar' in c for c in cols)

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override: standardise columns and map nationality to regions."""
        
        # 1. Identify key columns
        nat_col = next((c for c in df.columns if ('medborgar' in c or 'nation' in c) and len(str(c)) < 60), None)
        count_col = next((c for c in df.columns if 'total' in c or 'beviljade' in c or 'avgjorda' in c or 'summa' in c), None)
        if not count_col:
            count_col = df.columns[-1]

        year_col = next((c for c in df.columns if 'år' in c or 'year' in c), None)

        if nat_col:
            nationality_mapping = self._config.get('nationality_to_region', {})
            mapping_lower = {k.lower(): v for k, v in nationality_mapping.items()}
            
            df['birth_region_standardized'] = (
                df[nat_col].astype(str).str.strip().str.lower()
                .map(mapping_lower)
                .fillna('Other')
            )
            
        # 2. Extract numeric values (Handle Swedish spaces as thousands)
        target_name = f'mv_{self._data_type}_count'
        df[target_name] = pd.to_numeric(
            df[count_col].astype(str).str.replace(r'\s+', '', regex=True).replace('nan', '0').replace('-', '0'), 
            errors='coerce'
        ).fillna(0)
        
        if year_col:
            df = df.rename(columns={year_col: 'year'})

        # 3. Clean and Aggregate
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df = df[df['year'] > 0]
        
        if nat_col:
            df = df[df[nat_col].notna()]
            df = df[~df[nat_col].astype(str).str.lower().isin(['total', 'summa', 'samtliga'])]
            # Keep rows with valid mappings
            df = df[df['birth_region_standardized'] != 'Other']

        # Group by year and region
        df = df.groupby(['year', 'birth_region_standardized'], as_index=False)[target_name].sum()

        return df
