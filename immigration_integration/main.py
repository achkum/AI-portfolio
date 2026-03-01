# ============================================================
# main.py  –  Main Pipeline Script (Income & Welfare Focus)
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from commons.config_loader import ConfigLoader
from commons.constants import OUTCOME_COLUMNS
from commons.utils import ensure_directory
from commons.visualizer import IntegrationVisualizer

from data.scb_loader import SCBLoader
from data.migrationsverket_loader import MigrationsverketLoader
from data.socialstyrelsen_loader import SocialstyrelsenLoader
from data.preprocessor import DataPreprocessor
from data.aggregator import RegionAggregator
from data.merger import DataMerger


def main():
    config = ConfigLoader()
    cfg = config.config

    figures_dir = ensure_directory(os.path.join(PROJECT_ROOT, cfg['paths']['figures_dir']))
    processed_dir = ensure_directory(os.path.join(PROJECT_ROOT, cfg['paths']['processed_data_dir']))

    start_year = cfg['data']['time_period']['start_year']
    end_year = cfg['data']['time_period']['end_year']

    loaded_sources = {}

    def try_load(path, loader_cls, **kwargs):
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            loader = loader_cls(abs_path, cfg, **kwargs)
            return loader.load()
        return None

    # Load sources
    for key, path in cfg['paths']['scb'].items():
        print(f"Loading SCB {key} from {path}...", flush=True)
        loader = try_load(path, SCBLoader, indicator=key)
        if loader:
            loaded_sources[f'scb_{key}'] = loader.filter_years(start_year, end_year).data
            print(f"  Done. Rows: {len(loaded_sources[f'scb_{key}'])}", flush=True)

    print("Loading Migrationsverket data...", flush=True)
    for dtype in ['asylum', 'permits']:
        mv = try_load(cfg['paths']['migrationsverket'][dtype], MigrationsverketLoader, data_type=dtype)
        if mv:
            loaded_sources[f'mv_{dtype}'] = mv.data
            print(f"  Migrationsverket {dtype} Loaded. Rows: {len(mv.data)}", flush=True)

    print("Loading Socialstyrelsen data...", flush=True)
    soc = try_load(cfg['paths']['socialstyrelsen']['assistance'], SocialstyrelsenLoader)
    if soc: 
        loaded_sources['socialstyrelsen'] = soc.data
        print(f"  Welfare Loaded. Rows: {len(soc.data)}", flush=True)

    if not loaded_sources:
        print("ERROR: No data sources found in datasource/raw/.", flush=True)
        return

    # 1. Merge and Preprocess
    print("\nMerging and Preprocessing data...", flush=True)
    merger = DataMerger()
    for name, df in loaded_sources.items():
        merger.add_source(df, name=name)

    # Use Proxy Broadcasting (handled in merger.py)
    merged = merger.merge_on_keys(keys=['birth_region_standardized', 'year'], how='outer')
    
    # Preprocessing: EXCLUDE outcome columns from broad median filling to keep regional contrasts
    outcome_cols = list(OUTCOME_COLUMNS.keys()) + ['welfare_amount_avg']
    preproc = DataPreprocessor(merged)
    # Filter only regions matching standard SCB/MV categories
    cleaned = preproc.fill_missing(strategy='median', exclude_cols=outcome_cols).data
    
    # Drop artifacts
    drop_cols = [c for c in cleaned.columns if 'unnamed' in c.lower() or 'region' in c.lower() and c != 'birth_region_standardized']
    cleaned = cleaned.drop(columns=drop_cols)
    cleaned.to_csv(os.path.join(processed_dir, "merged_national.csv"), index=False)

    # 2. Visualization Suite
    available_outcomes = [c for c in outcome_cols if c in cleaned.columns]
    
    if available_outcomes:
        print(f"\nGenerating comparative visualizations...", flush=True)
        # Filter for plotting: show real regional variance vs Sweden baseline
        viz_data = cleaned[~cleaned['birth_region_standardized'].isin(['Total', 'Other'])]
        viz_data = viz_data.sort_values(by=['year', 'birth_region_standardized'])
        
        viz = IntegrationVisualizer(viz_data)
        
        # Comparative Charts
        for outcome in available_outcomes:
            if viz_data[outcome].nunique() > 1:
                print(f"  Creating plots for {outcome}...", flush=True)
                viz.plot_outcome_by_country(outcome)
                viz.plot_trend_by_country(outcome)
        
        # Dual Plot: Income vs Welfare
        if 'median_income' in viz_data.columns and 'welfare_amount_avg' in viz_data.columns:
            print(f"  Creating dual Comparison: Income vs Welfare Assistance...", flush=True)
            viz.plot_dual_outcome_side_by_side('median_income', 'welfare_amount_avg', 
                                              "Annual Median Income vs Representative Welfare Support")
            
        viz.save_all_figures(figures_dir, prefix='integration_regional')
        print(f"  Done. Figures saved to {figures_dir}", flush=True)

    aggregator = RegionAggregator(cleaned)
    region_summary = aggregator.aggregate_by_region(available_outcomes)
    region_summary.to_csv(os.path.join(processed_dir, "integration_by_country.csv"), index=False)
    
    print("\n[Analysis Suite Complete]: Broad regional comparisons are ready for review.", flush=True)


if __name__ == '__main__':
    main()
