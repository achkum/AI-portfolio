# ============================================================
# data package
# ============================================================
# This package contains all data loading, cleaning, merging,
# and aggregation logic.  Each data source has its own loader
# that inherits from BaseDataLoader.
# ============================================================

from data.base_loader import BaseDataLoader
from data.scb_loader import SCBLoader
from data.migrationsverket_loader import MigrationsverketLoader
from data.socialstyrelsen_loader import SocialstyrelsenLoader
from data.preprocessor import DataPreprocessor
from data.aggregator import RegionAggregator
from data.merger import DataMerger

__all__ = [
    'BaseDataLoader',
    'SCBLoader',
    'MigrationsverketLoader',
    'SocialstyrelsenLoader',
    'DataPreprocessor',
    'RegionAggregator',
    'DataMerger',
]
