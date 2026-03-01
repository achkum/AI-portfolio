# ============================================================
# visualizer.py  –  Visualization Classes
# ============================================================
# Provides IntegrationVisualizer with methods for bar charts,
# heatmaps, and trend lines.
# ============================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np


class IntegrationVisualizer:
    """Visualization class for integration outcomes."""

    DEFAULT_FIGSIZE: Tuple[int, int] = (14, 7)
    DEFAULT_STYLE: str = 'seaborn-v0_8-whitegrid'
    COLOR_PALETTE: Dict[str, str] = {
        'Sweden':                  '#1f77b4',
        'Nordic (excl. Sweden)':   '#2ca02c',
        'EU/EFTA (excl. Nordic)':  '#9467bd',
        'Europe (excl. EU/EFTA)':  '#17becf',
        'Africa':                  '#d62728',
        'Asia':                    '#ff7f0e',
        'Middle East':             '#8c564b',
        'North America':           '#bcbd22',
        'South America':           '#e377c2',
        'Oceania':                 '#7f7f7f',
        'Other':                   '#7f7f7f',
        'Foreign Born (Total)':     '#2b2b2b'
    }

    def __init__(self, data: pd.DataFrame, style: Optional[str] = None):
        self._data = data.copy()
        self._style = style or self.DEFAULT_STYLE
        self._figures: List[plt.Figure] = []
        plt.style.use(self._style)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    def plot_outcome_by_country(self,
                                outcome_col: str,
                                title: Optional[str] = None) -> plt.Figure:
        """Horizontal bar chart comparing an outcome across birth regions."""
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)

        grouped = self._data.groupby('birth_region_standardized')[outcome_col].mean().dropna()
        grouped = grouped.sort_values(ascending=True)

        if grouped.empty: return None

        colors = [self.COLOR_PALETTE.get(region, '#7f7f7f') for region in grouped.index]
        grouped.plot(kind='barh', ax=ax, color=colors)

        ax.set_xlabel(outcome_col.replace('_', ' ').title())
        ax.set_ylabel('Region of Birth')
        ax.set_title(title or f'{outcome_col.replace("_", " ").title()} by Region')

        for i, (idx, val) in enumerate(grouped.items()):
            label = f"{val:,.0f}" if val > 100 else f"{val:.1f}"
            ax.text(val + (val*0.01), i, label, va='center', fontweight='bold')

        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_trend_by_country(self,
                                outcome_col: str,
                                top_n: int = 8) -> plt.Figure:
        """Line plot showing outcome trends over time for regions."""
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)

        regions = self._data['birth_region_standardized'].unique()
        
        for region in regions:
            region_data = self._data[self._data['birth_region_standardized'] == region]
            yearly = region_data.groupby('year')[outcome_col].mean().dropna()
            if yearly.empty: continue

            color = self.COLOR_PALETTE.get(region, None)
            ax.plot(yearly.index, yearly.values, marker='o',
                    label=region, color=color, linewidth=2.5)

        ax.set_xlabel('Year')
        ax.set_ylabel(outcome_col.replace('_', ' ').title())
        ax.set_title(f'{outcome_col.replace("_", " ").title()} Trends (2013-2024)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def plot_dual_outcome_side_by_side(self, 
                                      col1: str, 
                                      col2: str,
                                      title: str = "Income vs Welfare Comparison") -> plt.Figure:
        """Grouped bar chart for direct comparison of two outcomes (e.g. Income and Welfare)."""
        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        
        # Prepare data: Average over the most recent 3 years for stability
        latest_years = sorted(self._data['year'].unique())[-3:]
        recent_data = self._data[self._data['year'].isin(latest_years)]
        
        summary = recent_data.groupby('birth_region_standardized')[[col1, col2]].mean().dropna()
        summary = summary.sort_values(by=col1, ascending=False)
        
        if summary.empty: return None

        # Scaling: If one is in thousands and other is absolute, align them?
        # Median income is in SEK thousands (e.g. 230). Welfare is monthly avg (e.g. 10000).
        # To compare yearly income vs yearly welfare aid:
        # We'll plot them as-is but label the units.
        
        x = np.arange(len(summary.index))
        width = 0.35
        
        ax.bar(x - width/2, summary[col1], width, label=col1.replace('_', ' ').title(), color='#1f77b4')
        ax.bar(x + width/2, summary[col2] / 10, width, label=f"{col2.replace('_', ' ').title()} (scaled 1/10)", color='#d62728')
        
        ax.set_ylabel('Value (Income in KSEK / Welfare/10)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=30, ha='right')
        ax.legend()
        
        plt.tight_layout()
        self._figures.append(fig)
        return fig

    def save_all_figures(self, output_dir: str, prefix: str = 'fig') -> List[str]:
        """Save every generated figure to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        for i, fig in enumerate(self._figures):
            path = os.path.join(output_dir, f'{prefix}_{i + 1}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved_paths.append(path)

        return saved_paths
