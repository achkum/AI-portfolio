# ============================================================
# config_loader.py  –  YAML Configuration Parser
# ============================================================
# Loads settings from config/config.yaml so every other module
# can access paths, parameters, and mappings from one place.
# ============================================================

import os
import yaml
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Loads and provides access to YAML configuration files.

    Attributes
    ----------
    _config : dict
        The parsed YAML configuration dictionary.
    _config_path : str
        Absolute path to the loaded configuration file.
    """

    def __init__(self, config_path: str = None):
        """
        Parameters
        ----------
        config_path : str, optional
            Path to the YAML config file.  If not given, defaults to
            ``config/config.yaml`` relative to the project root.
        """
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "config", "config.yaml")

        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Read and parse the YAML configuration file."""
        if os.path.exists(self._config_path):
            with open(self._config_path, "r", encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a top-level configuration value."""
        return self._config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Walk into nested dictionaries using a sequence of keys."""
        current = self._config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
            if current is None:
                return default
        return current

    @property
    def config(self) -> Dict[str, Any]:
        """Return a *copy* of the full config dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"ConfigLoader(path='{self._config_path}')"
