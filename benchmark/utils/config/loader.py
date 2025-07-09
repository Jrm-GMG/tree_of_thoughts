"""Configuration loader for benchmarks"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import difflib
class ConfigLoader:
    """Handles loading and merging configuration files"""

    def __init__(self, base_path: str = "benchmark/config"):
        """Create a loader rooted at base_path containing configs.

        Args:
            base_path: Directory containing configuration files.
        """
        self.base_path = Path(base_path)
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Relative path to the YAML configuration file.

        Returns:
            Parsed configuration dictionary after applying inheritance.
        """
        full_path = self.base_path / config_path
        
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # For inheritance
        if 'extends' in config:
            base_config = self.load(config['extends'])
            config = self._merge_configs(base_config, config)
            del config['extends']
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base: Base configuration that can be overridden.
            override: New configuration whose values take precedence.

        Returns:
            The merged configuration dictionary.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


def load_config(config_name: str) -> Dict[str, Any]:
    """Convenience function to load a configuration.

    Args:
        config_name: Relative path to the configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    loader = ConfigLoader()
    return loader.load(config_name)