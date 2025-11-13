import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

_global_config = None

def get_config() -> Config:
    """Get global configuration."""
    global _global_config
    if _global_config is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _global_config