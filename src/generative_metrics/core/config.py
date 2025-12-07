"""Core configuration management."""

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for generative metrics evaluation."""
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize configuration.
        
        Args:
            config: Optional configuration dictionary.
        """
        self._config = config or self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration.
        
        Returns:
            DictConfig: Default configuration.
        """
        default_config = {
            "seed": 42,
            "device": "auto",  # auto, cuda, mps, cpu
            "batch_size": 32,
            "num_workers": 4,
            "data": {
                "dataset": "cifar10",
                "data_dir": "./data",
                "image_size": 64,
                "normalize": True,
            },
            "metrics": {
                "fid": {
                    "enabled": True,
                    "batch_size": 64,
                    "feature_extractor": "inception_v3",
                },
                "is": {
                    "enabled": True,
                    "splits": 10,
                    "feature_extractor": "inception_v3",
                },
                "precision_recall": {
                    "enabled": True,
                    "k": 3,
                },
                "lpips": {
                    "enabled": True,
                    "net": "alex",
                },
            },
            "logging": {
                "log_dir": "./logs",
                "use_wandb": False,
                "use_tensorboard": True,
            },
            "output": {
                "save_samples": True,
                "sample_dir": "./assets/samples",
                "n_samples": 16,
            },
        }
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        self._config = OmegaConf.merge(self._config, updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the configuration object.
        
        Returns:
            DictConfig: Configuration object.
        """
        return self._config
