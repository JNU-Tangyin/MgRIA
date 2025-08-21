"""Base configuration classes."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""
    
    # Project paths (relative to project root)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    datasets_dir: str = "datasets"
    results_dir: str = "results" 
    figures_dir: str = "figures"
    tables_dir: str = "tables"
    models_dir: str = "models"
    
    # Random seed for reproducibility
    seed: int = 2300
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        self.project_root = Path(self.project_root)
        
    @property
    def datasets_path(self) -> Path:
        """Get absolute path to datasets directory."""
        return self.project_root / self.datasets_dir
    
    @property
    def results_path(self) -> Path:
        """Get absolute path to results directory."""
        return self.project_root / self.results_dir
    
    @property
    def figures_path(self) -> Path:
        """Get absolute path to figures directory."""
        return self.project_root / self.figures_dir
    
    @property
    def tables_path(self) -> Path:
        """Get absolute path to tables directory."""
        return self.project_root / self.tables_dir
    
    @property
    def models_path(self) -> Path:
        """Get absolute path to models directory."""
        return self.project_root / self.models_dir
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def update_vocab_size(self, vocab_size: int) -> None:
        """Update vocabulary size."""
        self.vocab_size = vocab_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, Path):
                result[field] = str(value)
            elif hasattr(value, 'to_dict'):
                result[field] = value.to_dict()
            elif isinstance(value, dict):
                result[field] = value.copy()
            else:
                result[field] = value
        return result
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
                
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def update(self, **kwargs) -> 'BaseConfig':
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
