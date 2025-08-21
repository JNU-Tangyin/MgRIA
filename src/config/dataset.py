"""Dataset configuration classes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from .base import BaseConfig
from .constants import DatasetType, SpecialTokens


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset-specific parameters."""
    
    # Dataset selection
    dataset_type: DatasetType = DatasetType.TAFENG
    
    # Sequence parameters
    max_len: int = 50
    mask_prob: float = 0.15
    
    # Data split ratios
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Batch processing
    batch_size: int = 256
    num_workers: int = 4
    
    # Dataset-specific file paths
    dataset_files: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize dataset-specific configurations."""
        super().__post_init__()
        
        # Set default file paths based on dataset type
        if not self.dataset_files:
            self.dataset_files = self._get_default_files()
    
    def _get_default_files(self) -> Dict[str, str]:
        """Get default file paths for the selected dataset."""
        dataset_name = self.dataset_type.value
        
        return {
            'train_file': f'{dataset_name}/{dataset_name}_train.txt',
            'test_file': f'{dataset_name}/{dataset_name}_test.txt',
            'vocab_file': f'{dataset_name}/{dataset_name}_vocab.txt',
            'time_file': f'{dataset_name}/{dataset_name}_time.txt'
        }
    
    @property
    def special_tokens(self) -> SpecialTokens:
        """Get special tokens for the current dataset."""
        if self.dataset_type == DatasetType.EQUITY:
            return SpecialTokens.Equity()
        elif self.dataset_type == DatasetType.TAFENG:
            return SpecialTokens.Tafeng()
        elif self.dataset_type == DatasetType.TAOBAO:
            return SpecialTokens.Taobao()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def get_file_path(self, file_key: str) -> Path:
        """Get absolute path for a dataset file."""
        if file_key not in self.dataset_files:
            raise KeyError(f"File key '{file_key}' not found in dataset_files")
        
        return self.datasets_path / self.dataset_files[file_key]
    
    @property
    def train_file_path(self) -> Path:
        """Get path to training data file."""
        return self.get_file_path('train_file')
    
    @property
    def test_file_path(self) -> Path:
        """Get path to test data file."""
        return self.get_file_path('test_file')
    
    @property
    def vocab_file_path(self) -> Path:
        """Get path to vocabulary file."""
        return self.get_file_path('vocab_file')
    
    @property
    def time_file_path(self) -> Path:
        """Get path to time data file."""
        return self.get_file_path('time_file')
    
    def validate_files(self) -> bool:
        """Validate that all required dataset files exist."""
        required_files = ['train_file', 'test_file', 'vocab_file', 'time_file']
        
        for file_key in required_files:
            file_path = self.get_file_path(file_key)
            if not file_path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {file_path}")
        
        return True
