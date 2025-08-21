"""Data factory for creating datasets and dataloaders."""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .dataset import MgRIADataset
from .dataloader import create_dataloader
from .preprocessing import DataPreprocessor
from ..config import DatasetConfig, DatasetType
from ..utils.helpers import load_vocab


class DataFactory:
    """Factory class for creating datasets and dataloaders."""
    
    @staticmethod
    def create_datasets(config: DatasetConfig) -> Tuple[MgRIADataset, MgRIADataset]:
        """
        Create train and test datasets.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Load vocabulary if available
        vocabulary = DataFactory._load_vocabulary(config)
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        train_data, test_data = preprocessor.load_and_preprocess()
        
        # Create datasets
        train_dataset = MgRIADataset(
            config=config,
            phase='train',
            user_sequences=train_data['user_sequences'],
            time_features=train_data['time_features'],
            time_stamps=train_data['time_stamps'],
            categories=train_data['categories'],
            vocabulary=vocabulary
        )
        
        test_dataset = MgRIADataset(
            config=config,
            phase='test',
            user_sequences=test_data['user_sequences'],
            time_features=test_data['time_features'],
            time_stamps=test_data['time_stamps'],
            categories=test_data['categories'],
            vocabulary=vocabulary
        )
        
        return train_dataset, test_dataset
    
    @staticmethod
    def create_dataloaders(
        config: DatasetConfig,
        train_dataset: MgRIADataset,
        test_dataset: MgRIADataset
    ) -> Tuple[Any, Any]:
        """
        Create train and test dataloaders.
        
        Args:
            config: Dataset configuration
            train_dataset: Training dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (train_dataloader, test_dataloader)
        """
        train_loader = create_dataloader(
            train_dataset,
            config,
            shuffle=True,
            batch_size=config.batch_size
        )
        
        test_loader = create_dataloader(
            test_dataset,
            config,
            shuffle=False,
            batch_size=config.batch_size * 2  # Larger batch for testing
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def _load_vocabulary(config: DatasetConfig) -> Optional[list]:
        """Load vocabulary if available."""
        try:
            vocab_path = config.vocab_file_path
            if vocab_path.exists():
                if vocab_path.suffix == '.npy':
                    import numpy as np
                    vocabulary = np.load(vocab_path).tolist()
                else:
                    vocabulary = load_vocab(vocab_path)
                return vocabulary
        except Exception as e:
            print(f"Warning: Could not load vocabulary: {e}")
        
        return None
    
    @staticmethod
    def get_dataset_info(dataset: MgRIADataset) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'num_users': len(dataset),
            'max_len': dataset.max_len,
            'vocab_size': dataset.special_tokens.vocab_size,
            'mask_id': dataset.mask_id,
            'pad_id': dataset.pad_id,
            'dataset_type': dataset.config.dataset_type.value
        }
