"""Test configuration modules."""

import pytest
import tempfile
from pathlib import Path

from src.config import BaseConfig, DatasetConfig, ModelConfig, TrainingConfig
from src.config.constants import DatasetType, ModelType


class TestBaseConfig:
    """Test BaseConfig class."""
    
    def test_init(self, project_root):
        """Test BaseConfig initialization."""
        config = BaseConfig(project_root=project_root)
        assert config.project_root == project_root
        assert config.seed == 2300
        assert config.device == "auto"
    
    def test_path_properties(self, test_config):
        """Test path properties."""
        assert test_config.datasets_path.name == "datasets"
        assert test_config.results_path.name == "results"
        assert test_config.figures_path.name == "figures"
        assert test_config.tables_path.name == "tables"
        assert test_config.models_path.name == "models"
    
    def test_yaml_serialization(self, test_config):
        """Test YAML save/load functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save to YAML
            test_config.to_yaml(yaml_path)
            
            # Load from YAML
            loaded_config = BaseConfig.from_yaml(yaml_path)
            
            assert loaded_config.seed == test_config.seed
            assert loaded_config.device == test_config.device
        finally:
            Path(yaml_path).unlink()


class TestDatasetConfig:
    """Test DatasetConfig class."""
    
    def test_init(self, project_root):
        """Test DatasetConfig initialization."""
        config = DatasetConfig(
            project_root=project_root,
            dataset_type=DatasetType.TAFENG
        )
        assert config.dataset_type == DatasetType.TAFENG
        assert config.max_len == 50
        assert config.batch_size == 256
    
    def test_special_tokens(self, test_dataset_config):
        """Test special tokens property."""
        tokens = test_dataset_config.special_tokens
        assert hasattr(tokens, 'vocab_size')
        assert hasattr(tokens, 'mask_id')
        assert hasattr(tokens, 'pad_id')
        assert hasattr(tokens, 'interest_id')
    
    def test_file_paths(self, test_dataset_config):
        """Test file path methods."""
        train_path = test_dataset_config.train_file_path
        test_path = test_dataset_config.test_file_path
        vocab_path = test_dataset_config.vocab_file_path
        time_path = test_dataset_config.time_file_path
        
        assert train_path.name.endswith('_train.txt')
        assert test_path.name.endswith('_test.txt')
        assert vocab_path.name.endswith('_vocab.txt')
        assert time_path.name.endswith('_time.txt')


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_init(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            vocab_size=1000,
            dim=64,
            n_layers=2,
            n_heads=4
        )
        assert config.vocab_size == 1000
        assert config.dim == 64
        assert config.n_layers == 2
        assert config.n_heads == 4
    
    def test_head_dim(self, test_model_config):
        """Test head dimension calculation."""
        assert test_model_config.head_dim == test_model_config.dim // test_model_config.n_heads
    
    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            ModelConfig(dim=63, n_heads=4)  # Not divisible
        
        with pytest.raises(ValueError):
            ModelConfig(n_layers=0)  # Non-positive
        
        with pytest.raises(ValueError):
            ModelConfig(vocab_size=-1)  # Non-positive
    
    def test_get_model_params(self, test_model_config):
        """Test model parameters extraction."""
        params = test_model_config.get_model_params()
        
        assert 'vocab_size' in params
        assert 'dim' in params
        assert 'n_layers' in params
        assert 'n_heads' in params


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_init(self):
        """Test TrainingConfig initialization."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=128,
            max_epochs=100
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 128
        assert config.max_epochs == 100
    
    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1)  # Non-positive
        
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)  # Non-positive
        
        with pytest.raises(ValueError):
            TrainingConfig(max_epochs=-1)  # Non-positive
    
    def test_optimizer_params(self, test_training_config):
        """Test optimizer parameters."""
        params = test_training_config.get_optimizer_params()
        
        assert 'lr' in params
        assert 'weight_decay' in params
        assert 'eps' in params
        assert 'betas' in params
    
    def test_is_better_metric(self, test_training_config):
        """Test metric comparison."""
        # Greater is better (default)
        assert test_training_config.is_better_metric(0.8, 0.7) == True
        assert test_training_config.is_better_metric(0.6, 0.7) == False
        
        # Lower is better
        test_training_config.greater_is_better = False
        assert test_training_config.is_better_metric(0.6, 0.7) == True
        assert test_training_config.is_better_metric(0.8, 0.7) == False
