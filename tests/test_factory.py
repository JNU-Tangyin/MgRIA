"""Tests for model factory and ensemble."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.factory import ModelFactory, ModelEnsemble
from src.models.output import MgRIAModel, MgRIALoss
from src.config import ModelConfig, DatasetConfig, TrainingConfig
from src.config.constants import DatasetType


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def model_config():
    return ModelConfig(
        dim=64,
        vocab_size=100,
        max_len=50,
        n_layers=2,
        n_heads=4,
        attention_dropout_prob=0.1
    )


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        dataset_type=DatasetType.TAFENG
    )


class TestModelFactory:
    """Test model factory functionality."""
    
    def test_create_model_basic(self, model_config, device):
        model = ModelFactory.create_model(model_config, device)
        
        assert isinstance(model, MgRIAModel)
        assert model.device == device
        assert model.config.vocab_size == model_config.vocab_size
    
    def test_create_model_with_dataset_config(self, model_config, dataset_config, device):
        model = ModelFactory.create_model(model_config, device, dataset_config)
        
        assert isinstance(model, MgRIAModel)
        assert model.config.model_params['dataset_name'] == dataset_config.dataset_type.value
    
    def test_create_model_auto_device(self, model_config):
        with patch('src.models.factory.get_device') as mock_get_device:
            mock_device = torch.device('cpu')
            mock_get_device.return_value = mock_device
            
            model = ModelFactory.create_model(model_config)
            
            mock_get_device.assert_called_once()
            assert model.device == mock_device
    
    def test_create_loss_function(self):
        loss_fn = ModelFactory.create_loss_function(ignore_index=-100, behavior_loss_weight=0.2)
        
        assert isinstance(loss_fn, MgRIALoss)
        assert loss_fn.ignore_index == -100
    
    def test_create_model_for_dataset_equity(self, device):
        model = ModelFactory.create_model_for_dataset('equity', device=device)
        
        assert isinstance(model, MgRIAModel)
        assert model.config.vocab_size == 259
        assert model.config.model_params['dataset_name'] == 'equity'
        assert model.config.model_params['special_tokens']['pad_id'] == 257
    
    def test_create_model_for_dataset_tafeng(self, device):
        model = ModelFactory.create_model_for_dataset('tafeng', device=device)
        
        assert isinstance(model, MgRIAModel)
        assert model.config.vocab_size == 15789
        assert model.config.model_params['dataset_name'] == 'tafeng'
        assert model.config.model_params['special_tokens']['pad_id'] == 15786
    
    def test_create_model_for_dataset_taobao(self, device):
        model = ModelFactory.create_model_for_dataset('taobao', device=device)
        
        assert isinstance(model, MgRIAModel)
        assert model.config.vocab_size == 287008
        assert model.config.model_params['dataset_name'] == 'taobao'
        assert model.config.model_params['special_tokens']['pad_id'] == 287005
    
    def test_create_model_for_unknown_dataset(self, device):
        with pytest.raises(ValueError, match="Unknown dataset"):
            ModelFactory.create_model_for_dataset('unknown', device=device)
    
    def test_get_model_info(self, model_config, device):
        model = ModelFactory.create_model(model_config, device)
        info = ModelFactory.get_model_info(model)
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'device' in info
        assert 'fusion_weight' in info
        assert 'components' in info
        assert 'architecture' in info
        
        # Check component info
        assert 'transformer' in info['components']
        assert 'user_behavior' in info['components']
        assert 'output_projection' in info['components']
        
        # Check architecture info
        arch = info['architecture']
        assert arch['vocab_size'] == model_config.vocab_size
        assert arch['hidden_size'] == model_config.dim
        assert arch['num_layers'] == model_config.n_layers
    
    def test_save_and_load_model(self, model_config, device):
        # Create model
        original_model = ModelFactory.create_model(model_config, device)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_model.pt'
            
            ModelFactory.save_model(
                original_model,
                save_path,
                config=model_config,
                epoch=10,
                metrics={'loss': 0.5, 'accuracy': 0.8}
            )
            
            assert save_path.exists()
            
            # Load model
            loaded_model = ModelFactory.load_model(save_path, model_config, device)
            
            assert isinstance(loaded_model, MgRIAModel)
            assert loaded_model.config.vocab_size == original_model.config.vocab_size
    
    def test_load_nonexistent_model(self, model_config, device):
        with pytest.raises(FileNotFoundError):
            ModelFactory.load_model('/nonexistent/path.pt', model_config, device)
    
    def test_save_model_with_all_options(self, model_config, device):
        model = ModelFactory.create_model(model_config, device)
        training_config = TrainingConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'full_model.pt'
            
            ModelFactory.save_model(
                model,
                save_path,
                config=model_config,
                training_config=training_config,
                epoch=5,
                metrics={'loss': 0.3},
                optimizer_state={'lr': 0.001}
            )
            
            assert save_path.exists()
            
            # Verify checkpoint contents
            checkpoint = torch.load(save_path, weights_only=False)
            assert 'model_state_dict' in checkpoint
            assert 'model_config' in checkpoint
            assert 'training_config' in checkpoint
            assert 'epoch' in checkpoint
            assert 'metrics' in checkpoint
            assert 'optimizer_state_dict' in checkpoint


class TestModelEnsemble:
    """Test model ensemble functionality."""
    
    def test_create_ensemble(self, device):
        configs = [
            ModelConfig(dim=32, vocab_size=100, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=100, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=100, n_layers=1, n_heads=2)
        ]
        
        ensemble = ModelFactory.create_ensemble(configs, device)
        
        assert isinstance(ensemble, ModelEnsemble)
        assert ensemble.num_models == 3
        assert len(ensemble.models) == 3
    
    def test_ensemble_forward(self, device):
        configs = [
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2)
        ]
        
        ensemble = ModelFactory.create_ensemble(configs, device)
        
        # Test data
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 48, (batch_size, seq_len))
        time_features = torch.stack([
            torch.randint(1, 12, (batch_size, seq_len)),
            torch.randint(1, 31, (batch_size, seq_len)),
            torch.randint(0, 7, (batch_size, seq_len))
        ], dim=-1)
        attention_mask = torch.ones(batch_size, seq_len)
        masked_positions = torch.tensor([[1, 3], [0, 4]])
        
        output = ensemble(
            input_ids=input_ids,
            time_features=time_features,
            attention_mask=attention_mask,
            masked_positions=masked_positions
        )
        
        assert output.logits.shape == (batch_size, 2, 50)
    
    def test_ensemble_predict(self, device):
        configs = [
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2)
        ]
        
        ensemble = ModelFactory.create_ensemble(configs, device)
        
        # Test data
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 48, (batch_size, seq_len))
        time_features = torch.stack([
            torch.randint(1, 12, (batch_size, seq_len)),
            torch.randint(1, 31, (batch_size, seq_len)),
            torch.randint(0, 7, (batch_size, seq_len))
        ], dim=-1)
        attention_mask = torch.ones(batch_size, seq_len)
        
        items, scores = ensemble.predict(
            input_ids=input_ids,
            time_features=time_features,
            attention_mask=attention_mask,
            top_k=5
        )
        
        assert items.shape == (batch_size, 5)
        assert scores.shape == (batch_size, 5)
    
    def test_ensemble_model_size(self, device):
        configs = [
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2)
        ]
        
        ensemble = ModelFactory.create_ensemble(configs, device)
        
        total_size = ensemble.get_model_size()
        trainable_size = ensemble.get_trainable_parameters()
        
        assert isinstance(total_size, int)
        assert isinstance(trainable_size, int)
        assert total_size > 0
        assert trainable_size > 0
        assert trainable_size <= total_size


class TestIntegration:
    """Integration tests for factory functionality."""
    
    def test_full_workflow(self, device):
        """Test complete workflow from creation to saving/loading."""
        # Create model using factory
        model = ModelFactory.create_model_for_dataset('equity', device=device)
        
        # Get model info
        info = ModelFactory.get_model_info(model)
        assert info['architecture']['vocab_size'] == 259
        
        # Create loss function
        loss_fn = ModelFactory.create_loss_function()
        
        # Test forward pass
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 256, (batch_size, seq_len))
        time_features = torch.stack([
            torch.randint(1, 12, (batch_size, seq_len)),
            torch.randint(1, 31, (batch_size, seq_len)),
            torch.randint(0, 7, (batch_size, seq_len))
        ], dim=-1)
        attention_mask = torch.ones(batch_size, seq_len)
        masked_positions = torch.tensor([[1, 3], [0, 4]])
        
        output = model(
            input_ids=input_ids,
            time_features=time_features,
            attention_mask=attention_mask,
            masked_positions=masked_positions
        )
        
        # Test loss computation
        targets = torch.randint(0, 259, (batch_size, 2))
        loss = loss_fn(output.logits, targets)
        assert isinstance(loss, torch.Tensor)
        
        # Test save/load cycle
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'equity_model.pt'
            
            ModelFactory.save_model(model, save_path)
            loaded_model = ModelFactory.load_model(save_path, model.config, device)
            
            # Verify loaded model works
            loaded_output = loaded_model(
                input_ids=input_ids,
                time_features=time_features,
                attention_mask=attention_mask,
                masked_positions=masked_positions
            )
            
            assert loaded_output.logits.shape == output.logits.shape
    
    def test_ensemble_workflow(self, device):
        """Test ensemble creation and usage."""
        configs = [
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2),
            ModelConfig(dim=32, vocab_size=50, n_layers=1, n_heads=2)
        ]
        
        ensemble = ModelFactory.create_ensemble(configs, device)
        
        # Test ensemble info
        total_params = ensemble.get_model_size()
        single_model_params = ModelFactory.create_model(configs[0], device).get_model_size()
        
        # Ensemble should have approximately 3x the parameters
        assert total_params >= 2.5 * single_model_params
        assert total_params <= 3.5 * single_model_params
    
    def test_different_dataset_models(self, device):
        """Test creating models for different datasets."""
        datasets = ['equity', 'tafeng', 'taobao']
        expected_vocab_sizes = [259, 15789, 287008]
        
        for dataset, expected_vocab in zip(datasets, expected_vocab_sizes):
            model = ModelFactory.create_model_for_dataset(dataset, device=device)
            info = ModelFactory.get_model_info(model)
            
            assert info['architecture']['vocab_size'] == expected_vocab
            assert model.config.model_params['dataset_name'] == dataset
