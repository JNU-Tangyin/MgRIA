"""Tests for output layer and MgRIA model integration."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.models.output import (
    MgRIAModel,
    MgRIALoss,
    OutputProjection,
    LayerNorm,
    MgRIAOutput,
    gelu_activation
)
from src.config import ModelConfig


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def model_config():
    config = ModelConfig(
        dim=64,
        vocab_size=100,
        max_len=50,
        n_layers=2,
        n_heads=4,
        attention_dropout_prob=0.1
    )
    # Add dataset_name for user behavior module
    config.model_params['dataset_name'] = 'tafeng'
    return config


@pytest.fixture
def sample_input():
    batch_size, seq_len, time_dim = 2, 10, 3  # time_dim should be 3 for [weekday, day, month]
    return {
        'input_ids': torch.randint(0, 98, (batch_size, seq_len)),  # Safe range for vocab_size=100
        'time_features': torch.stack([
            torch.randint(1, 12, (batch_size, seq_len)),   # month: 1-11 (index 0)
            torch.randint(1, 31, (batch_size, seq_len)),   # day: 1-30 (index 1)
            torch.randint(0, 7, (batch_size, seq_len))     # weekday: 0-6 (index 2)
        ], dim=-1),  # [batch_size, seq_len, 3] - order: [month, day, weekday]
        'attention_mask': torch.ones(batch_size, seq_len),
        'masked_positions': torch.randint(0, seq_len, (batch_size, 3)),  # 3 predictions per batch
        'time_gaps': torch.randint(1, 180, (batch_size, seq_len)).float(),
        'categories': torch.randint(0, 4, (batch_size, seq_len))
    }


class TestGeluActivation:
    """Test GELU activation function."""
    
    def test_gelu_activation(self):
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        result = gelu_activation(x)
        
        assert result.shape == x.shape
        assert torch.allclose(result[0], torch.tensor(0.0), atol=1e-6)  # GELU(0) ≈ 0
        assert result[1] > 0  # GELU(1) > 0
        assert result[2] < 0  # GELU(-1) < 0


class TestLayerNorm:
    """Test layer normalization module."""
    
    def test_layer_norm_init(self, model_config):
        layer_norm = LayerNorm(model_config)
        
        assert layer_norm.weight.shape == (model_config.dim,)
        assert layer_norm.bias.shape == (model_config.dim,)
        assert layer_norm.eps == model_config.layer_norm_eps
    
    def test_layer_norm_forward(self, model_config):
        layer_norm = LayerNorm(model_config)
        x = torch.randn(2, 10, model_config.dim)
        
        output = layer_norm(x)
        
        assert output.shape == x.shape
        # Check normalization properties
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-6)


class TestOutputProjection:
    """Test output projection layer."""
    
    def test_output_projection_init(self, model_config):
        projection = OutputProjection(model_config)
        
        assert isinstance(projection.linear, nn.Linear)
        assert isinstance(projection.norm, LayerNorm)
        assert isinstance(projection.decoder, nn.Linear)
        assert projection.decoder.out_features == model_config.vocab_size
        assert projection.decoder_bias.shape == (model_config.vocab_size,)
    
    def test_output_projection_forward(self, model_config):
        projection = OutputProjection(model_config)
        
        batch_size, seq_len = 2, 10
        num_predictions = 3
        hidden_states = torch.randn(batch_size, seq_len, model_config.dim)
        masked_positions = torch.randint(0, seq_len, (batch_size, num_predictions))
        
        logits = projection(hidden_states, masked_positions)
        
        assert logits.shape == (batch_size, num_predictions, model_config.vocab_size)


class TestMgRIAModel:
    """Test complete MgRIA model."""
    
    def test_mgria_model_init(self, model_config, device):
        model = MgRIAModel(model_config, device)
        
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'user_behavior')
        assert hasattr(model, 'output_projection')
        assert hasattr(model, 'fusion_weight')
        assert isinstance(model.fusion_weight, nn.Parameter)
    
    def test_mgria_model_forward(self, model_config, device):
        model = MgRIAModel(model_config, device)
        
        # Simple test data
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 98, (batch_size, seq_len))
        time_features = torch.stack([
            torch.randint(1, 12, (batch_size, seq_len)),   # month: 1-11
            torch.randint(1, 31, (batch_size, seq_len)),   # day: 1-30
            torch.randint(0, 7, (batch_size, seq_len))     # weekday: 0-6
        ], dim=-1)
        attention_mask = torch.ones(batch_size, seq_len)
        masked_positions = torch.tensor([[1, 3], [0, 4]])  # 2 predictions per batch
        
        output = model(
            input_ids=input_ids,
            time_features=time_features,
            attention_mask=attention_mask,
            masked_positions=masked_positions
        )
        
        assert isinstance(output, MgRIAOutput)
        assert output.logits.shape == (batch_size, 2, model_config.vocab_size)
        assert hasattr(output, 'transformer_output')
        assert hasattr(output, 'behavior_output')
        assert hasattr(output, 'fusion_weight')
    
    def test_mgria_model_forward_with_temporal(self, model_config, device, sample_input):
        # Set up equity config for temporal modeling
        model_config.model_params['dataset_name'] = 'equity'
        model = MgRIAModel(model_config, device)
        
        output = model(
            input_ids=sample_input['input_ids'],
            time_features=sample_input['time_features'],
            attention_mask=sample_input['attention_mask'],
            masked_positions=sample_input['masked_positions'],
            time_gaps=sample_input['time_gaps'],
            categories=sample_input['categories']
        )
        
        assert isinstance(output, MgRIAOutput)
        batch_size, num_predictions = sample_input['masked_positions'].shape
        assert output.logits.shape == (batch_size, num_predictions, model_config.vocab_size)
    
    def test_mgria_model_predict(self, model_config, device, sample_input):
        model = MgRIAModel(model_config, device)
        
        top_k_items, top_k_scores = model.predict(
            input_ids=sample_input['input_ids'],
            time_features=sample_input['time_features'],
            attention_mask=sample_input['attention_mask'],
            top_k=5
        )
        
        batch_size = sample_input['input_ids'].shape[0]
        assert top_k_items.shape == (batch_size, 5)
        assert top_k_scores.shape == (batch_size, 5)
        
        # Check that scores are probabilities
        assert torch.all(top_k_scores >= 0)
        assert torch.all(top_k_scores <= 1)
        
        # Check that scores are sorted in descending order
        for i in range(batch_size):
            scores = top_k_scores[i]
            assert torch.all(scores[:-1] >= scores[1:])
    
    def test_model_size_methods(self, model_config, device):
        model = MgRIAModel(model_config, device)
        
        total_params = model.get_model_size()
        trainable_params = model.get_trainable_parameters()
        
        assert isinstance(total_params, int)
        assert isinstance(trainable_params, int)
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


class TestMgRIALoss:
    """Test MgRIA loss function."""
    
    def test_mgria_loss_init(self):
        loss_fn = MgRIALoss(ignore_index=-100)
        
        assert loss_fn.ignore_index == -100
        assert isinstance(loss_fn.cross_entropy, nn.CrossEntropyLoss)
    
    def test_mgria_loss_forward_basic(self, model_config):
        loss_fn = MgRIALoss()
        
        batch_size, num_predictions, vocab_size = 2, 3, model_config.vocab_size
        logits = torch.randn(batch_size, num_predictions, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, num_predictions))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0
    
    def test_mgria_loss_forward_with_behavior(self, model_config):
        loss_fn = MgRIALoss()
        
        batch_size, num_predictions, vocab_size = 2, 3, model_config.vocab_size
        logits = torch.randn(batch_size, num_predictions, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, num_predictions))
        
        behavior_probs = torch.randn(batch_size, 2)
        behavior_targets = torch.randint(0, 2, (batch_size,))
        
        loss = loss_fn(
            logits=logits,
            targets=targets,
            behavior_probs=behavior_probs,
            behavior_targets=behavior_targets,
            behavior_loss_weight=0.1
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_mgria_loss_with_ignore_index(self, model_config):
        loss_fn = MgRIALoss(ignore_index=-100)
        
        batch_size, num_predictions, vocab_size = 2, 3, model_config.vocab_size
        logits = torch.randn(batch_size, num_predictions, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, num_predictions))
        
        # Set some targets to ignore index
        targets[0, 0] = -100
        targets[1, 2] = -100
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestMgRIAOutput:
    """Test MgRIA output dataclass."""
    
    def test_mgria_output_creation(self, model_config):
        batch_size, num_predictions, vocab_size = 2, 3, model_config.vocab_size
        
        logits = torch.randn(batch_size, num_predictions, vocab_size)
        transformer_output = Mock()
        behavior_output = Mock()
        fusion_weight = torch.tensor(0.5)
        
        output = MgRIAOutput(
            logits=logits,
            transformer_output=transformer_output,
            behavior_output=behavior_output,
            fusion_weight=fusion_weight
        )
        
        assert torch.equal(output.logits, logits)
        assert output.transformer_output == transformer_output
        assert output.behavior_output == behavior_output
        assert torch.equal(output.fusion_weight, fusion_weight)


class TestIntegration:
    """Integration tests for the complete MgRIA pipeline."""
    
    def test_end_to_end_training_step(self, model_config, device, sample_input):
        """Test a complete training step."""
        model = MgRIAModel(model_config, device)
        loss_fn = MgRIALoss()
        
        # Forward pass
        output = model(
            input_ids=sample_input['input_ids'],
            time_features=sample_input['time_features'],
            attention_mask=sample_input['attention_mask'],
            masked_positions=sample_input['masked_positions']
        )
        
        # Create targets
        batch_size, num_predictions = sample_input['masked_positions'].shape
        targets = torch.randint(0, model_config.vocab_size, (batch_size, num_predictions))
        
        # Compute loss
        loss = loss_fn(output.logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert any(p.grad is not None for p in model.parameters())
    
    def test_inference_mode(self, model_config, device, sample_input):
        """Test model in inference mode."""
        model = MgRIAModel(model_config, device)
        model.eval()
        
        with torch.no_grad():
            output = model(
                input_ids=sample_input['input_ids'],
                time_features=sample_input['time_features'],
                attention_mask=sample_input['attention_mask'],
                masked_positions=sample_input['masked_positions'],
                phase='test'
            )
        
        assert isinstance(output, MgRIAOutput)
        assert output.logits.requires_grad is False
    
    def test_different_batch_sizes(self, model_config, device):
        """Test model with different batch sizes."""
        model = MgRIAModel(model_config, device)
        
        for batch_size in [1, 2, 4]:
            seq_len, time_dim = 10, 16
            num_predictions = 2
            
            input_data = {
                'input_ids': torch.randint(0, 98, (batch_size, seq_len)),
                'time_features': torch.stack([
                    torch.randint(1, 12, (batch_size, seq_len)),  # month: 1-11
                    torch.randint(1, 31, (batch_size, seq_len)),  # day: 1-30
                    torch.randint(0, 7, (batch_size, seq_len))    # weekday: 0-6
                ], dim=-1),
                'attention_mask': torch.ones(batch_size, seq_len),
                'masked_positions': torch.randint(0, seq_len, (batch_size, num_predictions))
            }
            
            output = model(**input_data)
            
            assert output.logits.shape == (batch_size, num_predictions, model_config.vocab_size)
    
    def test_gradient_flow(self, model_config, device, sample_input):
        """Test that gradients flow through all components."""
        model = MgRIAModel(model_config, device)
        loss_fn = MgRIALoss()
        
        # Forward pass
        output = model(
            input_ids=sample_input['input_ids'],
            time_features=sample_input['time_features'],
            attention_mask=sample_input['attention_mask'],
            masked_positions=sample_input['masked_positions']
        )
        
        # Create targets and compute loss
        batch_size, num_predictions = sample_input['masked_positions'].shape
        targets = torch.randint(0, model_config.vocab_size, (batch_size, num_predictions))
        loss = loss_fn(output.logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Check gradients in all major components
        assert model.transformer.embeddings.token_embed.weight.grad is not None
        assert model.user_behavior.behavior_predictor.behavior_classifier.weight.grad is not None
        assert model.output_projection.decoder.weight.grad is not None
        assert model.fusion_weight.grad is not None
