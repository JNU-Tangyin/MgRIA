"""Tests for user behavior modules."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock

from src.models.user_behavior import (
    UserBehaviorModule,
    BehaviorPredictor,
    RepeatDecoder,
    ExploreDecoder,
    AdditiveAttention,
    MixtureDistribution,
    create_onehot_map,
    UserBehaviorOutput
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
    # Add dataset_name to model_params for user behavior module
    config.model_params['dataset_name'] = 'tafeng'
    return config


@pytest.fixture
def equity_config():
    config = ModelConfig(
        dim=64,
        vocab_size=100,
        max_len=50,
        n_layers=2,
        n_heads=4,
        attention_dropout_prob=0.1
    )
    # Add dataset_name to model_params for user behavior module
    config.model_params['dataset_name'] = 'equity'
    return config


@pytest.fixture
def sample_data():
    batch_size, seq_len, hidden_size = 2, 10, 64
    return {
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
        'input_ids': torch.randint(0, 98, (batch_size, seq_len)),  # Avoid special tokens
        'time_gaps': torch.randint(1, 180, (batch_size, seq_len)).float(),
        'categories': torch.randint(0, 4, (batch_size, seq_len))
    }


class TestCreateOnehotMap:
    """Test one-hot encoding utility."""
    
    def test_onehot_creation(self):
        indices = torch.tensor([[1, 2, 0], [3, 1, 2]])
        vocab_size = 5
        
        onehot = create_onehot_map(indices, vocab_size)
        
        assert onehot.shape == (2, 3, 5)
        assert torch.allclose(onehot[0, 0], torch.tensor([0., 1., 0., 0., 0.]))
        assert torch.allclose(onehot[1, 0], torch.tensor([0., 0., 0., 1., 0.]))
    
    def test_onehot_no_gradients(self):
        indices = torch.tensor([[1, 2]])
        onehot = create_onehot_map(indices, 5)
        assert not onehot.requires_grad


class TestAdditiveAttention:
    """Test additive attention mechanism."""
    
    def test_attention_forward(self):
        attention = AdditiveAttention(hidden_size=64)
        hidden_states = torch.randn(2, 10, 64)
        
        context, weights = attention(hidden_states)
        
        assert context.shape == (2, 64)
        assert weights.shape == (2, 10)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    
    def test_attention_with_interest_state(self):
        attention = AdditiveAttention(hidden_size=64)
        hidden_states = torch.randn(2, 10, 64)
        interest_state = torch.randn(2, 1, 64)
        
        context, weights = attention(hidden_states, interest_state)
        
        assert context.shape == (2, 64)
        assert weights.shape == (2, 10)


class TestBehaviorPredictor:
    """Test behavior prediction module."""
    
    def test_behavior_predictor_init(self, model_config):
        predictor = BehaviorPredictor(model_config)
        
        assert isinstance(predictor.attention, AdditiveAttention)
        assert isinstance(predictor.behavior_classifier, nn.Linear)
        assert predictor.behavior_classifier.out_features == 2
    
    def test_behavior_predictor_forward(self, model_config, sample_data):
        predictor = BehaviorPredictor(model_config)
        hidden_states = sample_data['hidden_states']
        
        behavior_logits = predictor(hidden_states)
        
        assert behavior_logits.shape == (2, 2)  # [batch_size, 2]


class TestMixtureDistribution:
    """Test mixture distribution for temporal modeling."""
    
    def test_mixture_distribution_init(self):
        mixture = MixtureDistribution()
        
        assert len(mixture.gaussian_weights) == 6
        assert len(mixture.gaussian_means) == 4
        assert len(mixture.gaussian_stds) == 4
        assert len(mixture.power_params) == 2
    
    def test_distribution_modes(self):
        mixture = MixtureDistribution()
        time_gaps = torch.tensor([[30.0, 60.0], [45.0, 90.0]])
        
        for mode in range(4):
            if mode == 3:  # Skip category 3 in actual implementation
                continue
            probs = mixture.compute_distribution(time_gaps, mode)
            assert probs.shape == time_gaps.shape
            assert torch.all(probs >= 0)  # Probabilities should be non-negative


class TestRepeatDecoder:
    """Test repeat behavior decoder."""
    
    def test_repeat_decoder_init(self, model_config, device):
        decoder = RepeatDecoder(model_config, device)
        
        assert isinstance(decoder.attention, AdditiveAttention)
        assert decoder.vocab_size == model_config.vocab_size
    
    def test_repeat_decoder_forward_basic(self, model_config, device, sample_data):
        decoder = RepeatDecoder(model_config, device)
        
        repeat_probs = decoder(
            sample_data['hidden_states'],
            sample_data['input_ids']
        )
        
        assert repeat_probs.shape == (2, model_config.vocab_size)
        assert torch.all(repeat_probs >= 0)
        assert torch.all(repeat_probs <= 1)
    
    def test_repeat_decoder_with_temporal(self, equity_config, device, sample_data):
        decoder = RepeatDecoder(equity_config, device)
        
        repeat_probs = decoder(
            sample_data['hidden_states'],
            sample_data['input_ids'],
            sample_data['time_gaps'],
            sample_data['categories']
        )
        
        assert repeat_probs.shape == (2, equity_config.vocab_size)


class TestExploreDecoder:
    """Test explore behavior decoder."""
    
    def test_explore_decoder_init(self, model_config, device):
        decoder = ExploreDecoder(model_config, device)
        
        assert isinstance(decoder.attention, AdditiveAttention)
        assert isinstance(decoder.explore_classifier, nn.Linear)
        assert decoder.explore_classifier.out_features == model_config.vocab_size
    
    def test_explore_decoder_forward(self, model_config, device, sample_data):
        decoder = ExploreDecoder(model_config, device)
        
        explore_probs = decoder(
            sample_data['hidden_states'],
            sample_data['input_ids']
        )
        
        assert explore_probs.shape == (2, model_config.vocab_size)
        assert torch.allclose(explore_probs.sum(dim=1), torch.ones(2), atol=1e-6)


class TestUserBehaviorModule:
    """Test complete user behavior module."""
    
    def test_user_behavior_module_init(self, model_config, device):
        module = UserBehaviorModule(model_config, device)
        
        assert isinstance(module.behavior_predictor, BehaviorPredictor)
        assert isinstance(module.repeat_decoder, RepeatDecoder)
        assert isinstance(module.explore_decoder, ExploreDecoder)
    
    def test_user_behavior_module_forward(self, model_config, device, sample_data):
        module = UserBehaviorModule(model_config, device)
        
        output = module(
            sample_data['hidden_states'],
            sample_data['input_ids']
        )
        
        assert isinstance(output, UserBehaviorOutput)
        assert output.repeat_probs.shape == (2, model_config.vocab_size)
        assert output.explore_probs.shape == (2, model_config.vocab_size)
        assert output.behavior_probs.shape == (2, 2)
    
    def test_user_behavior_module_with_temporal(self, equity_config, device, sample_data):
        module = UserBehaviorModule(equity_config, device)
        
        output = module(
            sample_data['hidden_states'],
            sample_data['input_ids'],
            sample_data['time_gaps'],
            sample_data['categories'],
            phase='train'
        )
        
        assert isinstance(output, UserBehaviorOutput)
        assert output.repeat_probs.shape == (2, equity_config.vocab_size)
        assert output.explore_probs.shape == (2, equity_config.vocab_size)
        assert output.behavior_probs.shape == (2, 2)


class TestUserBehaviorOutput:
    """Test output dataclass."""
    
    def test_output_creation(self):
        repeat_probs = torch.randn(2, 100)
        explore_probs = torch.randn(2, 100)
        behavior_probs = torch.randn(2, 2)
        
        output = UserBehaviorOutput(
            repeat_probs=repeat_probs,
            explore_probs=explore_probs,
            behavior_probs=behavior_probs
        )
        
        assert torch.equal(output.repeat_probs, repeat_probs)
        assert torch.equal(output.explore_probs, explore_probs)
        assert torch.equal(output.behavior_probs, behavior_probs)


class TestIntegration:
    """Integration tests for user behavior components."""
    
    def test_end_to_end_pipeline(self, model_config, device, sample_data):
        """Test complete pipeline from hidden states to behavior predictions."""
        module = UserBehaviorModule(model_config, device)
        
        # Forward pass
        output = module(
            sample_data['hidden_states'],
            sample_data['input_ids']
        )
        
        # Verify output structure
        assert isinstance(output, UserBehaviorOutput)
        
        # Verify tensor shapes
        batch_size = sample_data['hidden_states'].shape[0]
        vocab_size = model_config.vocab_size
        
        assert output.repeat_probs.shape == (batch_size, vocab_size)
        assert output.explore_probs.shape == (batch_size, vocab_size)
        assert output.behavior_probs.shape == (batch_size, 2)
        
        # Verify probability constraints
        assert torch.all(output.repeat_probs >= 0)
        assert torch.all(output.explore_probs >= 0)
        assert torch.allclose(output.explore_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_gradient_flow(self, model_config, device, sample_data):
        """Test that gradients flow through the module."""
        module = UserBehaviorModule(model_config, device)
        
        # Enable gradients
        sample_data['hidden_states'].requires_grad_(True)
        
        # Forward pass
        output = module(
            sample_data['hidden_states'],
            sample_data['input_ids']
        )
        
        # Compute loss and backward
        loss = output.repeat_probs.sum() + output.explore_probs.sum() + output.behavior_probs.sum()
        loss.backward()
        
        # Check gradients exist
        assert sample_data['hidden_states'].grad is not None
        assert torch.any(sample_data['hidden_states'].grad != 0)
