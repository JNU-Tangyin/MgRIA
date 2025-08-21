"""Refactored user behavior modules for MgRIA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
import math

from ..config import ModelConfig


@dataclass
class UserBehaviorOutput:
    """Output structure for user behavior prediction."""
    repeat_probs: torch.Tensor
    explore_probs: torch.Tensor
    behavior_probs: torch.Tensor  # [repeat_prob, explore_prob]


def create_onehot_map(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Create one-hot encoding map for given indices.
    
    Args:
        indices: [batch_size, seq_len]
        vocab_size: Size of vocabulary
        
    Returns:
        One-hot tensor [batch_size, seq_len, vocab_size]
    """
    batch_size, seq_len = indices.size()
    device = indices.device
    
    onehot = torch.zeros(batch_size, seq_len, vocab_size, device=device)
    onehot.scatter_(2, indices.unsqueeze(2), 1.0)
    onehot.requires_grad = False
    
    return onehot


class AdditiveAttention(nn.Module):
    """Additive attention mechanism for behavior prediction."""
    
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        interest_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply additive attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            interest_state: [batch_size, 1, hidden_size] or None (uses first token)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Use first token as interest if not provided
        if interest_state is None:
            interest_state = hidden_states[:, 0:1, :]  # [batch_size, 1, hidden_size]
        
        # Project queries and keys
        queries = self.query_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        keys = self.key_proj(interest_state)  # [batch_size, 1, hidden_size]
        
        # Expand keys to match sequence length
        keys = keys.expand(batch_size, seq_len, hidden_size)
        
        # Additive attention
        features = queries + keys
        features = self.activation(features)
        scores = self.value_proj(features).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector
        context = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, hidden_size]
        
        return context, attention_weights


class BehaviorPredictor(nn.Module):
    """Predicts repeat vs explore behavior."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = AdditiveAttention(
            config.dim, 
            config.attention_dropout_prob
        )
        self.behavior_classifier = nn.Linear(config.dim, 2)  # [repeat, explore]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict behavior probabilities.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            Behavior probabilities [batch_size, 2] (repeat, explore)
        """
        context, _ = self.attention(hidden_states)
        behavior_logits = self.behavior_classifier(context)
        return behavior_logits


class MixtureDistribution(nn.Module):
    """Mixture distribution for time interval modeling."""
    
    def __init__(self):
        super().__init__()
        # Gaussian components
        self.gaussian_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(6)
        ])
        self.gaussian_means = nn.ParameterList([
            nn.Parameter(torch.tensor(30.0)),  # mu0
            nn.Parameter(torch.tensor(30.0)),  # mu1_1
            nn.Parameter(torch.tensor(60.0)),  # mu1_2
            nn.Parameter(torch.tensor(30.0)),  # mu2
        ])
        self.gaussian_stds = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(4)
        ])
        
        # Power law components
        self.power_params = nn.ParameterList([
            nn.Parameter(torch.tensor(-0.5)),  # p2
            nn.Parameter(torch.tensor(-1.0)),  # p3
        ])
        
        # Mixture weight
        self.mixture_weight = nn.Parameter(torch.tensor(0.5))
    
    def compute_distribution(self, time_gaps: torch.Tensor, mode: int) -> torch.Tensor:
        """
        Compute mixture distribution probability.
        
        Args:
            time_gaps: [batch_size, seq_len]
            mode: Distribution mode (0-3)
            
        Returns:
            Probability scores [batch_size, seq_len]
        """
        if mode == 0:  # Single Gaussian
            dist = torch.distributions.Normal(
                self.gaussian_means[0], 
                self.gaussian_stds[0]
            )
            return self.gaussian_weights[0] * dist.log_prob(time_gaps).exp()
        
        elif mode == 1:  # Double Gaussian
            dist1 = torch.distributions.Normal(
                self.gaussian_means[1], 
                self.gaussian_stds[1]
            )
            dist2 = torch.distributions.Normal(
                self.gaussian_means[2], 
                self.gaussian_stds[2]
            )
            return (self.gaussian_weights[1] * dist1.log_prob(time_gaps).exp() + 
                    self.gaussian_weights[2] * dist2.log_prob(time_gaps).exp())
        
        elif mode == 2:  # Gaussian + Power law
            dist = torch.distributions.Normal(
                self.gaussian_means[3], 
                self.gaussian_stds[3]
            )
            gaussian_part = self.gaussian_weights[3] * dist.log_prob(time_gaps).exp()
            power_part = self.gaussian_weights[4] * torch.pow(time_gaps, self.power_params[0])
            return gaussian_part + power_part
        
        elif mode == 3:  # Pure power law
            return self.gaussian_weights[5] * torch.pow(time_gaps, self.power_params[1])
        
        else:
            raise ValueError(f"Unknown distribution mode: {mode}")


class RepeatDecoder(nn.Module):
    """Decoder for repeat behavior prediction."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.vocab_size = config.vocab_size
        
        # Special token IDs
        special_tokens = config.model_params.get('special_tokens', {})
        self.interest_id = special_tokens.get('interest_id', config.vocab_size - 1)
        self.pad_id = special_tokens.get('pad_id', config.vocab_size - 2)
        
        # Attention mechanism
        self.attention = AdditiveAttention(
            config.dim, 
            config.attention_dropout_prob
        )
        
        # Mixture distribution for time modeling
        dataset_name = config.model_params.get('dataset_name', 'tafeng')
        if dataset_name == 'equity':
            self.mixture_dist = MixtureDistribution()
        else:
            self.mixture_dist = None
        
        # Projection flag
        self.use_projection = getattr(config, 'repeat_proj', False)
        if self.use_projection:
            self.query_proj = nn.Linear(config.dim, config.dim)
            self.key_proj = nn.Linear(config.dim, config.dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        phase: str = 'train'
    ) -> torch.Tensor:
        """
        Forward pass for repeat decoder.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            time_gaps: [batch_size, seq_len] (for equity dataset)
            categories: [batch_size, seq_len] (for equity dataset)
            phase: 'train' or 'test'
            
        Returns:
            Repeat probabilities [batch_size, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask (exclude padding and interest tokens)
        mask = (input_ids != self.pad_id) & (input_ids != self.interest_id)
        
        # No dense one-hot: we will use scatter_add over vocab dimension
        
        # Apply projection if enabled
        if self.use_projection:
            queries = self.query_proj(hidden_states)
            keys = self.key_proj(hidden_states[:, 0:1, :])
        else:
            queries = hidden_states
            keys = hidden_states[:, 0:1, :]
        
        # Compute attention
        keys_expanded = keys.expand(batch_size, seq_len, -1)
        features = queries + keys_expanded
        features = torch.tanh(features)
        
        # Attention scores
        attention_proj = nn.Linear(self.config.dim, 1).to(self.device)
        scores = attention_proj(features).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask and softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute repeat probabilities via scatter_add over items in the sequence
        # attention_weights: [B, L], input_ids: [B, L]
        repeat_probs = torch.zeros(batch_size, self.vocab_size, device=self.device)
        repeat_probs.scatter_add_(1, input_ids, attention_weights)
        
        # Add temporal modeling for equity dataset
        if self.mixture_dist is not None and time_gaps is not None and categories is not None:
            repeat_probs = self._add_temporal_modeling(
                repeat_probs, input_ids, time_gaps, categories, phase
            )
        
        return repeat_probs
    
    def _add_temporal_modeling(
        self,
        repeat_probs: torch.Tensor,
        input_ids: torch.Tensor,
        time_gaps: torch.Tensor,
        categories: torch.Tensor,
        phase: str
    ) -> torch.Tensor:
        """Add temporal distribution modeling for equity dataset."""
        # Use sequence without first token for temporal modeling
        input_ids_shifted = input_ids[:, 1:]
        time_gaps_shifted = time_gaps[:, 1:]
        categories_shifted = categories[:, 1:]
        
        # Category-specific temporal modeling
        category_probs = []
        for cat_id in range(4):  # Categories 0-3
            cat_mask = categories_shifted != cat_id
            
            # Mask time gaps for non-matching categories
            pad_value = 180  # Default padding value
            masked_gaps = time_gaps_shifted.masked_fill(cat_mask, pad_value)
            
            # Compute distribution probabilities
            if cat_id == 0:
                dist_probs = self.mixture_dist.compute_distribution(masked_gaps, 0)
            elif cat_id == 1:
                dist_probs = self.mixture_dist.compute_distribution(masked_gaps, 1)
            elif cat_id == 2:
                dist_probs = self.mixture_dist.compute_distribution(masked_gaps, 2)
            else:  # cat_id == 3
                continue  # Skip category 3
            
            # Apply to vocabulary via scatter_add
            cat_probs = torch.zeros(repeat_probs.size(0), self.vocab_size, device=repeat_probs.device)
            cat_probs.scatter_add_(1, input_ids_shifted, dist_probs)
            category_probs.append(cat_probs)
        
        # Combine temporal probabilities
        if category_probs:
            temporal_probs = sum(category_probs)
            # Mix with original repeat probabilities
            repeat_probs = ((1 - self.mixture_dist.mixture_weight) * repeat_probs + 
                           self.mixture_dist.mixture_weight * temporal_probs)
        
        return repeat_probs


class ExploreDecoder(nn.Module):
    """Decoder for explore behavior prediction."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.vocab_size = config.vocab_size
        
        # Special token IDs
        special_tokens = config.model_params.get('special_tokens', {})
        self.interest_id = special_tokens.get('interest_id', config.vocab_size - 1)
        self.pad_id = special_tokens.get('pad_id', config.vocab_size - 2)
        
        # Attention mechanism
        self.attention = AdditiveAttention(
            config.dim, 
            config.attention_dropout_prob
        )
        
        # Exploration classifier
        self.explore_classifier = nn.Linear(2 * config.dim, config.vocab_size)
        
        # Projection flag
        self.use_projection = getattr(config, 'explore_proj', False)
        if self.use_projection:
            self.query_proj = nn.Linear(config.dim, config.dim)
            self.key_proj = nn.Linear(config.dim, config.dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for explore decoder.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            
        Returns:
            Explore probabilities [batch_size, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        mask = (input_ids != self.pad_id) & (input_ids != self.interest_id)
        
        # No dense one-hot here; we will derive masks via scatter
        
        # Apply projection if enabled
        if self.use_projection:
            queries = self.query_proj(hidden_states)
            keys = self.key_proj(hidden_states[:, 0:1, :])
        else:
            queries = hidden_states
            keys = hidden_states[:, 0:1, :]
        
        # Compute attention context
        context, attention_weights = self.attention(queries, keys)
        
        # Get interest representation
        interest_repr = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # Concatenate context and interest
        combined_repr = torch.cat([interest_repr, context], dim=-1)
        
        # Predict exploration probabilities
        explore_logits = self.explore_classifier(combined_repr)
        
        # Mask out items that appeared in sequence using scatter on mask
        # counts[b, item] = number of appearances of item in (non-pad) positions
        counts = torch.zeros(batch_size, self.vocab_size, device=self.device)
        counts.scatter_add_(1, input_ids, mask.float())
        explore_mask = counts > 0
        
        # Apply mask and softmax
        explore_logits = explore_logits.masked_fill(explore_mask.bool(), float('-inf'))
        explore_probs = F.softmax(explore_logits, dim=-1)
        
        return explore_probs


class UserBehaviorModule(nn.Module):
    """Complete user behavior prediction module."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.behavior_predictor = BehaviorPredictor(config)
        self.repeat_decoder = RepeatDecoder(config, device)
        self.explore_decoder = ExploreDecoder(config, device)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        time_gaps: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        phase: str = 'train'
    ) -> UserBehaviorOutput:
        """
        Forward pass for user behavior module.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            time_gaps: [batch_size, seq_len] (optional)
            categories: [batch_size, seq_len] (optional)
            phase: 'train' or 'test'
            
        Returns:
            UserBehaviorOutput with repeat/explore probabilities
        """
        # Predict behavior type (repeat vs explore)
        behavior_logits = self.behavior_predictor(hidden_states)
        
        # Get repeat and explore probabilities
        repeat_probs = self.repeat_decoder(
            hidden_states, input_ids, time_gaps, categories, phase
        )
        explore_probs = self.explore_decoder(hidden_states, input_ids)
        
        return UserBehaviorOutput(
            repeat_probs=repeat_probs,
            explore_probs=explore_probs,
            behavior_probs=behavior_logits
        )
