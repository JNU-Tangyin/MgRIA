"""Output layer and model integration for MgRIA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from dataclasses import dataclass

from .transformer import Transformer, TransformerOutput
from .user_behavior import UserBehaviorModule, UserBehaviorOutput
from ..config import ModelConfig


@dataclass
class MgRIAOutput:
    """Output structure for MgRIA model."""
    logits: torch.Tensor  # Final prediction logits [batch_size, num_predictions, vocab_size]
    transformer_output: TransformerOutput
    behavior_output: UserBehaviorOutput
    fusion_weight: torch.Tensor  # Fusion weight between transformer and behavior predictions


def gelu_activation(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function implementation."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    """Layer normalization module."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.dim))
        self.bias = nn.Parameter(torch.zeros(config.dim))
        self.eps = config.layer_norm_eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class OutputProjection(nn.Module):
    """Output projection layer for BERT-style predictions."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.dim, config.dim)
        self.norm = LayerNorm(config)
        self.decoder = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using config settings."""
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder_bias)
    
    def forward(self, hidden_states: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits for masked positions.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            masked_positions: [batch_size, num_predictions] - indices of masked positions
            
        Returns:
            Logits [batch_size, num_predictions, vocab_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_predictions = masked_positions.shape[1]
        
        # Expand masked positions to gather hidden states
        masked_pos_expanded = masked_positions[:, :, None].expand(-1, -1, hidden_size)
        
        # Gather hidden states at masked positions
        h_masked = torch.gather(hidden_states, 1, masked_pos_expanded)  # [batch_size, num_predictions, hidden_size]
        
        # Apply projection layers
        h_projected = self.linear(h_masked)
        h_projected = gelu_activation(h_projected)
        h_projected = self.norm(h_projected)
        
        # Generate logits
        logits = self.decoder(h_projected) + self.decoder_bias
        
        return logits


class MgRIAModel(nn.Module):
    """Complete MgRIA model with Transformer, user behavior modeling, and output fusion."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Core components
        self.transformer = Transformer(config, device)
        self.user_behavior = UserBehaviorModule(config, device)
        self.output_projection = OutputProjection(config)
        
        # Fusion parameter - learnable weight for combining predictions
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        phase: str = 'train'
    ) -> MgRIAOutput:
        """
        Forward pass of MgRIA model.
        
        Args:
            input_ids: [batch_size, seq_len] - input item sequences
            time_features: [batch_size, seq_len, time_dim] - temporal features
            attention_mask: [batch_size, seq_len] - attention mask
            masked_positions: [batch_size, num_predictions] - positions to predict
            time_matrix: [batch_size, seq_len, seq_len] - optional time interaction matrix
            time_gaps: [batch_size, seq_len] - time gaps between items (for equity dataset)
            categories: [batch_size, seq_len] - item categories (for equity dataset)
            phase: 'train' or 'test'
            
        Returns:
            MgRIAOutput with final predictions and intermediate outputs
        """
        # Transformer encoding
        transformer_output = self.transformer(
            input_ids=input_ids,
            time_features=time_features,
            attention_mask=attention_mask,
            time_gaps=time_matrix
        )
        
        hidden_states = transformer_output.hidden_states
        
        # User behavior prediction
        behavior_output = self.user_behavior(
            hidden_states=hidden_states,
            input_ids=input_ids,
            time_gaps=time_gaps,
            categories=categories,
            phase=phase
        )
        
        # Generate behavior-based predictions
        behavior_probs = F.softmax(behavior_output.behavior_probs, dim=-1)  # [batch_size, 2]
        repeat_weight = behavior_probs[:, 0].unsqueeze(-1)  # [batch_size, 1]
        explore_weight = behavior_probs[:, 1].unsqueeze(-1)  # [batch_size, 1]
        
        # Combine repeat and explore predictions
        behavior_logits = (repeat_weight * behavior_output.repeat_probs + 
                          explore_weight * behavior_output.explore_probs)  # [batch_size, vocab_size]
        behavior_logits = behavior_logits.unsqueeze(1)  # [batch_size, 1, vocab_size]
        
        # Expand to match number of predictions
        num_predictions = masked_positions.shape[1]
        behavior_logits = behavior_logits.expand(-1, num_predictions, -1)  # [batch_size, num_predictions, vocab_size]
        
        # Transformer-based predictions for masked positions
        transformer_logits = self.output_projection(hidden_states, masked_positions)
        
        # Fuse predictions using learnable weight
        final_logits = (self.fusion_weight * transformer_logits + 
                       (1 - self.fusion_weight) * behavior_logits)
        
        return MgRIAOutput(
            logits=final_logits,
            transformer_output=transformer_output,
            behavior_output=behavior_output,
            fusion_weight=self.fusion_weight
        )
    
    def predict(
        self,
        input_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: torch.Tensor,
        time_matrix: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
        categories: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for next items.
        
        Args:
            input_ids: [batch_size, seq_len]
            time_features: [batch_size, seq_len, time_dim]
            attention_mask: [batch_size, seq_len]
            time_matrix: Optional time interaction matrix
            time_gaps: Optional time gaps (for equity dataset)
            categories: Optional categories (for equity dataset)
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
        """
        self.eval()
        
        with torch.no_grad():
            # Use last position as prediction target
            batch_size, seq_len = input_ids.shape
            masked_positions = torch.tensor([[seq_len - 1]] * batch_size, device=self.device)
            
            # Forward pass
            output = self.forward(
                input_ids=input_ids,
                time_features=time_features,
                attention_mask=attention_mask,
                masked_positions=masked_positions,
                time_matrix=time_matrix,
                time_gaps=time_gaps,
                categories=categories,
                phase='test'
            )
            
            # Get predictions
            logits = output.logits.squeeze(1)  # [batch_size, vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            return top_k_indices, top_k_probs
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MgRIALoss(nn.Module):
    """Loss function for MgRIA model."""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        behavior_probs: Optional[torch.Tensor] = None,
        behavior_targets: Optional[torch.Tensor] = None,
        behavior_loss_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute loss for MgRIA model.
        
        Args:
            logits: [batch_size, num_predictions, vocab_size]
            targets: [batch_size, num_predictions]
            behavior_probs: [batch_size, 2] - repeat/explore probabilities
            behavior_targets: [batch_size] - behavior labels (0=repeat, 1=explore)
            behavior_loss_weight: Weight for behavior prediction loss
            
        Returns:
            Combined loss
        """
        # Reshape for cross entropy
        batch_size, num_predictions, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Main prediction loss
        main_loss = self.cross_entropy(logits_flat, targets_flat)
        
        # Behavior prediction loss (optional)
        if behavior_probs is not None and behavior_targets is not None:
            behavior_loss = self.cross_entropy(behavior_probs, behavior_targets)
            total_loss = main_loss + behavior_loss_weight * behavior_loss
        else:
            total_loss = main_loss
        
        return total_loss
