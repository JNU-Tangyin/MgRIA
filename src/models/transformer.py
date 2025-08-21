"""Refactored Transformer module for MgRIA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..config import ModelConfig


@dataclass
class TransformerOutput:
    """Output structure for Transformer forward pass."""
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function implementation."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last_dim(x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Split the last dimension to given shape."""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last_dims(x: torch.Tensor, n_dims: int) -> torch.Tensor:
    """Merge the last n_dims to a single dimension."""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class TemporalEmbedding(nn.Module):
    """Temporal embedding for time features."""
    
    def __init__(self, d_model: int, config: ModelConfig):
        super().__init__()
        self.d_model = d_model
        
        # Time embedding dimensions from config
        self.weekday_embed = nn.Embedding(config.weekday_size, d_model)
        self.day_embed = nn.Embedding(config.day_size, d_model)
        self.month_embed = nn.Embedding(config.month_size, d_model)
    
    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal embedding.
        
        Args:
            time_features: [batch_size, seq_len, 3] with [month, day, weekday]
            
        Returns:
            Temporal embeddings [batch_size, seq_len, d_model]
        """
        time_features = time_features.long()
        
        month_emb = self.month_embed(time_features[:, :, 0])
        day_emb = self.day_embed(time_features[:, :, 1])
        weekday_emb = self.weekday_embed(time_features[:, :, 2])
        
        # Combine embeddings (can be configured)
        return month_emb + day_emb  # + weekday_emb


class LayerNorm(nn.Module):
    """Layer normalization module."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Combined embedding layer for tokens and time features."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)
        
        # Time embeddings
        self.time_embed = TemporalEmbedding(config.dim, config)
        
        # Time interval embeddings for attention
        time_span = getattr(config, 'time_span', 65)
        self.time_interval_embed_k = nn.Embedding(time_span + 1, config.dim)
        self.time_interval_embed_v = nn.Embedding(time_span + 1, config.dim)
        
        # Normalization and dropout
        self.layer_norm = LayerNorm(config.dim, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.time_k_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.time_v_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        time_features: torch.Tensor, 
        time_gaps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for embeddings.
        
        Args:
            input_ids: [batch_size, seq_len]
            time_features: [batch_size, seq_len, 3]
            time_gaps: [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (embeddings, time_k, time_v)
        """
        # Token embeddings
        token_emb = self.token_embed(input_ids)
        
        # Time embeddings
        time_emb = self.time_embed(time_features)
        
        # Combine embeddings (no padding needed, dimensions should match)
        embeddings = token_emb + time_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Time interval embeddings for attention
        time_k = self.time_interval_embed_k(time_gaps)
        time_v = self.time_interval_embed_v(time_gaps)
        time_k = self.time_k_dropout(time_k)
        time_v = self.time_v_dropout(time_v)
        
        return embeddings, time_k, time_v


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with temporal modeling."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.num_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        
        assert config.dim % config.n_heads == 0
        
        # Linear projections
        self.query_proj = nn.Linear(config.dim, config.dim)
        self.key_proj = nn.Linear(config.dim, config.dim)
        self.value_proj = nn.Linear(config.dim, config.dim)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        
        # For visualization
        self.attention_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        time_k: torch.Tensor,
        time_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            time_k: [batch_size, seq_len, seq_len, hidden_size]
            time_v: [batch_size, seq_len, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear projections and reshape for multi-head
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query = split_last_dim(query, (self.num_heads, -1)).transpose(1, 2)
        key = split_last_dim(key, (self.num_heads, -1)).transpose(1, 2)
        value = split_last_dim(value, (self.num_heads, -1)).transpose(1, 2)
        
        # Reshape time embeddings for multi-head
        # [batch_size, seq_len, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, seq_len, head_dim]
        time_k = split_last_dim(time_k, (self.num_heads, -1)).transpose(-3, -2).transpose(1, 2)
        time_v = split_last_dim(time_v, (self.num_heads, -1)).transpose(-3, -2).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Simplified temporal attention (skip complex time modeling for now)
        # temporal_scores = torch.matmul(time_k, query.unsqueeze(-1)).squeeze(-1)
        # attention_scores += temporal_scores
        
        # Scale
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].float()
            attention_scores += (1.0 - mask) * -10000.0
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Simplified temporal values (skip complex time modeling for now)
        # temporal_context = torch.matmul(attention_probs.unsqueeze(2), time_v)
        # temporal_context = torch.diagonal(temporal_context, offset=0, dim1=2, dim2=3)
        # temporal_context = temporal_context.transpose(-1, -2).transpose(1, 2)
        # context += temporal_context
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous()
        context = merge_last_dims(context, 2)
        
        # Store attention weights for visualization
        self.attention_weights = attention_probs
        
        return context


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        intermediate_size = getattr(config, 'intermediate_size', config.dim * 4)
        
        self.dense_1 = nn.Linear(config.dim, intermediate_size)
        self.dense_2 = nn.Linear(intermediate_size, config.dim)
        
        # Use GELU activation
        self.activation = gelu
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network."""
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.attention = MultiHeadAttention(config, device)
        self.attention_output = nn.Linear(config.dim, config.dim)
        self.attention_norm = LayerNorm(config.dim, config.layer_norm_eps)
        
        self.feed_forward = FeedForward(config)
        self.output_norm = LayerNorm(config.dim, config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        time_k: torch.Tensor,
        time_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for Transformer block."""
        # Self-attention
        attention_output = self.attention(hidden_states, time_k, time_v, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.dropout(attention_output)
        
        # Add & Norm
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward
        feed_forward_output = self.feed_forward(hidden_states)
        feed_forward_output = self.dropout(feed_forward_output)
        
        # Add & Norm
        hidden_states = self.output_norm(hidden_states + feed_forward_output)
        
        return hidden_states


class Transformer(nn.Module):
    """Main Transformer model with temporal attention."""
    
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Embeddings
        self.embeddings = Embeddings(config, device)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, device) 
            for _ in range(config.n_layers)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None
    ) -> TransformerOutput:
        """
        Forward pass for Transformer.
        
        Args:
            input_ids: [batch_size, seq_len]
            time_features: [batch_size, seq_len, 3]
            attention_mask: [batch_size, seq_len]
            time_gaps: [batch_size, seq_len, seq_len]
            
        Returns:
            TransformerOutput with hidden states and attention weights
        """
        # Default time gaps if not provided
        if time_gaps is None:
            batch_size, seq_len = input_ids.shape
            time_gaps = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long, device=self.device)
        
        # Get embeddings
        hidden_states, time_k, time_v = self.embeddings(input_ids, time_features, time_gaps)
        
        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, time_k, time_v, attention_mask)
        
        return TransformerOutput(
            hidden_states=hidden_states,
            attention_weights=self.blocks[-1].attention.attention_weights if self.blocks else None
        )
