"""Model configuration classes."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .base import BaseConfig
from .constants import ModelType, TIME_EMBEDDING_SIZES


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture parameters."""
    
    # Model type
    model_type: ModelType = ModelType.MGRIA
    
    # Architecture parameters
    vocab_size: int = 15789  # Will be set based on dataset
    dim: int = 64  # Hidden dimension
    n_layers: int = 2  # Number of transformer layers
    n_heads: int = 2  # Number of attention heads
    max_len: int = 50  # Maximum sequence length
    
    # Dropout rates
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    
    # Time embedding dimensions
    weekday_size: int = TIME_EMBEDDING_SIZES['weekday_size']
    day_size: int = TIME_EMBEDDING_SIZES['day_size'] 
    month_size: int = TIME_EMBEDDING_SIZES['month_size']
    
    # Activation function
    hidden_act: str = "gelu"
    
    # Layer normalization epsilon
    layer_norm_eps: float = 1e-12
    
    # Initializer range for weights
    initializer_range: float = 0.02
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        
        # Validate architecture parameters
        self._validate_params()
    
    def _validate_params(self):
        """Validate model parameters."""
        if self.dim % self.n_heads != 0:
            raise ValueError(f"Hidden dimension ({self.dim}) must be divisible by number of heads ({self.n_heads})")
        
        if self.n_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.n_layers}")
        
        if self.vocab_size <= 0:
            raise ValueError(f"Vocabulary size must be positive, got {self.vocab_size}")
    
    @property
    def head_dim(self) -> int:
        """Get dimension per attention head."""
        return self.dim // self.n_heads
    
    def update_vocab_size(self, vocab_size: int) -> None:
        """Update vocabulary size based on dataset."""
        self.vocab_size = vocab_size
        self._validate_params()
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get all model parameters as a dictionary."""
        params = {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'max_len': self.max_len,
            'dropout_prob': self.dropout_prob,
            'attention_dropout_prob': self.attention_dropout_prob,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'weekday_size': self.weekday_size,
            'day_size': self.day_size,
            'month_size': self.month_size,
            'hidden_act': self.hidden_act,
            'layer_norm_eps': self.layer_norm_eps,
            'initializer_range': self.initializer_range
        }
        
        # Add model-specific parameters
        params.update(self.model_params)
        
        return params
