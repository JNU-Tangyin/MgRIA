"""Training configuration classes."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .base import BaseConfig
from .constants import MetricType


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training parameters."""
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_epochs: int = 200
    patience: int = 10  # Early stopping patience
    
    # Batch processing
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer: str = "adam"
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Learning rate scheduler
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"
    
    # Evaluation settings
    eval_steps: int = 1000
    save_steps: int = 1000
    # Epoch-based checkpoint frequency
    save_every: int = 1
    logging_steps: int = 100
    
    # Metrics to evaluate
    eval_metrics: List[str] = field(default_factory=lambda: ['recall@10', 'mrr@10', 'ndcg@10'])
    
    # Model saving
    save_total_limit: int = 3
    save_best_model: bool = True
    metric_for_best_model: str = "ndcg@10"
    greater_is_better: bool = True
    
    # Reproducibility
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Training options
    fp16: bool = False  # Mixed precision training
    gradient_checkpointing: bool = False
    
    # Loss function parameters
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        self._validate_params()
    
    def _validate_params(self):
        """Validate training parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        if self.max_epochs <= 0:
            raise ValueError(f"Max epochs must be positive, got {self.max_epochs}")
        
        if self.patience <= 0:
            raise ValueError(f"Patience must be positive, got {self.patience}")
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """Get optimizer parameters."""
        return {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'eps': self.adam_epsilon,
            'betas': (self.adam_beta1, self.adam_beta2)
        }
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """Get learning rate scheduler parameters."""
        return {
            'num_warmup_steps': self.warmup_steps,
            'num_training_steps': self.max_epochs * 1000  # Approximate
        }
    
    def is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.greater_is_better:
            return current > best
        else:
            return current < best
