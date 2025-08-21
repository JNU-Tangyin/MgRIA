"""Model factory for creating MgRIA models with different configurations."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..config import ModelConfig, DatasetConfig, TrainingConfig
from .output import MgRIAModel, MgRIALoss
from ..utils.helpers import get_device, count_parameters


class ModelFactory:
    """Factory class for creating and managing MgRIA models."""
    
    @staticmethod
    def create_model(
        config: ModelConfig,
        device: Optional[torch.device] = None,
        dataset_config: Optional[DatasetConfig] = None
    ) -> MgRIAModel:
        """
        Create a MgRIA model with the given configuration.
        
        Args:
            config: Model configuration
            device: Target device (auto-detected if None)
            dataset_config: Dataset configuration for vocab size adjustment
            
        Returns:
            Initialized MgRIA model
        """
        if device is None:
            device = get_device()
        
        # Add dataset-specific parameters
        if dataset_config is not None:
            config.model_params['dataset_name'] = dataset_config.dataset_type.value
        
        # Create model
        model = MgRIAModel(config, device)
        
        return model
    
    @staticmethod
    def create_loss_function(
        ignore_index: int = -100,
        behavior_loss_weight: float = 0.1
    ) -> MgRIALoss:
        """
        Create loss function for MgRIA model.
        
        Args:
            ignore_index: Index to ignore in loss computation
            behavior_loss_weight: Weight for behavior prediction loss
            
        Returns:
            MgRIA loss function
        """
        return MgRIALoss(ignore_index=ignore_index)
    
    @staticmethod
    def load_model(
        checkpoint_path: Union[str, Path],
        config: ModelConfig,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> MgRIAModel:
        """
        Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            device: Target device (auto-detected if None)
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Loaded MgRIA model
        """
        if device is None:
            device = get_device()
        
        # Create model
        model = MgRIAModel(config, device)
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
        
        return model
    
    @staticmethod
    def save_model(
        model: MgRIAModel,
        save_path: Union[str, Path],
        config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            save_path: Path to save checkpoint
            config: Model configuration to save
            training_config: Training configuration to save
            epoch: Current epoch number
            metrics: Training metrics to save
            optimizer_state: Optimizer state dict to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_size': model.get_model_size(),
            'trainable_params': model.get_trainable_parameters()
        }
        
        # Add optional data
        if config is not None:
            checkpoint['model_config'] = config.to_dict()
        
        if training_config is not None:
            checkpoint['training_config'] = training_config.to_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
    
    @staticmethod
    def get_model_info(model: MgRIAModel) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model: MgRIA model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'total_parameters': model.get_model_size(),
            'trainable_parameters': model.get_trainable_parameters(),
            'device': str(model.device),
            'fusion_weight': model.fusion_weight.item(),
            'components': {}
        }
        
        # Component parameter counts
        info['components']['transformer'] = count_parameters(model.transformer)
        info['components']['user_behavior'] = count_parameters(model.user_behavior)
        info['components']['output_projection'] = count_parameters(model.output_projection)
        
        # Model architecture details
        config = model.config
        info['architecture'] = {
            'vocab_size': config.vocab_size,
            'hidden_size': config.dim,
            'num_layers': config.n_layers,
            'num_heads': config.n_heads,
            'max_length': config.max_len,
            'dropout': config.dropout_prob
        }
        
        return info
    
    @staticmethod
    def create_model_for_dataset(
        dataset_name: str,
        model_config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None
    ) -> MgRIAModel:
        """
        Create a model configured for a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('equity', 'tafeng', 'taobao')
            model_config: Base model configuration (uses default if None)
            device: Target device (auto-detected if None)
            
        Returns:
            Model configured for the dataset
        """
        if device is None:
            device = get_device()
        
        if model_config is None:
            model_config = ModelConfig()
        
        # Dataset-specific configurations
        dataset_configs = {
            'equity': {
                'vocab_size': 259,
                'special_tokens': {'pad_id': 257, 'mask_id': 256, 'interest_id': 258}
            },
            'tafeng': {
                'vocab_size': 15789,
                'special_tokens': {'pad_id': 15786, 'mask_id': 15787, 'interest_id': 15788}
            },
            'taobao': {
                'vocab_size': 287008,
                'special_tokens': {'pad_id': 287005, 'mask_id': 287006, 'interest_id': 287007}
            }
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_configs.keys())}")
        
        # Update configuration
        dataset_config = dataset_configs[dataset_name]
        model_config.update_vocab_size(dataset_config['vocab_size'])
        model_config.model_params['dataset_name'] = dataset_name
        model_config.model_params['special_tokens'] = dataset_config['special_tokens']
        
        # Create model
        model = MgRIAModel(model_config, device)
        
        return model
    
    @staticmethod
    def create_ensemble(
        configs: list[ModelConfig],
        device: Optional[torch.device] = None
    ) -> 'ModelEnsemble':
        """
        Create an ensemble of MgRIA models.
        
        Args:
            configs: List of model configurations
            device: Target device (auto-detected if None)
            
        Returns:
            Model ensemble
        """
        if device is None:
            device = get_device()
        
        models = [ModelFactory.create_model(config, device) for config in configs]
        return ModelEnsemble(models)


class ModelEnsemble(nn.Module):
    """Ensemble of MgRIA models for improved predictions."""
    
    def __init__(self, models: list[MgRIAModel]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, *args, **kwargs):
        """Forward pass through all models and average predictions."""
        outputs = [model(*args, **kwargs) for model in self.models]
        
        # Average logits
        avg_logits = torch.stack([out.logits for out in outputs]).mean(dim=0)
        
        # Return averaged output (using first model's structure)
        return outputs[0].__class__(
            logits=avg_logits,
            transformer_output=outputs[0].transformer_output,
            behavior_output=outputs[0].behavior_output,
            fusion_weight=outputs[0].fusion_weight
        )
    
    def predict(self, *args, **kwargs):
        """Generate ensemble predictions."""
        predictions = [model.predict(*args, **kwargs) for model in self.models]
        
        # Average prediction scores
        all_items = torch.stack([pred[0] for pred in predictions])  # [num_models, batch_size, top_k]
        all_scores = torch.stack([pred[1] for pred in predictions])  # [num_models, batch_size, top_k]
        
        # Simple averaging (could be improved with weighted voting)
        avg_scores = all_scores.mean(dim=0)
        
        # Use items from first model (could be improved with consensus)
        ensemble_items = all_items[0]
        
        return ensemble_items, avg_scores
    
    def get_model_size(self) -> int:
        """Get total parameters across all models."""
        return sum(model.get_model_size() for model in self.models)
    
    def get_trainable_parameters(self) -> int:
        """Get total trainable parameters across all models."""
        return sum(model.get_trainable_parameters() for model in self.models)
