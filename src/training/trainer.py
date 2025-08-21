"""Training module for MgRIA models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

from ..models import ModelFactory, MgRIAModel, MgRIALoss
from ..config import ModelConfig, TrainingConfig
from ..utils.helpers import get_device, setup_logging


class MgRIATrainer:
    """Trainer class for MgRIA models."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            device: Training device (auto-detected if None)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or get_device()
        
        # Initialize model and loss
        self.model = ModelFactory.create_model(model_config, self.device)
        self.loss_fn = ModelFactory.create_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        
        # Setup logging
        setup_logging('INFO')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized trainer with {self.model.get_model_size():,} parameters")
        self.logger.info(f"Training on device: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.training_config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.training_config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.max_epochs
            )
        elif self.training_config.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.max_epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                # Convert list/tuple to dictionary (synthetic pipeline)
                batch_dict = {
                    'input_ids': batch[0].to(self.device),
                    'time_features': batch[1].to(self.device),
                    'attention_mask': batch[2].to(self.device),
                    'masked_positions': batch[3].to(self.device),
                    'targets': batch[4].to(self.device)
                }
            else:
                # Move batch (dict) to device
                batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch.items()}
                # Map alternative keys from real dataset pipeline
                if 'input_ids' not in batch_dict and 'sequences' in batch_dict:
                    batch_dict['input_ids'] = batch_dict['sequences']
                if 'attention_mask' not in batch_dict and 'input_mask' in batch_dict:
                    batch_dict['attention_mask'] = batch_dict['input_mask']
                if 'masked_positions' not in batch_dict and 'masked_pos' in batch_dict:
                    batch_dict['masked_positions'] = batch_dict['masked_pos']
                if 'targets' not in batch_dict and 'masked_ids' in batch_dict:
                    # If masked_ids is a list (from collate), convert to tensor
                    if isinstance(batch_dict['masked_ids'], list):
                        import torch as _torch
                        batch_dict['targets'] = _torch.stack(batch_dict['masked_ids']).to(self.device)
                    else:
                        batch_dict['targets'] = batch_dict['masked_ids']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                input_ids=batch_dict['input_ids'],
                time_features=batch_dict['time_features'],
                attention_mask=batch_dict['attention_mask'],
                masked_positions=batch_dict['masked_positions']
            )
            
            # Compute loss
            loss = self.loss_fn(output.logits, batch_dict['targets'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(eval_loader)
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    # Convert list/tuple to dictionary (synthetic pipeline)
                    batch_dict = {
                        'input_ids': batch[0].to(self.device),
                        'time_features': batch[1].to(self.device),
                        'attention_mask': batch[2].to(self.device),
                        'masked_positions': batch[3].to(self.device),
                        'targets': batch[4].to(self.device)
                    }
                else:
                    # Move batch (dict) to device
                    batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in batch.items()}
                    # Map alternative keys from real dataset pipeline
                    if 'input_ids' not in batch_dict and 'sequences' in batch_dict:
                        batch_dict['input_ids'] = batch_dict['sequences']
                    if 'attention_mask' not in batch_dict and 'input_mask' in batch_dict:
                        batch_dict['attention_mask'] = batch_dict['input_mask']
                    if 'masked_positions' not in batch_dict and 'masked_pos' in batch_dict:
                        batch_dict['masked_positions'] = batch_dict['masked_pos']
                    if 'targets' not in batch_dict and 'masked_ids' in batch_dict:
                        if isinstance(batch_dict['masked_ids'], list):
                            import torch as _torch
                            batch_dict['targets'] = _torch.stack(batch_dict['masked_ids']).to(self.device)
                        else:
                            batch_dict['targets'] = batch_dict['masked_ids']
                
                # Forward pass
                output = self.model(
                    input_ids=batch_dict['input_ids'],
                    time_features=batch_dict['time_features'],
                    attention_mask=batch_dict['attention_mask'],
                    masked_positions=batch_dict['masked_positions']
                )
                
                # Compute loss
                loss = self.loss_fn(output.logits, batch_dict['targets'])
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'eval_loss': avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            eval_loader: Validation data loader (optional)
            save_dir: Directory to save checkpoints (optional)
            
        Returns:
            Training history
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {'train_loss': [], 'eval_loss': []}
        
        self.logger.info(f"Starting training for {self.training_config.max_epochs} epochs")
        
        for epoch in range(self.training_config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Evaluate
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                history['eval_loss'].append(eval_metrics['eval_loss'])
                
                # Check for best model
                current_metric = -eval_metrics['eval_loss']  # Use negative loss as metric
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    
                    if save_dir is not None:
                        self.save_checkpoint(save_dir / 'best_model.pt', is_best=True)
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.training_config.max_epochs}: "
                    f"train_loss={train_metrics['train_loss']:.4f}, "
                    f"eval_loss={eval_metrics['eval_loss']:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.training_config.max_epochs}: "
                    f"train_loss={train_metrics['train_loss']:.4f}"
                )
            
            # Save checkpoint
            if save_dir is not None and (epoch + 1) % self.training_config.save_every == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch + 1}.pt')
        
        self.logger.info("Training completed")
        return history
    
    def save_checkpoint(
        self,
        save_path: Path,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save training checkpoint."""
        checkpoint_info = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict()
        }
        
        if self.scheduler is not None:
            checkpoint_info['scheduler_state'] = self.scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint_info.update(additional_info)
        
        ModelFactory.save_model(
            self.model,
            save_path,
            config=self.model_config,
            training_config=self.training_config,
            epoch=self.current_epoch,
            optimizer_state=checkpoint_info
        )
        
        if is_best:
            self.logger.info(f"Saved best model to {save_path}")
        else:
            self.logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model
        self.model = ModelFactory.load_model(
            checkpoint_path,
            self.model_config,
            self.device
        )
        
        # Load training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if load_scheduler and self.scheduler is not None and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def predict(
        self,
        input_ids: torch.Tensor,
        time_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            time_features: Time features [batch_size, seq_len, 3]
            attention_mask: Attention mask [batch_size, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (predicted_items, prediction_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_ids = input_ids.to(self.device)
            time_features = time_features.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Generate predictions
            items, scores = self.model.predict(
                input_ids=input_ids,
                time_features=time_features,
                attention_mask=attention_mask,
                top_k=top_k
            )
        
        return items, scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return ModelFactory.get_model_info(self.model)
