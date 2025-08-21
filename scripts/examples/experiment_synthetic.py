"""Experiment script for testing the refactored MgRIA model."""

import torch
import numpy as np
from pathlib import Path
import yaml
import logging

from src.config import ModelConfig, DatasetConfig, TrainingConfig
from src.models import ModelFactory, MgRIAModel, MgRIALoss
from src.training import MgRIATrainer
from src.utils.helpers import set_seed, setup_logging


def create_synthetic_data(config: ModelConfig, num_samples: int = 1000):
    """Create synthetic data for testing."""
    batch_size = 32
    seq_len = config.max_len
    
    # Generate synthetic sequences
    input_ids = torch.randint(0, config.vocab_size - 10, (num_samples, seq_len))
    
    # Generate time features [month, day, weekday]
    time_features = torch.stack([
        torch.randint(1, 12, (num_samples, seq_len)),  # month: 1-11
        torch.randint(1, 31, (num_samples, seq_len)),  # day: 1-30
        torch.randint(0, 7, (num_samples, seq_len))    # weekday: 0-6
    ], dim=-1)
    
    # Generate attention masks
    attention_mask = torch.ones(num_samples, seq_len)
    
    # Generate masked positions (2-3 positions per sequence)
    num_predictions = 3
    masked_positions = torch.stack([
        torch.randint(0, seq_len, (num_samples,)) for _ in range(num_predictions)
    ], dim=1)
    
    # Generate targets
    targets = torch.randint(0, config.vocab_size, (num_samples, num_predictions))
    
    return {
        'input_ids': input_ids,
        'time_features': time_features,
        'attention_mask': attention_mask,
        'masked_positions': masked_positions,
        'targets': targets
    }


def create_data_loader(data, batch_size=32):
    """Create a simple data loader from synthetic data."""
    dataset = torch.utils.data.TensorDataset(
        data['input_ids'],
        data['time_features'],
        data['attention_mask'],
        data['masked_positions'],
        data['targets']
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )


def run_experiment():
    """Run a simple experiment with the refactored MgRIA model."""
    # Load configuration
    config_path = Path("configs/default.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configurations
    model_config = ModelConfig(
        vocab_size=config_dict['model']['vocab_size'],
        dim=config_dict['model']['dim'],
        n_layers=config_dict['model']['n_layers'],
        n_heads=config_dict['model']['n_heads'],
        max_len=config_dict['model']['max_len'],
        dropout_prob=config_dict['model']['dropout_prob']
    )
    
    training_config = TrainingConfig(
        learning_rate=float(config_dict['training']['learning_rate']),
        weight_decay=float(config_dict['training']['weight_decay']),
        max_epochs=5,  # Short experiment
        batch_size=int(config_dict['training']['batch_size'])
    )
    
    # Set seed for reproducibility
    set_seed(config_dict['seed'])
    
    # Setup logging
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting MgRIA experiment with refactored architecture")
    logger.info(f"Model config: vocab_size={model_config.vocab_size}, dim={model_config.dim}")
    
    # Create model using factory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create_model(model_config, device)
    
    # Get model info
    model_info = ModelFactory.get_model_info(model)
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Create synthetic data
    logger.info("Generating synthetic training data...")
    train_data = create_synthetic_data(model_config, num_samples=1000)
    val_data = create_synthetic_data(model_config, num_samples=200)
    
    # Create data loaders
    train_loader = create_data_loader(train_data, batch_size=32)
    val_loader = create_data_loader(val_data, batch_size=32)
    
    logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Test forward pass
    logger.info("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        batch_dict = {
            'input_ids': sample_batch[0].to(device),
            'time_features': sample_batch[1].to(device),
            'attention_mask': sample_batch[2].to(device),
            'masked_positions': sample_batch[3].to(device)
        }
        
        output = model(**batch_dict)
        logger.info(f"Forward pass successful! Output shape: {output.logits.shape}")
        logger.info(f"Fusion weight: {output.fusion_weight.item():.4f}")
    
    # Test loss computation
    logger.info("Testing loss computation...")
    loss_fn = ModelFactory.create_loss_function()
    targets = sample_batch[4].to(device)
    loss = loss_fn(output.logits, targets)
    logger.info(f"Loss computation successful! Loss: {loss.item():.4f}")
    
    # Test prediction
    logger.info("Testing prediction...")
    items, scores = model.predict(
        input_ids=batch_dict['input_ids'][:2],  # Test with 2 samples
        time_features=batch_dict['time_features'][:2],
        attention_mask=batch_dict['attention_mask'][:2],
        top_k=10
    )
    logger.info(f"Prediction successful! Items shape: {items.shape}, Scores shape: {scores.shape}")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MgRIATrainer(model_config, training_config, device)
    
    # Run short training
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['eval_loss'][-1]:.4f}")
    
    # Test model ensemble
    logger.info("Testing model ensemble...")
    configs = [model_config] * 2  # Create ensemble of 2 models
    ensemble = ModelFactory.create_ensemble(configs, device)
    
    ensemble_items, ensemble_scores = ensemble.predict(
        input_ids=batch_dict['input_ids'][:2],
        time_features=batch_dict['time_features'][:2],
        attention_mask=batch_dict['attention_mask'][:2],
        top_k=5
    )
    logger.info(f"Ensemble prediction successful! Items: {ensemble_items.shape}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"✅ Model creation: {model_info['total_parameters']:,} parameters")
    logger.info(f"✅ Forward pass: output shape {output.logits.shape}")
    logger.info(f"✅ Loss computation: {loss.item():.4f}")
    logger.info(f"✅ Prediction: top-{items.shape[1]} items")
    logger.info(f"✅ Training: {len(history['train_loss'])} epochs")
    logger.info(f"✅ Ensemble: {ensemble.num_models} models")
    logger.info("✅ All components working correctly!")
    
    return {
        'model_info': model_info,
        'training_history': history,
        'final_loss': loss.item(),
        'ensemble_size': ensemble.num_models
    }


if __name__ == "__main__":
    try:
        results = run_experiment()
        print("\n🎉 Experiment completed successfully!")
        print(f"Model has {results['model_info']['total_parameters']:,} parameters")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Training epochs: {len(results['training_history']['train_loss'])}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise
