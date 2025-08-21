"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.config import BaseConfig, DatasetConfig, ModelConfig, TrainingConfig
from src.config.constants import DatasetType


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_config(project_root):
    """Create test configuration."""
    return BaseConfig(
        project_root=project_root,
        seed=42,
        device="cpu"
    )


@pytest.fixture(scope="session")
def test_dataset_config(project_root):
    """Create test dataset configuration."""
    return DatasetConfig(
        project_root=project_root,
        dataset_type=DatasetType.TAFENG,
        max_len=10,
        batch_size=4,
        num_workers=0
    )


@pytest.fixture(scope="session")
def test_model_config():
    """Create test model configuration."""
    return ModelConfig(
        vocab_size=100,
        dim=32,
        n_layers=1,
        n_heads=2,
        max_len=10
    )


@pytest.fixture(scope="session")
def test_training_config():
    """Create test training configuration."""
    return TrainingConfig(
        learning_rate=1e-3,
        max_epochs=2,
        batch_size=4,
        eval_steps=10,
        save_steps=10,
        logging_steps=5
    )


@pytest.fixture(scope="function")
def sample_data():
    """Create sample data for testing."""
    return {
        'sequences': torch.randint(0, 100, (4, 10)),
        'labels': torch.randint(0, 100, (4,)),
        'time_features': {
            'weekday': torch.randint(0, 7, (4, 10)),
            'day': torch.randint(1, 32, (4, 10)),
            'month': torch.randint(1, 13, (4, 10))
        }
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
