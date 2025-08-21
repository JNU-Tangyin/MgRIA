"""Utility helper functions."""

import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import csv


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device based on availability and preference."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save results to CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to list of dictionaries if needed
    if isinstance(results, dict) and not isinstance(list(results.values())[0], list):
        results = [results]
    elif isinstance(results, dict):
        # Convert dict of lists to list of dicts
        keys = list(results.keys())
        rows = []
        for i in range(len(results[keys[0]])):
            row = {key: results[key][i] for key in keys}
            rows.append(row)
        results = rows
    
    if results:
        fieldnames = list(results[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def load_vocab(vocab_path: Union[str, Path]) -> Dict[str, int]:
    """Load vocabulary from file."""
    vocab_path = Path(vocab_path)
    vocab = {}
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = line.strip()
            if item:
                vocab[item] = idx
    
    return vocab


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
