"""DataLoader utilities for MgRIA."""

from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from .dataset import MgRIADataset
from ..config import DatasetConfig


def create_dataloader(
    dataset: MgRIADataset,
    config: DatasetConfig,
    shuffle: bool = True,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for MgRIA dataset.
    
    Args:
        dataset: MgRIA dataset instance
        config: Dataset configuration
        shuffle: Whether to shuffle data
        batch_size: Batch size (uses config if None)
        num_workers: Number of workers (uses config if None)
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers or config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """
    Custom collate function for MgRIA batches.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched tensors
    """
    # Stack all tensors in the batch
    import torch
    batched = {}

    # First, stack fixed-size tensors
    for key in batch[0].keys():
        if key not in ['masked_ids', 'masked_pos', 'masked_weights']:
            batched[key] = torch.stack([item[key] for item in batch])

    # Pad variable-length fields to max length in batch
    # Use -100 for masked_ids padding (ignored by CrossEntropyLoss)
    # Use 0 for positions and weights padding
    var_keys = ['masked_ids', 'masked_pos', 'masked_weights']
    max_len = 0
    for item in batch:
        for k in var_keys:
            l = item[k].shape[0] if hasattr(item[k], 'shape') else len(item[k])
            if l > max_len:
                max_len = l

    if max_len == 0:
        # Edge case: no masks, create dummy of length 1
        max_len = 1

    # Prepare padded tensors
    ids_pad_val = -100
    pos_pad_val = 0
    w_pad_val = 0.0

    masked_ids = torch.full((len(batch), max_len), ids_pad_val, dtype=torch.long)
    masked_pos = torch.full((len(batch), max_len), pos_pad_val, dtype=torch.long)
    masked_weights = torch.full((len(batch), max_len), w_pad_val, dtype=torch.float)

    for i, item in enumerate(batch):
        ids = item['masked_ids']
        pos = item['masked_pos']
        wts = item['masked_weights']
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(pos, torch.Tensor):
            pos = pos.tolist()
        if isinstance(wts, torch.Tensor):
            wts = wts.tolist()
        L = min(len(ids), max_len)
        if L > 0:
            masked_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            masked_pos[i, :L] = torch.tensor(pos[:L], dtype=torch.long)
            masked_weights[i, :L] = torch.tensor(wts[:L], dtype=torch.float)

    batched['masked_ids'] = masked_ids
    batched['masked_pos'] = masked_pos
    batched['masked_weights'] = masked_weights

    return batched
