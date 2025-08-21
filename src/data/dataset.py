"""Dataset classes for MgRIA."""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import datetime

from ..config import DatasetConfig
from ..utils.helpers import load_vocab


class MgRIADataset(Dataset):
    """Dataset class for MgRIA model."""
    
    def __init__(
        self,
        config: DatasetConfig,
        phase: str = 'train',
        user_sequences: Optional[List[List[int]]] = None,
        time_features: Optional[Dict[str, List[List[int]]]] = None,
        time_stamps: Optional[List[List[float]]] = None,
        categories: Optional[List[List[int]]] = None,
        vocabulary: Optional[List[str]] = None
    ):
        """
        Initialize MgRIA dataset.
        
        Args:
            config: Dataset configuration
            phase: 'train' or 'test'
            user_sequences: User item sequences
            time_features: Time feature dict with 'month', 'day', 'weekday'
            time_stamps: Timestamp sequences
            categories: Category sequences (for equity dataset)
            vocabulary: Item vocabulary
        """
        self.config = config
        self.phase = phase
        self.max_len = config.max_len
        self.mask_prob = config.mask_prob
        
        # Special tokens
        self.special_tokens = config.special_tokens
        self.mask_id = self.special_tokens.mask_id
        self.pad_id = self.special_tokens.pad_id
        self.interest_id = self.special_tokens.interest_id
        
        # Data
        self.user_sequences = user_sequences or []
        self.time_features = time_features or {}
        self.time_stamps = time_stamps or []
        self.categories = categories or []
        self.vocabulary = vocabulary or []
        
        # Validate data consistency
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate data consistency."""
        if not self.user_sequences:
            return
            
        n_users = len(self.user_sequences)
        
        # Check time features consistency
        for feature_name, feature_data in self.time_features.items():
            if len(feature_data) != n_users:
                raise ValueError(f"Time feature '{feature_name}' length mismatch: {len(feature_data)} vs {n_users}")
        
        # Check timestamps consistency
        if self.time_stamps and len(self.time_stamps) != n_users:
            raise ValueError(f"Timestamps length mismatch: {len(self.time_stamps)} vs {n_users}")
        
        # Check categories consistency (for equity dataset)
        if self.categories and len(self.categories) != n_users:
            raise ValueError(f"Categories length mismatch: {len(self.categories)} vs {n_users}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.user_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        sequence = self.user_sequences[idx].copy()
        
        # Get time features
        time_feature_dict = {}
        for feature_name, feature_data in self.time_features.items():
            time_feature_dict[feature_name] = feature_data[idx].copy()
        
        # Get timestamps and categories
        timestamps = self.time_stamps[idx].copy() if self.time_stamps else []
        categories = self.categories[idx].copy() if self.categories else []
        
        # Process sequence
        processed_data = self._process_sequence(
            sequence, time_feature_dict, timestamps, categories
        )
        
        return processed_data
    
    def _process_sequence(
        self,
        sequence: List[int],
        time_features: Dict[str, List[int]],
        timestamps: List[float],
        categories: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Process a single sequence."""
        # Truncate or pad sequence to max_len
        if len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
            for feature_name in time_features:
                time_features[feature_name] = time_features[feature_name][-self.max_len:]
            if timestamps:
                timestamps = timestamps[-self.max_len:]
            if categories:
                categories = categories[-self.max_len:]
        
        # Padding
        seq_len = len(sequence)
        padded_sequence = sequence + [self.pad_id] * (self.max_len - seq_len)
        
        # Pad time features
        padded_time_features = {}
        for feature_name, feature_values in time_features.items():
            padded_time_features[feature_name] = feature_values + [0] * (self.max_len - seq_len)
        
        # Pad timestamps and categories
        padded_timestamps = timestamps + [0.0] * (self.max_len - seq_len) if timestamps else [0.0] * self.max_len
        padded_categories = categories + [0] * (self.max_len - seq_len) if categories else [0] * self.max_len
        
        # Create input mask
        input_mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        # Masking for training
        if self.phase == 'train':
            masked_sequence, masked_ids, masked_pos, masked_weights = self._apply_masking(
                padded_sequence, input_mask
            )
        else:
            # For testing, mask the last item
            masked_sequence = padded_sequence.copy()
            if seq_len > 0:
                masked_sequence[seq_len - 1] = self.mask_id
                masked_ids = [padded_sequence[seq_len - 1]]
                masked_pos = [seq_len - 1]
                masked_weights = [1.0]
            else:
                masked_ids = [self.pad_id]
                masked_pos = [0]
                masked_weights = [0.0]
        
        # Calculate time gaps
        time_gaps = self._calculate_time_gaps(padded_timestamps)
        
        # Combine time features into time matrix
        time_matrix = []
        for i in range(self.max_len):
            time_row = []
            for feature_name in ['month', 'day', 'weekday']:
                if feature_name in padded_time_features:
                    time_row.append(padded_time_features[feature_name][i])
                else:
                    time_row.append(0)
            time_matrix.append(time_row)
        
        return {
            'sequences': torch.tensor(masked_sequence, dtype=torch.long),
            'time_features': torch.tensor(time_matrix, dtype=torch.long),
            'input_mask': torch.tensor(input_mask, dtype=torch.float),
            'masked_ids': torch.tensor(masked_ids, dtype=torch.long),
            'masked_pos': torch.tensor(masked_pos, dtype=torch.long),
            'masked_weights': torch.tensor(masked_weights, dtype=torch.float),
            'time_gaps': torch.tensor(time_gaps, dtype=torch.long),
            'categories': torch.tensor(padded_categories, dtype=torch.long)
        }
    
    def _apply_masking(
        self, 
        sequence: List[int], 
        input_mask: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[float]]:
        """Apply masking to sequence for training."""
        masked_sequence = sequence.copy()
        masked_ids = []
        masked_pos = []
        masked_weights = []
        
        # Get valid positions (non-padded)
        valid_positions = [i for i, mask in enumerate(input_mask) if mask == 1]
        
        if not valid_positions:
            return masked_sequence, [self.pad_id], [0], [0.0]
        
        # Random masking
        n_mask = max(1, int(len(valid_positions) * self.mask_prob))
        mask_positions = np.random.choice(valid_positions, size=n_mask, replace=False)
        
        for pos in mask_positions:
            original_id = sequence[pos]
            masked_sequence[pos] = self.mask_id
            masked_ids.append(original_id)
            masked_pos.append(pos)
            masked_weights.append(1.0)
        
        return masked_sequence, masked_ids, masked_pos, masked_weights
    
    def _calculate_time_gaps(self, timestamps: List[float]) -> List[int]:
        """Calculate time gaps between consecutive items."""
        time_gaps = [0]  # First item has no gap
        
        for i in range(1, len(timestamps)):
            if timestamps[i] > 0 and timestamps[i-1] > 0:
                gap = timestamps[i] - timestamps[i-1]
                # Convert to days and clip to reasonable range
                gap_days = max(0, min(365, int(gap / (24 * 3600))))
                time_gaps.append(gap_days)
            else:
                time_gaps.append(0)
        
        return time_gaps
