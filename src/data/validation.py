"""Data validation utilities."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import DatasetConfig, DatasetType


class DataValidator:
    """Data validation utilities for MgRIA datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize validator with configuration."""
        self.config = config
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw dataset.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report['valid'] = False
            report['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types and ranges
        if report['valid']:
            self._validate_data_types(df, report)
            self._validate_data_ranges(df, report)
            self._compute_stats(df, report)
        
        return report
    
    def validate_processed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed data.
        
        Args:
            data: Processed data dictionary
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required keys
        required_keys = ['user_sequences', 'time_features', 'time_stamps']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            report['valid'] = False
            report['errors'].append(f"Missing required keys: {missing_keys}")
            return report
        
        # Validate sequences
        self._validate_sequences(data['user_sequences'], report)
        
        # Validate time features
        self._validate_time_features(data['time_features'], report)
        
        # Validate consistency
        self._validate_consistency(data, report)
        
        return report
    
    def _get_required_columns(self) -> List[str]:
        """Get required columns for dataset type."""
        if self.config.dataset_type == DatasetType.EQUITY:
            return ['user', 'item', 'createtime']
        elif self.config.dataset_type == DatasetType.TAFENG:
            return ['session_id', 'item_id', 'time']
        elif self.config.dataset_type == DatasetType.TAOBAO:
            return ['SessionID', 'ItemID', 'Time']
        else:
            return []
    
    def _validate_data_types(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate data types."""
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            report['warnings'].append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report['warnings'].append(f"Found {duplicates} duplicate rows")
    
    def _validate_data_ranges(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate data ranges."""
        # Check for reasonable timestamp ranges
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        for col in time_cols:
            if df[col].dtype in ['int64', 'float64']:
                min_val, max_val = df[col].min(), df[col].max()
                if min_val < 0:
                    report['warnings'].append(f"Negative timestamps in {col}")
                if max_val > 2e9:  # Year 2033
                    report['warnings'].append(f"Future timestamps in {col}")
    
    def _compute_stats(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Compute basic statistics."""
        report['stats'] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # User/session statistics
        session_cols = [col for col in df.columns if 'session' in col.lower() or 'user' in col.lower()]
        if session_cols:
            session_col = session_cols[0]
            report['stats']['num_users'] = df[session_col].nunique()
            report['stats']['avg_interactions_per_user'] = len(df) / df[session_col].nunique()
        
        # Item statistics
        item_cols = [col for col in df.columns if 'item' in col.lower()]
        if item_cols:
            item_col = item_cols[0]
            report['stats']['num_items'] = df[item_col].nunique()
            report['stats']['avg_interactions_per_item'] = len(df) / df[item_col].nunique()
    
    def _validate_sequences(self, sequences: List[List[int]], report: Dict[str, Any]) -> None:
        """Validate user sequences."""
        if not sequences:
            report['valid'] = False
            report['errors'].append("Empty sequences")
            return
        
        # Check sequence lengths
        seq_lengths = [len(seq) for seq in sequences]
        if not seq_lengths:
            report['errors'].append("All sequences are empty")
            return
        
        min_len, max_len, avg_len = min(seq_lengths), max(seq_lengths), np.mean(seq_lengths)
        report['stats']['sequence_lengths'] = {
            'min': min_len,
            'max': max_len,
            'avg': avg_len,
            'std': np.std(seq_lengths)
        }
        
        # Check for sequences that are too short
        short_sequences = sum(1 for length in seq_lengths if length < 2)
        if short_sequences > 0:
            report['warnings'].append(f"Found {short_sequences} sequences with length < 2")
    
    def _validate_time_features(self, time_features: Dict[str, List[List[int]]], report: Dict[str, Any]) -> None:
        """Validate time features."""
        expected_features = ['month', 'day', 'weekday']
        
        for feature in expected_features:
            if feature not in time_features:
                report['warnings'].append(f"Missing time feature: {feature}")
                continue
            
            # Validate ranges
            feature_data = time_features[feature]
            if feature_data:
                flat_values = [val for seq in feature_data for val in seq]
                if flat_values:
                    min_val, max_val = min(flat_values), max(flat_values)
                    
                    if feature == 'month' and (min_val < 1 or max_val > 12):
                        report['warnings'].append(f"Invalid month values: {min_val}-{max_val}")
                    elif feature == 'day' and (min_val < 1 or max_val > 31):
                        report['warnings'].append(f"Invalid day values: {min_val}-{max_val}")
                    elif feature == 'weekday' and (min_val < 0 or max_val > 6):
                        report['warnings'].append(f"Invalid weekday values: {min_val}-{max_val}")
    
    def _validate_consistency(self, data: Dict[str, Any], report: Dict[str, Any]) -> None:
        """Validate data consistency."""
        sequences = data['user_sequences']
        time_features = data['time_features']
        time_stamps = data.get('time_stamps', [])
        
        n_users = len(sequences)
        
        # Check time features consistency
        for feature_name, feature_data in time_features.items():
            if len(feature_data) != n_users:
                report['errors'].append(f"Time feature '{feature_name}' length mismatch: {len(feature_data)} vs {n_users}")
        
        # Check timestamps consistency
        if time_stamps and len(time_stamps) != n_users:
            report['errors'].append(f"Timestamps length mismatch: {len(time_stamps)} vs {n_users}")
        
        # Check sequence-level consistency
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            
            # Check time features have same length as sequence
            for feature_name, feature_data in time_features.items():
                if i < len(feature_data) and len(feature_data[i]) != seq_len:
                    report['errors'].append(f"User {i}: sequence length mismatch for {feature_name}")
            
            # Check timestamps have same length as sequence
            if time_stamps and i < len(time_stamps) and len(time_stamps[i]) != seq_len:
                report['errors'].append(f"User {i}: sequence length mismatch for timestamps")
