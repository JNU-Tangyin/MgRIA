"""Test data processing modules."""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import MgRIADataset, DataPreprocessor, DataValidator, DataFactory
from src.config import DatasetConfig
from src.config.constants import DatasetType


class TestMgRIADataset:
    """Test MgRIADataset class."""
    
    def test_init(self, test_dataset_config):
        """Test dataset initialization."""
        dataset = MgRIADataset(test_dataset_config)
        assert dataset.config == test_dataset_config
        assert dataset.max_len == test_dataset_config.max_len
        assert len(dataset) == 0
    
    def test_with_data(self, test_dataset_config):
        """Test dataset with sample data."""
        # Create sample data
        user_sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        time_features = {
            'month': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            'day': [[10, 11, 12], [13, 14], [15, 16, 17, 18]],
            'weekday': [[0, 1, 2], [3, 4], [5, 6, 0, 1]]
        }
        time_stamps = [[100.0, 200.0, 300.0], [400.0, 500.0], [600.0, 700.0, 800.0, 900.0]]
        
        dataset = MgRIADataset(
            test_dataset_config,
            phase='train',
            user_sequences=user_sequences,
            time_features=time_features,
            time_stamps=time_stamps
        )
        
        assert len(dataset) == 3
        
        # Test getting item
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'sequences' in item
        assert 'time_features' in item
        assert 'input_mask' in item
        assert item['sequences'].shape[0] == test_dataset_config.max_len
    
    def test_validation_error(self, test_dataset_config):
        """Test data validation errors."""
        # Mismatched lengths should raise error
        user_sequences = [[1, 2, 3], [4, 5]]
        time_features = {
            'month': [[1, 2, 3]]  # Only one user, should be two
        }
        
        with pytest.raises(ValueError):
            MgRIADataset(
                test_dataset_config,
                user_sequences=user_sequences,
                time_features=time_features
            )


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_init(self, test_dataset_config):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(test_dataset_config)
        assert preprocessor.config == test_dataset_config
        assert preprocessor.dataset_type == test_dataset_config.dataset_type
    
    def test_column_mapping(self, test_dataset_config):
        """Test column name mapping."""
        preprocessor = DataPreprocessor(test_dataset_config)
        mapping = preprocessor._get_column_mapping()
        
        if test_dataset_config.dataset_type == DatasetType.TAFENG:
            assert 'session_id' in mapping
            assert 'item_id' in mapping
            assert 'time' in mapping


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_init(self, test_dataset_config):
        """Test validator initialization."""
        validator = DataValidator(test_dataset_config)
        assert validator.config == test_dataset_config
    
    def test_validate_sequences(self, test_dataset_config):
        """Test sequence validation."""
        validator = DataValidator(test_dataset_config)
        
        # Valid sequences
        sequences = [[1, 2, 3], [4, 5, 6, 7]]
        report = {'valid': True, 'errors': [], 'warnings': [], 'stats': {}}
        validator._validate_sequences(sequences, report)
        
        assert report['valid']
        assert 'sequence_lengths' in report['stats']
        assert report['stats']['sequence_lengths']['min'] == 3
        assert report['stats']['sequence_lengths']['max'] == 4
    
    def test_validate_empty_sequences(self, test_dataset_config):
        """Test validation with empty sequences."""
        validator = DataValidator(test_dataset_config)
        
        sequences = []
        report = {'valid': True, 'errors': [], 'warnings': [], 'stats': {}}
        validator._validate_sequences(sequences, report)
        
        assert not report['valid']
        assert len(report['errors']) > 0
    
    def test_validate_time_features(self, test_dataset_config):
        """Test time feature validation."""
        validator = DataValidator(test_dataset_config)
        
        # Valid time features
        time_features = {
            'month': [[1, 2, 3], [4, 5]],
            'day': [[10, 11, 12], [13, 14]],
            'weekday': [[0, 1, 2], [3, 4]]
        }
        
        report = {'valid': True, 'errors': [], 'warnings': [], 'stats': {}}
        validator._validate_time_features(time_features, report)
        
        # Should have no errors for valid ranges
        month_errors = [w for w in report['warnings'] if 'month' in w and 'Invalid' in w]
        assert len(month_errors) == 0
    
    def test_validate_invalid_time_features(self, test_dataset_config):
        """Test validation with invalid time features."""
        validator = DataValidator(test_dataset_config)
        
        # Invalid time features
        time_features = {
            'month': [[13, 14, 15]],  # Invalid months
            'day': [[32, 33, 34]],    # Invalid days
            'weekday': [[7, 8, 9]]    # Invalid weekdays
        }
        
        report = {'valid': True, 'errors': [], 'warnings': [], 'stats': {}}
        validator._validate_time_features(time_features, report)
        
        # Should have warnings for invalid ranges
        assert len(report['warnings']) >= 3


class TestDataFactory:
    """Test DataFactory class."""
    
    def test_get_dataset_info(self, test_dataset_config):
        """Test getting dataset info."""
        dataset = MgRIADataset(test_dataset_config)
        info = DataFactory.get_dataset_info(dataset)
        
        assert 'num_users' in info
        assert 'max_len' in info
        assert 'vocab_size' in info
        assert info['max_len'] == test_dataset_config.max_len
