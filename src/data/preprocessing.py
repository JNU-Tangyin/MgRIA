"""Data preprocessing utilities for MgRIA."""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..config import DatasetConfig, DatasetType
from ..utils.helpers import ensure_dir


class DataPreprocessor:
    """Data preprocessing pipeline for MgRIA datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.dataset_type = config.dataset_type
    
    def load_and_preprocess(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load and preprocess dataset.
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        # Load raw data
        df = self._load_raw_data()
        
        # Split train/test
        df_train, df_test = train_test_split(
            df, 
            test_size=self.config.test_ratio + self.config.valid_ratio,
            random_state=42
        )
        
        # Process train and test data
        train_data = self._process_dataframe(df_train)
        test_data = self._process_dataframe(df_test)
        
        return train_data, test_data
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset file."""
        # Get dataset file path based on type
        if self.dataset_type == DatasetType.EQUITY:
            file_path = self.config.datasets_path / "equity.csv"
            encoding = 'gbk'
        elif self.dataset_type == DatasetType.TAFENG:
            file_path = self.config.datasets_path / "tafeng.csv"
            encoding = 'utf-8'
        elif self.dataset_type == DatasetType.TAOBAO:
            file_path = self.config.datasets_path / "taobao.csv"
            encoding = 'utf-8'
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        return pd.read_csv(file_path, encoding=encoding)
    
    def _process_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process dataframe to extract sequences and features."""
        # Standardize column names based on dataset type
        df = self._standardize_columns(df)
        
        # Sort by time
        df = df.sort_values(by='time')
        
        # Convert time to timestamp
        df = self._process_timestamps(df)
        
        # Process categories (for equity dataset)
        if self.dataset_type == DatasetType.EQUITY:
            df, category_mapping = self._process_categories(df)
        else:
            category_mapping = None
        
        # Extract time features
        df = self._extract_time_features(df)
        
        # Group by user to create sequences
        user_data = self._create_user_sequences(df)
        
        return {
            'user_sequences': user_data['sequences'],
            'time_features': user_data['time_features'],
            'time_stamps': user_data['timestamps'],
            'categories': user_data['categories'],
            'category_mapping': category_mapping
        }
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on dataset type."""
        column_mapping = self._get_column_mapping()
        return df.rename(columns=column_mapping)
    
    def _get_column_mapping(self) -> Dict[str, str]:
        """Get column name mapping for dataset type."""
        if self.dataset_type == DatasetType.EQUITY:
            return {
                'user': 'session_id',
                'item': 'item_id',
                'createtime': 'time'
            }
        elif self.dataset_type == DatasetType.TAFENG:
            return {
                'session_id': 'session_id',
                'item_id': 'item_id',
                'time': 'time'
            }
        elif self.dataset_type == DatasetType.TAOBAO:
            return {
                'SessionID': 'session_id',
                'ItemID': 'item_id',
                'Time': 'time'
            }
        else:
            return {}
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process timestamps based on dataset type."""
        if self.dataset_type == DatasetType.EQUITY:
            # Convert string timestamps to unix timestamps
            df['time'] = df['time'].apply(
                lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp()
            )
        
        # Convert to datetime for feature extraction
        df['datetime'] = pd.to_datetime(df['time'], unit='s' if self.dataset_type == DatasetType.EQUITY else None)
        
        return df
    
    def _process_categories(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Process categories for equity dataset."""
        category_mapping = {
            '视频': 1, '游戏': 3, '出行': 3, '电商': 0, '音频': 1, 
            '阅读': 2, '美食': 2, '酒店': 4, '医疗': 4, '生活服务': 3, 
            '通信': 4, '工具': 0, '教育': 4, '办公': 2, '快递': 4
        }
        
        df['classify'] = df['classify'].map(category_mapping)
        df['classify'] = df['classify'].fillna(0).astype(int)
        
        return df, category_mapping
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time features from datetime."""
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        
        return df
    
    def _create_user_sequences(self, df: pd.DataFrame) -> Dict[str, List[List]]:
        """Create user sequences from processed dataframe."""
        # Group by session_id and aggregate sequences
        grouped = df.groupby('session_id')
        
        # Extract sequences
        sequences = grouped['item_id'].apply(list).tolist()
        timestamps = grouped['time'].apply(list).tolist()
        
        # Extract time features
        time_features = {
            'month': grouped['month'].apply(list).tolist(),
            'day': grouped['day'].apply(list).tolist(),
            'weekday': grouped['weekday'].apply(list).tolist()
        }
        
        # Extract categories (for equity dataset)
        if self.dataset_type == DatasetType.EQUITY and 'classify' in df.columns:
            categories = grouped['classify'].apply(list).tolist()
        else:
            categories = []
        
        return {
            'sequences': sequences,
            'time_features': time_features,
            'timestamps': timestamps,
            'categories': categories
        }
    
    @staticmethod
    def convert_unique_idx(df: pd.DataFrame, column_name: str, mapping: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """Convert column values to unique indices."""
        if mapping is None:
            unique_values = df[column_name].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
        
        df[column_name] = df[column_name].map(mapping)
        df[column_name] = df[column_name].fillna(0).astype(int)
        
        return df, mapping
