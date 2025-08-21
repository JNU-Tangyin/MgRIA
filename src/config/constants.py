"""Constants and enumerations for MgRIA project."""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class DatasetType(Enum):
    """Dataset types supported by MgRIA."""
    EQUITY = "equity"
    TAFENG = "tafeng" 
    TAOBAO = "taobao"


class MetricType(Enum):
    """Evaluation metrics."""
    RECALL = "recall"
    MRR = "mrr"
    NDCG = "ndcg"


class ModelType(Enum):
    """Model types."""
    MGRIA = "MgRIA"
    BERT4REC = "BERT4Rec"
    GRU4REC = "GRU4Rec"
    REPEATNET = "RepeatNet"


class PhaseType(Enum):
    """Training/testing phases."""
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"


# Dataset configurations
DATASET_METRICS: Dict[DatasetType, List[str]] = {
    DatasetType.EQUITY: ['recall@3', 'recall@10', 'mrr@10', 'ndcg@10'],
    DatasetType.TAFENG: ['recall@10', 'recall@20', 'mrr@20', 'ndcg@20'],
    DatasetType.TAOBAO: ['recall@20', 'mrr@20', 'ndcg@20']
}

# Model baselines for comparison
BASELINE_MODELS: List[str] = [
    'POP', 'S-POP', 'BPR-MF', 'Item-KNN', 'TIFU-KNN', 'STAN', 
    'GRU4Rec', 'NARM', 'RepeatNet', 'SR-GNN', 'BERT4Rec', 
    'CORE', 'CoHHN', 'RepeatNet-DASP', 'Ride Buy-Cycle', 'MgRIA'
]

# Time embedding constants
TIME_EMBEDDING_SIZES = {
    'weekday_size': 8,
    'day_size': 32,
    'month_size': 13
}

# Default paths (relative to project root)
DEFAULT_PATHS = {
    'datasets': 'datasets',
    'results': 'results',
    'figures': 'figures',
    'tables': 'tables',
    'models': 'models'
}

# Special token IDs
@dataclass(frozen=True)
class SpecialTokens:
    """Special token IDs for different datasets."""
    
    @dataclass(frozen=True)
    class Equity:
        vocab_size: int = 259
        mask_id: int = 256
        pad_id: int = 257
        interest_id: int = 258
    
    @dataclass(frozen=True)
    class Tafeng:
        vocab_size: int = 15789  # 15786+3
        mask_id: int = 15787
        pad_id: int = 15786
        interest_id: int = 15788
    
    @dataclass(frozen=True)
    class Taobao:
        vocab_size: int = 287008  # 287005+3
        mask_id: int = 287006
        pad_id: int = 287005
        interest_id: int = 287007
