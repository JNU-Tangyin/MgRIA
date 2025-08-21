"""Real data loading utilities for MgRIA (Tafeng).

Builds user/session sequences, time features, and splits into train/val/test
compatible with MgRIADataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from .dataset import MgRIADataset
from ..config import DatasetConfig, ModelConfig
from ..config.constants import DatasetType


@dataclass
class LoadedData:
    train: MgRIADataset
    valid: MgRIADataset
    test: MgRIADataset
    vocab_size: int


def _build_vocab(items: pd.Series, special_tokens) -> Tuple[Dict[object, int], int]:
    """Map original item ids (any dtype) to contiguous ids starting at 0 when needed.
    - If items are numeric and fit under pad_id, we keep original numeric ids (no mapping).
    - Otherwise (strings or large ids), we remap to a dense [0..n-1] and set vocab_size=n+3.
    """
    unique_items = pd.Index(items.dropna().unique())
    pad_id = special_tokens.pad_id
    # Check if purely numeric and within range
    if pd.api.types.is_numeric_dtype(items) and len(unique_items) > 0:
        try:
            max_item_id = float(pd.to_numeric(unique_items, errors='coerce').max())
        except Exception:
            max_item_id = float('inf')
        if np.isfinite(max_item_id) and max_item_id < pad_id:
            return {}, pad_id + 1
    # Remap for non-numeric or large numeric ids
    mapping: Dict[object, int] = {v: i for i, v in enumerate(unique_items)}
    vocab_size = len(unique_items) + 3
    return mapping, vocab_size


def _apply_item_mapping(seq: List[object], mapping: Dict[object, int]) -> List[int]:
    if not mapping:
        # assume seq already contains small integers
        return [int(x) for x in seq]
    return [mapping.get(x, 0) for x in seq]


def _extract_time_features(ts: List[float]) -> Tuple[List[int], List[int], List[int]]:
    # ts: epoch seconds
    import datetime as dt
    months, days, weekdays = [], [], []
    for t in ts:
        try:
            d = dt.datetime.utcfromtimestamp(float(t))
        except Exception:
            # Fallback: if it's already in days or bad, default zeros
            d = dt.datetime.utcfromtimestamp(0)
        months.append(d.month)
        days.append(d.day)
        weekdays.append(d.weekday())
    return months, days, weekdays


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with encoding fallbacks (utf-8 -> latin1 -> gb18030)."""
    encodings = ['utf-8', 'latin1', 'gb18030']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def _normalize_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Normalize to columns: session_id, item_id, time.
    Supports common alternates: user/user_id, item/itemId, createtime/updatatime, timestamp/behavior_time.
    Ensures time is epoch seconds (float).
    """
    # Build candidate lists
    session_candidates = ['session_id', 'SessionID', 'sessionId', 'user', 'user_id', 'UserID']
    item_candidates = ['item_id', 'ItemID', 'item', 'itemId', 'sku_id']
    time_candidates = ['time', 'Time', 'timestamp', 'Timestamp', 'createtime', 'create_time', 'updatatime', 'update_time', 'behavior_time', 'datetime']

    def pick(col_list):
        for c in col_list:
            if c in df.columns:
                return c
        return None

    s_col = pick(session_candidates)
    i_col = pick(item_candidates)
    # Choose time column with smartest fallback: prefer the one with most non-nulls;
    # also consider coalesced pairs like createtime/updatatime
    present_time_cols = [c for c in time_candidates if c in df.columns]
    t_col = None
    if present_time_cols:
        # Build candidate series map
        cand = {c: df[c] for c in present_time_cols}
        # Add coalesced pairs if present
        if 'createtime' in cand and 'updatatime' in cand:
            cand['createtime_coalesced'] = cand['createtime'].fillna(cand['updatatime'])
        if 'create_time' in cand and 'update_time' in cand:
            cand['create_time_coalesced'] = cand['create_time'].fillna(cand['update_time'])
        # Pick key with max non-null
        t_col = max(cand.keys(), key=lambda k: cand[k].notna().sum())
        t_series = cand[t_col]
    else:
        t_series = None

    if s_col is None or i_col is None or t_series is None:
        raise AssertionError(f"{filename} missing one of required columns; found {list(df.columns)}")

    out = df.rename(columns={s_col: 'session_id', i_col: 'item_id'})
    out['time'] = t_series

    # Ensure types
    # Keep item_id as-is (can be string); session_id as string
    out['session_id'] = out['session_id'].astype(str)

    # Normalize time to epoch seconds
    if not np.issubdtype(out['time'].dtype, np.number):
        # parse datetime strings with several strategies
        t_candidates = []
        # 1) utc True
        t_candidates.append(pd.to_datetime(out['time'], errors='coerce', utc=True))
        # 2) utc False
        t_candidates.append(pd.to_datetime(out['time'], errors='coerce', utc=False))
        # 3) dayfirst
        t_candidates.append(pd.to_datetime(out['time'], errors='coerce', dayfirst=True, utc=False))
        # 4) common formats
        common_formats = [
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
            '%Y-%m-%d', '%Y/%m/%d'
        ]
        for fmt in common_formats:
            try:
                t_candidates.append(pd.to_datetime(out['time'], format=fmt, errors='coerce', utc=False))
            except Exception:
                continue
        # Pick candidate with most non-nulls
        t = max(t_candidates, key=lambda s: s.notna().sum()) if t_candidates else pd.Series(pd.NaT, index=out.index)
        if t.notna().sum() == 0:
            # As a last resort, try to coerce numeric-looking strings
            num = pd.to_numeric(out['time'], errors='coerce')
            if num.notna().sum() > 0:
                # detect ms vs s
                med = float(np.nanmedian(num.values)) if num.notna().any() else 0.0
                if med > 1e12:
                    num = num / 1000.0
                out['time'] = num.astype(float)
            else:
                # Provide diagnostics
                sample_vals = out['time'].astype(str).dropna().unique().tolist()[:5]
                raise ValueError(f"Unable to parse time column in {filename}. Sample values: {sample_vals}")
        else:
            out['time'] = t.view('int64') // 10**9
    else:
        # numeric; detect ms vs s
        t = out['time'].astype(float)
        # Heuristic: if median > 1e12, it's ms
        try:
            med = float(np.nanmedian(t.values))
        except Exception:
            med = float(t.iloc[0]) if len(t) else 0.0
        if med > 1e12:
            t = t / 1000.0
        out['time'] = t

    # Drop rows with NaNs in key fields
    out = out.dropna(subset=['session_id', 'item_id', 'time'])
    # Cast time to float seconds; keep item_id original dtype
    out['time'] = out['time'].astype(float)
    out = out[['session_id', 'item_id', 'time']]
    if out.empty:
        raise ValueError(f"After normalization, no valid rows found in {filename}. Check time/item/session columns.")
    return out


def _load_generic_csv(
    config: DatasetConfig,
    model_config: ModelConfig,
    filename: str
) -> Tuple[LoadedData, ModelConfig]:
    """Generic CSV loader used by Tafeng/Equity/Taobao.
    Expected columns: ['session_id', 'item_id', 'time'] (case-insensitive variants supported).
    """
    data_path = config.project_root / 'datasets' / filename
    df = _read_csv_robust(data_path)
    # Normalize columns to expected names and types
    df = _normalize_columns(df, filename)

    # Build vocab mapping if necessary
    mapping, vocab_size = _build_vocab(df['item_id'], config.special_tokens)

    # Group by session as sequences, sorted by time
    df = df.sort_values(['session_id', 'time'])
    sequences: List[List[int]] = []
    months_l: List[List[int]] = []
    days_l: List[List[int]] = []
    weekdays_l: List[List[int]] = []
    timestamps_l: List[List[float]] = []

    for sid, g in df.groupby('session_id'):
        items = g['item_id'].tolist()
        times = g['time'].astype(float).tolist()
        items = _apply_item_mapping(items, mapping)
        m, d, w = _extract_time_features(times)
        sequences.append(items)
        months_l.append(m)
        days_l.append(d)
        weekdays_l.append(w)
        timestamps_l.append(times)

    # Split train/valid/test by sessions
    n = len(sequences)
    n_train = int(n * config.train_ratio)
    n_valid = int(n * config.valid_ratio)
    idxs = np.arange(n)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(idxs)

    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train:n_train + n_valid]
    test_idx = idxs[n_train + n_valid:]

    def subset(idxs_arr):
        return (
            [sequences[i] for i in idxs_arr],
            {
                'month': [months_l[i] for i in idxs_arr],
                'day': [days_l[i] for i in idxs_arr],
                'weekday': [weekdays_l[i] for i in idxs_arr],
            },
            [timestamps_l[i] for i in idxs_arr],
        )

    seq_tr, tf_tr, ts_tr = subset(train_idx)
    seq_va, tf_va, ts_va = subset(valid_idx)
    seq_te, tf_te, ts_te = subset(test_idx)

    # Build datasets
    train_ds = MgRIADataset(config=config, phase='train', user_sequences=seq_tr, time_features=tf_tr, time_stamps=ts_tr)
    valid_ds = MgRIADataset(config=config, phase='test', user_sequences=seq_va, time_features=tf_va, time_stamps=ts_va)
    test_ds = MgRIADataset(config=config, phase='test', user_sequences=seq_te, time_features=tf_te, time_stamps=ts_te)

    # Update vocab size to either mapping-based or config default
    if mapping:
        model_config.vocab_size = vocab_size
    else:
        # Ensure it matches special tokens' vocab if provided in config
        model_config.vocab_size = getattr(config.special_tokens, 'vocab_size', model_config.vocab_size)

    return LoadedData(train=train_ds, valid=valid_ds, test=test_ds, vocab_size=model_config.vocab_size), model_config


def load_tafeng(config: DatasetConfig, model_config: ModelConfig) -> Tuple[LoadedData, ModelConfig]:
    """Load Tafeng dataset from CSV."""
    return _load_generic_csv(config, model_config, 'tafeng.csv')


def load_equity(config: DatasetConfig, model_config: ModelConfig) -> Tuple[LoadedData, ModelConfig]:
    """Load Equity dataset from CSV."""
    return _load_generic_csv(config, model_config, 'equity.csv')


def load_taobao(config: DatasetConfig, model_config: ModelConfig) -> Tuple[LoadedData, ModelConfig]:
    """Load Taobao dataset from CSV."""
    return _load_generic_csv(config, model_config, 'taobao.csv')


def load_real_dataset(config: DatasetConfig, model_config: ModelConfig) -> Tuple[LoadedData, ModelConfig]:
    """Dispatcher for real datasets by DatasetType."""
    if config.dataset_type == DatasetType.TAFENG:
        return load_tafeng(config, model_config)
    elif config.dataset_type == DatasetType.EQUITY:
        return load_equity(config, model_config)
    elif config.dataset_type == DatasetType.TAOBAO:
        return load_taobao(config, model_config)
    else:
        raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")
