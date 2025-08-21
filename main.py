"""Run MgRIA on real Tafeng data with metrics and saved results."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

import yaml
import torch
import pandas as pd

from src.config import ModelConfig, DatasetConfig, TrainingConfig
from src.config.constants import DatasetType, DATASET_METRICS
from src.data.real_data import load_real_dataset
from src.data.dataloader import create_dataloader
from src.models import ModelFactory
from src.training import MgRIATrainer
from src.utils.helpers import setup_logging, set_seed
from metrics import eval_run  # functions expect a mapping with keys 'R_u' and 'V_u'


def compute_topk_metrics(model, dataloader, device, eval_metrics: List[str]) -> Dict[str, float]:
    model.eval()
    # Initialize accumulators
    agg: Dict[str, float] = {m: 0.0 for m in eval_metrics}
    count = 0
    topk = max(int(m.split('@')[1]) for m in eval_metrics)
    with torch.no_grad():
        for batch in dataloader:
            # Map keys explicitly to avoid tensor truth-value ambiguity
            if 'sequences' in batch:
                input_ids = batch['sequences']
            else:
                input_ids = batch['input_ids']

            time_features = batch['time_features']

            if 'input_mask' in batch:
                attn = batch['input_mask']
            else:
                attn = batch['attention_mask']

            if 'masked_ids' in batch:
                masked_ids = batch['masked_ids']
            else:
                masked_ids = batch['targets']

            # Move to device
            input_ids = input_ids.to(device)
            time_features = time_features.to(device)
            attn = attn.to(device)

            # Predict top-k per sample
            items, _ = model.predict(
                input_ids=input_ids,
                time_features=time_features,
                attention_mask=attn,
                top_k=topk,
            )
            items_cpu = items.cpu().tolist()  # List[List[int]]

            # Resolve ground-truth list per sample
            # Expect tensor from collate_fn: [B, P] padded with -100 or [B]
            if isinstance(masked_ids, list):
                # Backward compatibility, though current collate pads to tensor
                gt_list = []
                for x in masked_ids:
                    if hasattr(x, 'numel') and x.numel() > 0:
                        gt_list.append(int(x[0].item()))
                    else:
                        gt_list.append(-100)
            else:
                if masked_ids.dim() == 2:
                    gt_list = [int(v) for v in masked_ids[:, 0].tolist()]
                else:
                    gt_list = [int(v) for v in masked_ids.tolist()]

            # Accumulate metrics per user
            batch_size = len(items_cpu)
            for i in range(batch_size):
                gt = gt_list[i]
                # Skip samples with no valid target (e.g., padding -100)
                if gt is None or gt < 0:
                    continue
                sample = {'R_u': [int(v) for v in items_cpu[i][:topk]], 'V_u': [int(gt)]}
                for m in eval_metrics:
                    agg[m] += float(eval_run(sample, m))
                count += 1

    if count == 0:
        return {m: 0.0 for m in eval_metrics}
    return {m: agg[m] / count for m in eval_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MgRIA on real datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[d.value for d in DatasetType],
        default=DatasetType.TAFENG.value,
        help="Dataset to run (equity|tafeng|taobao)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoint missing")
    return parser.parse_args()


def _select_device() -> torch.device:
    # Prefer CUDA, then Apple Metal (MPS), else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    setup_logging('INFO')
    logger = logging.getLogger(__name__)

    cfg_path = Path('configs/default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 2300))

    # CLI
    args = parse_args()
    ds_type = DatasetType(args.dataset)

    # Build configs
    dataset_cfg = DatasetConfig(
        dataset_type=ds_type,
        max_len=cfg['dataset']['max_len'],
        mask_prob=cfg['dataset']['mask_prob'],
        train_ratio=cfg['dataset']['train_ratio'],
        valid_ratio=cfg['dataset']['valid_ratio'],
        test_ratio=cfg['dataset']['test_ratio'],
        batch_size=cfg['dataset']['batch_size'],
        num_workers=cfg['dataset']['num_workers'],
    )

    m = cfg['model']
    model_cfg = ModelConfig(
        model_type=m['model_type'],
        vocab_size=int(m['vocab_size']),  # will be overwritten by loader
        dim=int(m['dim']),
        n_layers=int(m['n_layers']),
        n_heads=int(m['n_heads']),
        max_len=int(m['max_len']),
        dropout_prob=float(m['dropout_prob']),
        attention_dropout_prob=float(m['attention_dropout_prob']),
        hidden_dropout_prob=float(m['hidden_dropout_prob']),
        weekday_size=int(m['weekday_size']),
        day_size=int(m['day_size']),
        month_size=int(m['month_size']),
        hidden_act=str(m['hidden_act']),
        layer_norm_eps=float(m['layer_norm_eps']),
        initializer_range=float(m['initializer_range']),
    )

    # Limit epochs for a quick initial run; allow CLI override
    default_cap = 5
    max_epochs_quick = cfg['training']['max_epochs'] if args.epochs is None else args.epochs
    max_epochs_quick = min(max_epochs_quick, default_cap) if args.epochs is None else args.epochs
    t = cfg['training']
    train_cfg = TrainingConfig(
        learning_rate=float(t['learning_rate']),
        weight_decay=float(t['weight_decay']),
        warmup_steps=int(t['warmup_steps']),
        max_epochs=int(max_epochs_quick),
        patience=int(t['patience']),
        batch_size=int(t['batch_size']),
        gradient_accumulation_steps=int(t['gradient_accumulation_steps']),
        max_grad_norm=float(t['max_grad_norm']),
        optimizer=str(t['optimizer']),
        adam_epsilon=float(t['adam_epsilon']),
        adam_beta1=float(t['adam_beta1']),
        adam_beta2=float(t['adam_beta2']),
        lr_scheduler=str(t['lr_scheduler']),
        eval_steps=int(t['eval_steps']),
        save_steps=int(t['save_steps']),
        logging_steps=int(t['logging_steps']),
        eval_metrics=list(t['eval_metrics']),
        save_total_limit=int(t['save_total_limit']),
        save_best_model=bool(t['save_best_model']),
        metric_for_best_model=str(t['metric_for_best_model']),
        greater_is_better=bool(t['greater_is_better']),
        dataloader_num_workers=int(t['dataloader_num_workers']),
        dataloader_pin_memory=bool(t['dataloader_pin_memory']),
        fp16=bool(t['fp16']),
        gradient_checkpointing=bool(t['gradient_checkpointing']),
    )

    device = _select_device()

    # Load real data and update vocab_size
    loaded, model_cfg = load_real_dataset(dataset_cfg, model_cfg)
    logger.info(
        f"Loaded {dataset_cfg.dataset_type.value}: "
        f"train={len(loaded.train)}, valid={len(loaded.valid)}, test={len(loaded.test)}; "
        f"vocab_size={model_cfg.vocab_size}"
    )

    # DataLoaders
    train_loader = create_dataloader(loaded.train, dataset_cfg, shuffle=True, batch_size=train_cfg.batch_size)
    valid_loader = create_dataloader(loaded.valid, dataset_cfg, shuffle=False, batch_size=train_cfg.batch_size)
    test_loader = create_dataloader(loaded.test, dataset_cfg, shuffle=False, batch_size=train_cfg.batch_size)

    # Model and trainer
    model = ModelFactory.create_model(model_cfg, device)
    trainer = MgRIATrainer(model_cfg, train_cfg, device)
    # Use trainer's internal model for consistency
    trainer.model = model

    # Train or load from checkpoint if available
    ckpt_dir = dataset_cfg.results_path / f"{dataset_cfg.dataset_type.value}_checkpoints"
    best_ckpt = ckpt_dir / 'best_model.pt'
    if best_ckpt.exists():
        logger.info(f"Found existing checkpoint {best_ckpt}, loading and skipping training...")
        trainer.model = ModelFactory.load_model(best_ckpt, model_cfg, device)
        history = {'train_loss': [], 'eval_loss': []}
    else:
        if args.skip_train:
            logger.info("skip-train set and no checkpoint found; proceeding without training")
            history = {'train_loss': [], 'eval_loss': []}
        else:
            logger.info(f"Starting training on {dataset_cfg.dataset_type.value}...")
            history = trainer.train(train_loader, valid_loader, save_dir=ckpt_dir)

    # Evaluate on test set with top-k metrics
    logger.info("Evaluating on test set (top-k metrics)...")
    eval_metrics = DATASET_METRICS.get(dataset_cfg.dataset_type, train_cfg.eval_metrics)
    metrics = compute_topk_metrics(trainer.model, test_loader, device, eval_metrics)

    # Save results
    out_dir = dataset_cfg.results_path / dataset_cfg.dataset_type.value / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved results to {out_dir}")

    # Optionally plot loss curves
    try:
        import matplotlib.pyplot as plt
        train_losses = history.get('train_loss', [])
        eval_losses = history.get('eval_loss', [])
        if (isinstance(train_losses, list) and len(train_losses) > 0) or (isinstance(eval_losses, list) and len(eval_losses) > 0):
            plt.figure(figsize=(6,4))
            if len(train_losses) > 0:
                plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
            if len(eval_losses) > 0:
                plt.plot(range(1, len(eval_losses)+1), eval_losses, label='eval_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"Loss Curve - {dataset_cfg.dataset_type.value}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'loss_curve.png', dpi=150)
            plt.close()
            logger.info(f"Saved loss curve to {out_dir / 'loss_curve.png'}")
    except Exception as e:
        logger.warning(f"Could not generate loss curve: {e}")

    print(f"\n=== {dataset_cfg.dataset_type.value.capitalize()} Results ===")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
