from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import importlib
import sys

import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.config import GEMORNA_3UTR_Config, three_prime_utr_vocab
from src.data.conditional_gemorna_dataset import (
    build_conditional_vocab,
    ConditionalGEMORNADataset,
    conditional_collate_fn,
)
from src.data.prepare_conditional_splits import prepare_conditional_splits
from src.models.gemorna_runtime import build_gemorna_3utr_model
from src.utils.utils_utr import UTR_
import torch.nn as nn


def _ensure_legacy_module_aliases():
    sys.modules['config'] = importlib.import_module('src.config')
    sys.modules['tokenization'] = importlib.import_module('src.tokenization')
    sys.modules['models.gemorna_utr'] = importlib.import_module('src.models.gemorna_utr')
    sys.modules['utils.utils_utr'] = importlib.import_module('src.utils.utils_utr')


def load_model_for_finetuning(checkpoint_path: str, device: str | None = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_obj = torch.device(device)
    _ensure_legacy_module_aliases()

    model = build_gemorna_3utr_model()
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device_obj)
    return model


def build_conditional_finetune_model(checkpoint_path: str, device: str | None = None):
    """Build 3'UTR model with control-tag vocab added into unused vocab slots."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_obj = torch.device(device)
    _ensure_legacy_module_aliases()

    conditional_vocab = build_conditional_vocab(three_prime_utr_vocab)

    cfg = deepcopy(GEMORNA_3UTR_Config())
    cfg.vocab_size = max(conditional_vocab.values()) + 1
    model = UTR_(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model_state = model.state_dict()

    # Load all same-shaped tensors directly.
    for key, tensor in state_dict.items():
        if key in model_state and model_state[key].shape == tensor.shape:
            model_state[key] = tensor

    # For vocab-sized weights, copy the pretrained rows into the new bigger matrices.
    for key in ['transformer.wte.weight', 'lm_head.weight']:
        if key in state_dict and key in model_state:
            rows = min(state_dict[key].shape[0], model_state[key].shape[0])
            model_state[key][:rows] = state_dict[key][:rows]

    model.load_state_dict(model_state)
    model.to(device_obj)
    return model, conditional_vocab


def freeze_lower_transformer_blocks(model: torch.nn.Module, freeze_ratio: float = 0.5):
    n_blocks = len(model.transformer['h'])
    freeze_n = int(n_blocks * freeze_ratio)
    for i, block in enumerate(model.transformer['h']):
        if i < freeze_n:
            for p in block.parameters():
                p.requires_grad = False
    return freeze_n


def build_conditional_dataloader(
    csv_path: str,
    vocab: dict,
    batch_size: int = 4,
    shuffle: bool = True,
    max_length: int = 1024,
):
    dataset = ConditionalGEMORNADataset(csv_path=csv_path, vocab=vocab, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=conditional_collate_fn,
    )


def run_conditional_finetuning(
    train_csv: str,
    checkpoint_path: str,
    save_path: str,
    val_csv: str | None = None,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    max_length: int = 1024,
    device: str | None = None,
    max_steps_per_epoch: int | None = None,
    log_every: int = 20,
    make_split_if_missing: bool = True,
    freeze_lower_ratio: float = 0.0,
    use_length_aux_loss: bool = False,
    length_aux_weight: float = 0.2,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_obj = torch.device(device)

    if val_csv is None:
        val_csv = str(Path(train_csv).with_name(Path(train_csv).stem + '_val.csv'))

    if make_split_if_missing and not Path(val_csv).exists():
        split_train = str(Path(train_csv).with_name(Path(train_csv).stem + '_train.csv'))
        prepare_conditional_splits(train_csv, split_train, val_csv)
        train_csv = split_train

    model, conditional_vocab = build_conditional_finetune_model(checkpoint_path, device=device)
    frozen_blocks = 0
    if freeze_lower_ratio > 0:
        frozen_blocks = freeze_lower_transformer_blocks(model, freeze_ratio=freeze_lower_ratio)
    length_head = None
    if use_length_aux_loss:
        length_head = nn.Linear(model.config.n_embd, 1).to(device_obj)
    params = [p for p in model.parameters() if p.requires_grad]
    if length_head is not None:
        params += list(length_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    train_loader = build_conditional_dataloader(
        csv_path=train_csv,
        vocab=conditional_vocab,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
    )
    val_loader = None
    if val_csv and Path(val_csv).exists():
        val_loader = build_conditional_dataloader(
            csv_path=val_csv,
            vocab=conditional_vocab,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_length,
        )

    history = []
    best_val_loss = None
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(device_obj)
            labels = batch['labels'].to(device_obj)
            optimizer.zero_grad()
            if use_length_aux_loss:
                _, lm_loss, hidden_states = model(input_ids, targets=labels, return_hidden_states=True)
                prompt_token_count = torch.tensor(batch['prompt_token_count'], dtype=torch.long, device=device_obj)
                target_token_length = torch.tensor(batch['utr3_token_length'], dtype=torch.float32, device=device_obj)
                pooled = []
                for bi in range(hidden_states.shape[0]):
                    idx = max(0, min(int(prompt_token_count[bi].item()) - 1, hidden_states.shape[1] - 1))
                    pooled.append(hidden_states[bi, idx, :])
                pooled = torch.stack(pooled, dim=0)
                pred_len = length_head(pooled).squeeze(-1)
                length_loss = torch.nn.functional.mse_loss(pred_len, target_token_length)
                loss = lm_loss + length_aux_weight * length_loss
            else:
                _, loss = model(input_ids, targets=labels)
                lm_loss = loss
                length_loss = None
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            total_loss += loss_value
            steps += 1

            if batch_idx == 1 or batch_idx % log_every == 0:
                extra = ''
                if use_length_aux_loss and length_loss is not None:
                    extra = f' lm_loss={float(lm_loss.item()):.6f} len_loss={float(length_loss.item()):.6f}'
                print(
                    f'[Epoch {epoch} Step {batch_idx}] '
                    f'loss={loss_value:.6f}{extra} '
                    f'batch_shape={tuple(input_ids.shape)} '
                    f'sample_tags={batch["pgk_tag"][0]} {batch["len_tag"][0]}'
                )

            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                break

        avg_loss = total_loss / max(steps, 1)
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch['input_ids'].to(device_obj)
                    val_labels = val_batch['labels'].to(device_obj)
                    if use_length_aux_loss:
                        _, vlm_loss, vhidden = model(val_input_ids, targets=val_labels, return_hidden_states=True)
                        vprompt_token_count = torch.tensor(val_batch['prompt_token_count'], dtype=torch.long, device=device_obj)
                        vtarget_token_length = torch.tensor(val_batch['utr3_token_length'], dtype=torch.float32, device=device_obj)
                        vpooled = []
                        for bi in range(vhidden.shape[0]):
                            idx = max(0, min(int(vprompt_token_count[bi].item()) - 1, vhidden.shape[1] - 1))
                            vpooled.append(vhidden[bi, idx, :])
                        vpooled = torch.stack(vpooled, dim=0)
                        vpred_len = length_head(vpooled).squeeze(-1)
                        vlen_loss = torch.nn.functional.mse_loss(vpred_len, vtarget_token_length)
                        vloss = vlm_loss + length_aux_weight * vlen_loss
                    else:
                        _, vloss = model(val_input_ids, targets=val_labels)
                    val_total += float(vloss.item())
                    val_steps += 1
            val_loss = val_total / max(val_steps, 1)
            model.train()

        history.append({'epoch': epoch, 'train_loss': avg_loss, 'val_loss': val_loss, 'steps': steps})
        print(f'[Epoch {epoch}] train_loss={avg_loss:.6f} val_loss={val_loss if val_loss is not None else "NA"} steps={steps}')

        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            best_path = str(Path(save_path).with_name(Path(save_path).stem + '_best.pt'))
            save_finetuned_checkpoint(
                model=model,
                save_path=best_path,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    'control_vocab': conditional_vocab,
                    'history': history,
                    'base_vocab_size': max(three_prime_utr_vocab.values()) + 1,
                    'best_val_loss': best_val_loss,
                },
            )

    save_finetuned_checkpoint(
        model=model,
        save_path=save_path,
        optimizer=optimizer,
        epoch=num_epochs,
        extra={
            'control_vocab': conditional_vocab,
            'history': history,
            'base_vocab_size': max(three_prime_utr_vocab.values()) + 1,
            'best_val_loss': best_val_loss,
            'train_csv': train_csv,
            'val_csv': val_csv,
            'freeze_lower_ratio': freeze_lower_ratio,
            'frozen_blocks': frozen_blocks,
            'use_length_aux_loss': use_length_aux_loss,
            'length_aux_weight': length_aux_weight,
        },
    )
    return {'save_path': save_path, 'history': history, 'control_vocab': conditional_vocab, 'best_val_loss': best_val_loss, 'train_csv': train_csv, 'val_csv': val_csv, 'freeze_lower_ratio': freeze_lower_ratio, 'frozen_blocks': frozen_blocks, 'use_length_aux_loss': use_length_aux_loss}


def save_finetuned_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {'model': model.state_dict()}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        payload['epoch'] = epoch
    if extra:
        payload.update(extra)

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_file)
