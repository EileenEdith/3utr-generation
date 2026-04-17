from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import importlib
import sys

import torch
import pandas as pd

from src.config import three_prime_utr_vocab
from src.data.conditional_gemorna_dataset import build_conditional_vocab, CONTROL_TAGS
from src.generation.generate_gemorna import decode_conditional_sequence, extract_generated_sequence
from src.generation.quality_eval import gc_fraction, repetition_fraction, shannon_diversity
from src.generation.finetune_gemorna import build_conditional_finetune_model


def _ensure_legacy_module_aliases():
    sys.modules['config'] = importlib.import_module('src.config')
    sys.modules['tokenization'] = importlib.import_module('src.tokenization')
    sys.modules['models.gemorna_utr'] = importlib.import_module('src.models.gemorna_utr')
    sys.modules['utils.utils_utr'] = importlib.import_module('src.utils.utils_utr')


def inspect_control_vocab_source(checkpoint_path: str) -> dict[str, Any]:
    _ensure_legacy_module_aliases()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    built_vocab = build_conditional_vocab(three_prime_utr_vocab)
    checkpoint_vocab = ckpt.get('control_vocab') if isinstance(ckpt, dict) else None
    effective_vocab = checkpoint_vocab if checkpoint_vocab is not None else built_vocab
    source = 'checkpoint[control_vocab]' if checkpoint_vocab is not None else 'build_conditional_vocab(three_prime_utr_vocab)'
    return {
        'source_of_truth': source,
        'built_vocab_size': len(built_vocab),
        'checkpoint_vocab_size': len(checkpoint_vocab) if checkpoint_vocab is not None else None,
        'effective_vocab_size': len(effective_vocab),
        'control_tag_ids': {tag: effective_vocab.get(tag) for tag in CONTROL_TAGS},
    }


def inspect_weight_initialization(pretrained_checkpoint: str, finetuned_checkpoint: str) -> dict[str, Any]:
    _ensure_legacy_module_aliases()
    pre = torch.load(pretrained_checkpoint, map_location='cpu')
    ft = torch.load(finetuned_checkpoint, map_location='cpu')
    pre_state = pre['model'] if isinstance(pre, dict) and 'model' in pre else pre
    ft_state = ft['model'] if isinstance(ft, dict) and 'model' in ft else ft

    compared = {}
    keys = ['transformer.wte.weight', 'lm_head.weight', 'transformer.wpe.weight']
    for key in keys:
        if key in pre_state and key in ft_state:
            a = pre_state[key]
            b = ft_state[key]
            overlap_rows = min(a.shape[0], b.shape[0]) if a.ndim > 1 else None
            if overlap_rows is not None:
                diff = (a[:overlap_rows] - b[:overlap_rows]).abs().mean().item()
            else:
                diff = (a - b).abs().mean().item()
            compared[key] = {
                'pre_shape': list(a.shape),
                'ft_shape': list(b.shape),
                'mean_abs_diff_overlap': diff,
            }

    total_params = sum(v.numel() for v in ft_state.values())
    return {
        'compared': compared,
        'total_checkpoint_params': total_params,
        'note': 'If mean_abs_diff_overlap is non-zero, pretrained weights were modified during fine-tuning.',
    }


def inspect_trainable_params(checkpoint_path: str, device: str = 'cpu') -> dict[str, Any]:
    model, vocab = build_conditional_finetune_model(checkpoint_path, device=device)
    rows = []
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        rows.append({'name': name, 'shape': list(p.shape), 'requires_grad': bool(p.requires_grad), 'numel': int(n)})
    return {
        'total_params': total,
        'trainable_params': trainable,
        'trainable_fraction': trainable / total if total else 0.0,
        'all_trainable': trainable == total,
        'sample': rows[:12],
    }


def run_prompt_comparison(
    checkpoint_path: str,
    prompts: list[str],
    num_samples: int = 20,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 20,
    device: str | None = None,
    constrained: bool = True,
):
    rows = []
    for prompt in prompts:
        for i in range(num_samples):
            tokens = decode_conditional_sequence(
                checkpoint_path=checkpoint_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
                constrained=constrained,
            )
            seq = extract_generated_sequence(tokens)
            rows.append({
                'prompt': prompt,
                'sample_idx': i,
                'generated_token_length': len(tokens),
                'decoded_sequence_length': len(seq),
                'gc_fraction': gc_fraction(seq),
                'repetition_fraction': repetition_fraction(seq),
                'sequence': seq,
            })
    df = pd.DataFrame(rows)
    summary = df.groupby('prompt').agg(
        n=('sequence', 'size'),
        unique_outputs=('sequence', 'nunique'),
        mean_token_length=('generated_token_length', 'mean'),
        std_token_length=('generated_token_length', 'std'),
        mean_seq_length=('decoded_sequence_length', 'mean'),
        std_seq_length=('decoded_sequence_length', 'std'),
        mean_gc=('gc_fraction', 'mean'),
        mean_repetition=('repetition_fraction', 'mean'),
    ).reset_index()
    div = []
    for prompt, g in df.groupby('prompt'):
        div.append({'prompt': prompt, 'shannon_diversity': shannon_diversity(g['sequence'].tolist())})
    summary = summary.merge(pd.DataFrame(div), on='prompt', how='left')
    return df, summary
