from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.tokenization import numericalize


CONTROL_TAGS = [
    '<pgk_high>',
    '<pgk_mid>',
    '<pgk_low>',
    '<len_short>',
    '<len_medium>',
    '<len_long>',
]


def build_conditional_vocab(base_vocab: Dict[str, int]) -> Dict[str, int]: #컨트롤 택을 추가한 vocab을 만들어주는 함수
    vocab = dict(base_vocab)
    next_id = max(vocab.values()) + 1
    for tag in CONTROL_TAGS:
        if tag not in vocab:
            vocab[tag] = next_id
            next_id += 1
    return vocab


@dataclass
class ConditionalSample:
    ensembl_gene_id: str
    text: str
    pgk_tag: str
    len_tag: str
    utr3_length: int


class ConditionalGEMORNADataset(Dataset):
    def __init__(self, csv_path: str, vocab: Dict[str, int], max_length: int = 1024):
        self.df = pd.read_csv(csv_path)
        required = ['ensembl_gene_id', 'utr3', 'pgk_tag', 'len_tag', 'gemorna_control_text']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f'Missing required columns in {csv_path}: {missing}')

        self.df = self.df.dropna(subset=['utr3', 'pgk_tag', 'len_tag', 'gemorna_control_text']).copy()
        self.df['utr3'] = self.df['utr3'].astype(str)
        self.df['gemorna_control_text'] = self.df['gemorna_control_text'].astype(str)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ids = numericalize(row['gemorna_control_text'], self.vocab)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        prompt_text = f"{row['pgk_tag']} {row['len_tag']}"
        prompt_ids = numericalize(prompt_text, self.vocab)
        prompt_token_count = max(1, len(prompt_ids) - 1)  # exclude trailing eos from prompt-only numericalize
        return {
            'input_ids': input_ids,
            'labels': labels,
            'ensembl_gene_id': row['ensembl_gene_id'],
            'pgk_tag': row['pgk_tag'],
            'len_tag': row['len_tag'],
            'utr3_length': int(row.get('utr3_length', len(row['utr3']))),
            'utr3_token_length': int((len(str(row['utr3'])) + 2) // 3),
            'prompt_token_count': int(prompt_token_count),
        }


def conditional_collate_fn(batch: List[dict], pad_token_id: int = 0):
    max_len = max(item['input_ids'].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    meta = {
        'ensembl_gene_id': [],
        'pgk_tag': [],
        'len_tag': [],
        'utr3_length': [],
        'utr3_token_length': [],
        'prompt_token_count': [],
    }

    for i, item in enumerate(batch):
        n = item['input_ids'].shape[0]
        input_ids[i, :n] = item['input_ids']
        labels[i, :n] = item['labels']
        meta['ensembl_gene_id'].append(item['ensembl_gene_id'])
        meta['pgk_tag'].append(item['pgk_tag'])
        meta['len_tag'].append(item['len_tag'])
        meta['utr3_length'].append(item['utr3_length'])
        meta['utr3_token_length'].append(item['utr3_token_length'])
        meta['prompt_token_count'].append(item['prompt_token_count'])

    return {
        'input_ids': input_ids,
        'labels': labels,
        **meta,
    }
