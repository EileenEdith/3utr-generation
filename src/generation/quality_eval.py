from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from src.generation.generate_gemorna import decode_conditional_sequence, extract_generated_sequence


def shannon_diversity(sequences: list[str]) -> float:
    concat = ''.join(sequences)
    if not concat:
        return 0.0
    counts = Counter(concat)
    total = sum(counts.values())
    import math
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in {'G', 'C'})
    return gc / len(seq)


def repetition_fraction(seq: str) -> float:
    if len(seq) < 2:
        return 0.0
    repeated = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i-1])
    return repeated / (len(seq) - 1)


def evaluate_prompts(
    checkpoint_path: str,
    prompts: list[str],
    num_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 20,
    device: str | None = None,
):
    rows = []
    for prompt in prompts:
        for sample_idx in range(num_samples):
            tokens = decode_conditional_sequence(
                checkpoint_path=checkpoint_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            seq = extract_generated_sequence(tokens)
            rows.append({
                'prompt': prompt,
                'sample_idx': sample_idx,
                'generated_token_length': len(tokens),
                'decoded_sequence_length': len(seq),
                'gc_fraction': gc_fraction(seq),
                'repetition_fraction': repetition_fraction(seq),
                'generated_sequence': seq,
            })
    df = pd.DataFrame(rows)
    summary = df.groupby('prompt').agg(
        n=('generated_sequence', 'size'),
        mean_token_length=('generated_token_length', 'mean'),
        mean_seq_length=('decoded_sequence_length', 'mean'),
        std_seq_length=('decoded_sequence_length', 'std'),
        mean_gc=('gc_fraction', 'mean'),
        mean_repetition=('repetition_fraction', 'mean'),
    ).reset_index()
    div_rows = []
    for prompt, g in df.groupby('prompt'):
        div_rows.append({'prompt': prompt, 'shannon_diversity': shannon_diversity(g['generated_sequence'].tolist())})
    div_df = pd.DataFrame(div_rows)
    summary = summary.merge(div_df, on='prompt', how='left')
    return df, summary
