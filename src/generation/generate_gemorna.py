from __future__ import annotations

from pathlib import Path
from typing import Optional
import importlib
import sys

import torch
import torch.nn.functional as F

from src.data.conditional_gemorna_dataset import build_conditional_vocab, CONTROL_TAGS
from src.models.gemorna_runtime import load_generation_model
from src.tokenization import tokenize_seq
from src.config import three_prime_utr_vocab, eos_token, init_token, GEMORNA_3UTR_Config
from src.utils.utils_utr import UTR_


def _ensure_legacy_module_aliases():
    sys.modules['config'] = importlib.import_module('src.config')
    sys.modules['tokenization'] = importlib.import_module('src.tokenization')
    sys.modules['models.gemorna_utr'] = importlib.import_module('src.models.gemorna_utr')
    sys.modules['utils.utils_utr'] = importlib.import_module('src.utils.utils_utr')


def generate_sequences(
    checkpoint_path: str,
    utr_length: str,
    output_path: Optional[str] = None,
    shared_library_path: Optional[str] = None,
    device: Optional[str] = None,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = load_generation_model(
        checkpoint_path=checkpoint_path,
        shared_library_path=shared_library_path,
        device=device,
    )

    generated = model.gen('3utr', vocab, torch.device(device), utr_length)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(str(generated), encoding='utf-8')

    return generated


def encode_prompt_only(prompt: str, vocab: dict[str, int]) -> list[int]:
    tokens = [init_token] + tokenize_seq(prompt)
    ids = []
    for tok in tokens:
        if tok not in vocab:
            raise KeyError(f'Unknown token in prompt: {tok}')
        ids.append(vocab[tok])
    return ids


def load_conditional_model_and_vocab(checkpoint_path: str, device: str | None = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device_obj = torch.device(device)
    _ensure_legacy_module_aliases()

    conditional_vocab = build_conditional_vocab(three_prime_utr_vocab)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    if isinstance(ckpt, dict) and ckpt.get('control_vocab'):
        conditional_vocab = ckpt['control_vocab']

    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    cfg = GEMORNA_3UTR_Config()
    if 'transformer.wte.weight' in state:
        cfg.vocab_size = state['transformer.wte.weight'].shape[0]
    model = UTR_(cfg).to(device_obj)
    model.load_state_dict(state)
    model.eval()
    return model, conditional_vocab, ckpt


def sample_next_token(logits, temperature=1.0, top_k=None):
    next_token_logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
    probs = F.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def decode_conditional_sequence(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: Optional[str] = None,
):
    model, conditional_vocab, _ = load_conditional_model_and_vocab(checkpoint_path, device=device)
    device_obj = next(model.parameters()).device

    input_ids = encode_prompt_only(prompt, conditional_vocab)
    generated = torch.tensor([input_ids], dtype=torch.long, device=device_obj)
    eos_id = conditional_vocab[eos_token]
    blocked_ids = [conditional_vocab[tag] for tag in CONTROL_TAGS if tag in conditional_vocab]
    blocked_ids += [conditional_vocab[init_token]]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            for bid in blocked_ids:
                next_token_logits[:, bid] = -float('inf')
            next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
            generated = torch.cat([generated, next_token], dim=1)
            if int(next_token.item()) == eos_id:
                break

    inv_vocab = {v: k for k, v in conditional_vocab.items()}
    decoded_tokens = [inv_vocab.get(int(tok), '<unk>') for tok in generated[0].tolist()]
    return decoded_tokens


def extract_generated_sequence(tokens: list[str]) -> str:
    prompt_tags = set(CONTROL_TAGS) | {init_token, eos_token}
    seq_tokens = []
    for tok in tokens:
        if tok in prompt_tags:
            continue
        if tok == '<unk>':
            continue
        seq_tokens.append(tok)
    return ''.join(seq_tokens).replace(' ', '')


def tokens_to_text(tokens: list[str]) -> str:
    return ' '.join(tokens)


def build_control_prompt(pgk: str | None = None, length: str | None = None, prompt: str | None = None) -> str:
    if prompt:
        return prompt.strip()

    parts = []
    if pgk is not None:
        pgk = pgk.lower().strip()
        mapping = {'high': '<pgk_high>', 'low': '<pgk_low>'}
        if pgk not in mapping:
            raise ValueError("--pgk must be one of: high, low")
        parts.append(mapping[pgk])

    if length is not None:
        length = length.lower().strip()
        mapping = {'short': '<len_short>', 'medium': '<len_medium>', 'long': '<len_long>'}
        if length not in mapping:
            raise ValueError("--len must be one of: short, medium, long")
        parts.append(mapping[length])

    if not parts:
        raise ValueError('Provide either --prompt or at least one of --pgk / --len for conditional generation.')
    return ' '.join(parts)


def run_conditional_generation(
    checkpoint_path: str,
    pgk: str | None = None,
    length: str | None = None,
    prompt: str | None = None,
    output_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 20,
    device: str | None = None,
):
    control_prompt = build_control_prompt(pgk=pgk, length=length, prompt=prompt)
    tokens = decode_conditional_sequence(
        checkpoint_path=checkpoint_path,
        prompt=control_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    sequence = extract_generated_sequence(tokens)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(sequence + '\n', encoding='utf-8')

    return {
        'prompt': control_prompt,
        'tokens': tokens,
        'tokens_text': tokens_to_text(tokens),
        'sequence': sequence,
        'generated_token_length': len(tokens),
        'decoded_sequence_length': len(sequence),
    }
