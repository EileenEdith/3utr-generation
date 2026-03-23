from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.config import GEMORNA_3UTR_Config, three_prime_utr_vocab
from src.utils.utils_utr import UTR_


def build_gemorna_3utr_model() -> UTR_:
    return UTR_(GEMORNA_3UTR_Config())


def _extract_state_dict(checkpoint: Dict[str, Any] | Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError('Unsupported checkpoint format for GEMORNA 3UTR model.')


def load_pretrained_gemorna_3utr(
    checkpoint_path: str | Path,
    device: str | torch.device = 'cpu',
) -> Tuple[UTR_, Dict[str, int]]:
    device = torch.device(device)
    model = build_gemorna_3utr_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, three_prime_utr_vocab


def load_generation_model(
    checkpoint_path: str | Path,
    shared_library_path: str | Path | None = None,
    device: str | torch.device = 'cpu',
):
    device = torch.device(device)
    if shared_library_path is None:
        shared_library_path = Path(__file__).resolve().parent / 'gemorna_shared' / 'libg2m.so'
    shared_library_path = Path(shared_library_path).resolve()
    shared_dir = str(shared_library_path.parent)
    src_root = str(Path(__file__).resolve().parents[1])

    for path in [shared_dir, src_root]:
        if path not in sys.path:
            sys.path.insert(0, path)

    module = importlib.import_module(shared_library_path.stem)
    if not hasattr(module, 'UTR'):
        raise AttributeError('Loaded shared library does not expose UTR generation class.')

    model = module.UTR(GEMORNA_3UTR_Config())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, three_prime_utr_vocab
