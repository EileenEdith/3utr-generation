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


def _extract_state_dict(checkpoint: Dict[str, Any] | Any) -> Dict[str, torch.Tensor]: # checkpoint이 dict이고 'model' 키가 있으면 그걸로 모델 state_dict로 사용. checkpoint이 dict이지만 'model' 키가 없으면 checkpoint 자체가 state_dict라고 가정. 그 외에는 지원하지 않는 포맷으로 간주해서 에러 발생.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError('Unsupported checkpoint format for GEMORNA 3UTR model.')


def load_pretrained_gemorna_3utr( # fine-tuning된 모델과 vocab 불러오기. vocab은 checkpoint에 없으면 3utr vocab에서 빌드. checkpoint에 있으면 그걸로 덮어쓰기.
    checkpoint_path: str | Path,
    device: str | torch.device = 'cpu',
) -> Tuple[UTR_, Dict[str, int]]:
    device = torch.device(device)
    model = build_gemorna_3utr_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint) # 실제 weight만 꺼냄
    model.load_state_dict(state_dict) # 빈 모델에 학습된 파라미터를 넣음
    model.to(device)
    model.eval()
    return model, three_prime_utr_vocab


def load_generation_model( # .so에 들어있는 generation용 UTR 클래스 불러와서 모댈 구성
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
