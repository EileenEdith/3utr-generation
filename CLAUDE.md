# CLAUDE.md

## Project Overview

Conditional 3'UTR sequence generation framework based on the pretrained **GEMORNA 3'UTR model**. GEMORNA is fine-tuned with control tags (`<pgk_high>`, `<pgk_mid>`, `<pgk_low>`, `<len_short>`, `<len_medium>`, `<len_long>`) to steer generation toward 3'UTR sequences with desired expression properties.

## Environment

```bash
conda activate utrgen
```

Python 3.10, PyTorch 2.2, scikit-learn, pandas, biopython, sentencepiece.

---

## Current Pipeline

### Classification (completed)

PGK classification is **already done**. Results are stored at:

```
src/classification/merged.json
```

- Format: list of `{ensembl_transcript_id, label, sequence}`
- Labels: `high` (6,864), `mid` (38,660), `low` (38,046) — total 83,570
- Classifier: `src/classification/model_mlp_final.py` (4-layer MLP on 1024-dim HydraRNA embeddings)
- **No need to re-run classification.**

### Generation (current focus)

**Goal:** Label-specific continued fine-tuning of GEMORNA, starting from existing fine-tuned weights (not from the pretrained base). Each label class (high/mid/low) gets its own fine-tuned model to capture class-internal 3'UTR patterns while preserving general 3'UTR properties.

#### Existing fine-tuned checkpoints (starting points for continued fine-tuning):

```
weights/finetuned/
├── gemorna_3utr_conditional_epoch3_best.pt   # best val loss, all-label conditional
├── gemorna_3utr_conditional_epoch3.pt
├── lenaux_freeze50_best.pt                   # with length aux loss + 50% frozen
├── lenaux_unfrozen_best.pt                   # with length aux loss, fully unfrozen
└── ...
```

#### Workflow:

```
src/classification/merged.json
        │  {sequence, label} for 83,570 sequences
        ▼
[prepare label-specific training CSVs]
        │  filter by label → per-label CSV with columns:
        │  ensembl_gene_id, utr3, pgk_tag, len_tag, gemorna_control_text
        ▼
src/generation/finetune_gemorna.py :: run_conditional_finetuning()
        │  checkpoint_path = existing fine-tuned weights (NOT pretrained base)
        │  → continued fine-tuning per label class
        ▼
weights/finetuned/{label}_model.pt
        ▼
src/generation/generate_gemorna.py :: run_conditional_generation()
        │  label-specific checkpoint → generate 3'UTR sequences
        ▼
results/generated_sequences/
```

---

## Architecture

### Model

- `src/utils/utils_utr.py::UTR_` — GPT-style transformer
- Config: `src/config.py::GEMORNA_3UTR_Config` — 12 layers, 12 heads, 288 embed dim, block_size 1024
- Base vocab: 431 codon-level tokens (`src/config.py::three_prime_utr_vocab`), extended to 437 with 6 control tags

### Tokenization

- RNA codon triplets (3-char chunks), not character-level
- `src/tokenization.py`: `tokenize_seq` (splitting), `numericalize` (token→id, prepends `<sos>`, appends `<eos>`)
- During generation, control tags and `<sos>` are blocked from being sampled

### Control Tags

Defined in `src/data/conditional_gemorna_dataset.py::CONTROL_TAGS`:
```python
['<pgk_high>', '<pgk_mid>', '<pgk_low>', '<len_short>', '<len_medium>', '<len_long>']
```

`build_conditional_vocab()` appends these to the base vocab. `build_control_prompt()` in `generate_gemorna.py` maps CLI args to tag strings.

### Vocab Expansion

`build_conditional_finetune_model()` increases `vocab_size` beyond 431 for control tags. When loading a fine-tuned checkpoint, `load_conditional_model_and_vocab()` reads `vocab_size` from `state['transformer.wte.weight'].shape[0]`.

### Legacy Module Aliases

`libg2m.so` imports `config`, `tokenization`, etc. using flat paths. `_ensure_legacy_module_aliases()` in both `finetune_gemorna.py` and `generate_gemorna.py` injects `src.*` modules under legacy names into `sys.modules`. Must be called before loading the `.so`.

---

## Module Map

| Path | Role |
|------|------|
| `main.py` | CLI entry point: routes to `generate_sequences` or `run_conditional_generation` |
| `src/config.py` | Model configs, base vocab, init/eos tokens; also `config` alias for `libg2m.so` |
| `src/tokenization.py` | `tokenize_seq`, `numericalize` |
| `src/models/gemorna_runtime.py` | `load_pretrained_gemorna_3utr`, `load_generation_model`, `build_gemorna_3utr_model` |
| `src/models/gemorna_utr.py` | Pure-Python UTR model (inference without `.so`) |
| `src/utils/utils_utr.py` | `UTR_` — the GPT model class |
| `src/data/conditional_gemorna_dataset.py` | `CONTROL_TAGS`, `build_conditional_vocab`, `ConditionalGEMORNADataset`, `conditional_collate_fn` |
| `src/data/prepare_conditional_splits.py` | Stratified train/val CSV split |
| `src/generation/finetune_gemorna.py` | `run_conditional_finetuning`, `build_conditional_finetune_model`, `freeze_lower_transformer_blocks` |
| `src/generation/generate_gemorna.py` | `run_conditional_generation`, `decode_conditional_sequence`, `build_control_prompt` |
| `src/generation/quality_eval.py` | `evaluate_prompts` — batch eval with GC%, repetition, Shannon diversity |
| `src/generation/validation_diagnostics.py` | Validation diagnostics |
| `src/classification/model_mlp_final.py` | `EmbMLP` classifier (already run, results in `merged.json`) |
| `src/classification/merged.json` | Classification results: 83,570 entries with sequence + label |

---

## Fine-tuning API

```python
from src.generation.finetune_gemorna import run_conditional_finetuning

run_conditional_finetuning(
    train_csv="data/processed/high_training_table.csv",
    checkpoint_path="weights/finetuned/gemorna_3utr_conditional_epoch3_best.pt",
    save_path="weights/finetuned/high_model.pt",
    num_epochs=3,
    freeze_lower_ratio=0.5,
    use_length_aux_loss=True,
)
```

Training CSV requires columns: `ensembl_gene_id`, `utr3`, `pgk_tag`, `len_tag`, `gemorna_control_text`.

`ConditionalGEMORNADataset` reads `gemorna_control_text` as the full input text (control tags + sequence), applies causal LM tokenization.

---

## CLI

```bash
# Conditional generation
python main.py --checkpoint weights/finetuned/model.pt --pgk high --len short
python main.py --checkpoint weights/finetuned/model.pt --prompt "<pgk_high> <len_short>"

# Original GEMORNA generation (requires libg2m.so)
python main.py --checkpoint weights/pretrained/gemorna_3utr.pt --utr_length medium
```

Flags: `--checkpoint`, `--output`, `--device`, `--max_new_tokens` (256), `--temperature` (0.8), `--top_k` (20).

---

## File Locations

| Resource | Path |
|----------|------|
| Pretrained GEMORNA | `weights/pretrained/gemorna_3utr.pt` |
| Fine-tuned checkpoints | `weights/finetuned/` |
| Shared library | `src/models/gemorna_shared/libg2m.so` |
| Classification results | `src/classification/merged.json` |
| Generated sequences | `results/generated_sequences/` |

---

## Data Facts

- PGK distribution: median=255, mean=1177, max=225K (right-skewed, use log1p)
- `merged.json`: 83,570 entries — high: 6,864, mid: 38,660, low: 38,046
- Sequences are DNA (ACGT), variable length
