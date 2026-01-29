# Step-by-Step: Full Fine-Tuning Whisper Large-V3-Turbo on SADA22 Saudi Arabic

This guide runs **full** fine-tuning of **whisper-large-v3-turbo** on the [60k-SADA22_Saudi](https://huggingface.co/datasets/MahmoudIbrahim/60k-SADA22_Saudi) dataset (~40k rows, Saudi Arabic, `audio` + `cleaned_text`).

---

## 1. Environment

- **Python 3.11+**
- **GPU**: 1× A100 80GB (or similar; 24GB+ with smaller batch possible)

```bash
cd whisper-finetune
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e .
```

---

## 2. Dataset

- **Name**: `MahmoudIbrahim/60k-SADA22_Saudi`
- **Splits**: Only `train` (~40.2k rows)
- **Columns used**: `audio` (array + sampling_rate), `cleaned_text` (transcription)
- **Train/val**: The config uses `train_val_split_fraction: 0.1` so 90% train / 10% val are created from `train` automatically.

No manual split needed; the code does it.

---

## 3. Config

Use **`configs/sada22_full.yaml`**. Main options:

| Option | Value | Meaning |
|--------|--------|--------|
| `model.init_name` | `large-v3-turbo` | Base model |
| `dataset.train_datasets` | `[MahmoudIbrahim/60k-SADA22_Saudi]` | Single dataset |
| `dataset.text_column` | `cleaned_text` | Use this column as transcript (renamed to `text` internally) |
| `dataset.train_val_split_fraction` | `0.1` | 10% of data → validation |
| `dataset.default_language` / `language` | `ar` | Arabic |
| `dataset.batch_size` | `16` | Tune down if OOM (e.g. 8) |
| `training.epochs` | `2` | Increase if you want more epochs |

---

## 4. Run Training

**Local (single GPU):**

```bash
python src/whisper_finetune/scripts/finetune.py --config configs/sada22_full.yaml
```

**SLURM (e.g. 1× A100 80GB):**

```bash
# Copy and edit .env from .env-template (WANDB_*, HF_TOKEN if needed)
sbatch sc_sbatch.sh configs/sada22_full.yaml
```

Outputs go to `output/<job_id_or_timestamp>/`:
- `best_model.pt` (lowest validation WER)
- `last_model.pt`
- Training logs and W&B (if configured).

---

## 5. Optional: Adjust Batch Size / Epochs

If you hit **out-of-memory**:

- In `configs/sada22_full.yaml` set `dataset.batch_size: 8` (or 4) and/or increase `training.accum_grad_steps` to keep effective batch size.

If you want **more training**:

- Increase `training.epochs` (e.g. 3 or 4).

---

## 6. After Training

- **Best checkpoint**: `output/<run>/best_model.pt`
- **Merge LoRA** (only for LoRA runs): not needed for full fine-tuning.
- **Convert to faster-whisper**: use the repo’s conversion script with `best_model.pt` and the turbo tokenizer/config (e.g. `whisper_v3_turbo_utils/`).

---

## Summary

1. Install deps: `pip install -e .`
2. Use dataset: `MahmoudIbrahim/60k-SADA22_Saudi`, columns `audio` + `cleaned_text`.
3. Use config: `configs/sada22_full.yaml` (sets `text_column: cleaned_text`, `train_val_split_fraction: 0.1`).
4. Run: `python src/whisper_finetune/scripts/finetune.py --config configs/sada22_full.yaml` (or via SLURM).
5. Check `output/<run>/best_model.pt` and W&B for WER/CER and latency.
