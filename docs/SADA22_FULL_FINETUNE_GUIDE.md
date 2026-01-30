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

## 1.5. .env and HF token

Create a **`.env`** file in the repo root so scripts (and SLURM jobs) can load your Hugging Face token and other settings. **`.env` is gitignored** — never commit it.

**1. Copy the template:**

```bash
cp .env-template .env
```

**2. Get a Hugging Face token (if needed):**

- Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Create a token (read for datasets, write if you will upload models).
- Copy the token (it starts with `hf_`).

**3. Edit `.env` and set `HF_TOKEN`:**

```bash
# In .env, set (replace with your token; no quotes):
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- **Public datasets only** (e.g. `MahmoudIbrahim/60k-SADA22_Saudi`): you can leave `HF_TOKEN=` empty; download often works without a token.
- **Gated or private datasets**, or **uploading models to the Hub**: set `HF_TOKEN` to your token.

**4. Optional: adjust paths and W&B**

- `HF_HOME`, `HF_DATASETS_CACHE`: change if you want a different cache dir (e.g. on a shared drive).
- `HF_DATASETS_OFFLINE=0`, `TRANSFORMERS_OFFLINE=0`: keep `0` so the dataset can be downloaded; set `1` only for fully offline runs.
- `WANDB_*`: set `WANDB_API_KEY` and your `WANDB_ENTITY`/`WANDB_PROJECT` if you use Weights & Biases.

**5. How it’s loaded**

- **SLURM** (`sc_sbatch.sh`, `sc_sbatch_2gpu.sh`): `export $(cat .env | xargs)` loads `.env` before running the training script.
- **Local**: either run in a shell that has already run `export $(cat .env | xargs)` or `set -a; source .env; set +a` (Linux/Mac), or use a tool that loads `.env` (e.g. `python-dotenv` if you add it to the script).

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
| **`upload.repo_id`** | **`YOUR_HF_USERNAME/whisper-large-v3-turbo-sada22`** | **Where to set the name for training + Hub push** (see below) |

### Where to set the name (training + Hub push)

Set **one name** in the config and it is used for **auto-pushing the best model** to your Hugging Face Hub:

- In **`configs/sada22_full.yaml`**, edit the **`upload`** section:
  - **`upload.repo_id`**: Your Hub repo ID, e.g. **`athenasaurav/whisper-large-v3-turbo-sada22`** (replace with your HF username and the repo name you want).
  - **`upload.push_to_hub: true`**: Enables auto-push after training.
- After training finishes, the script uses **`HF_TOKEN`** from your `.env` and pushes **`best_model.pt`** (and optionally the faster-whisper version) to **`https://huggingface.co/<upload.repo_id>`**. You do not need to find or run a separate upload step.

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

- **Best checkpoint**: saved locally at `output/<run>/best_model.pt`.
- **Auto-push to Hub**: if **`upload.push_to_hub: true`** and **`upload.repo_id`** are set in the config and **`HF_TOKEN`** is set in `.env`, the **best** model is pushed automatically to **`https://huggingface.co/<upload.repo_id>`** (both `.pt` and faster-whisper format by default). No need to find the best model or run a separate upload script.
- **Merge LoRA** (only for LoRA runs): not needed for full fine-tuning.

---

## 7. Run inference

Use your fine-tuned checkpoint to transcribe audio (single file or a folder):

```bash
# Single file (prints transcript to stdout)
python -m whisper_finetune.scripts.inference \
  --checkpoint output/20260130_090017/best_model.pt \
  --audio path/to/audio.wav \
  --language ar

# Folder of audio files; write one .txt per file
python -m whisper_finetune.scripts.inference \
  --checkpoint output/<your_run>/best_model.pt \
  --audio path/to/audio_folder/ \
  --language ar \
  --output-dir ./transcripts
```

- **`--checkpoint`**: path to `best_model.pt` from your training run (e.g. `output/20260130_090017/best_model.pt`).
- **`--audio`**: path to a single file (`.wav`, `.mp3`, etc.) or a directory (all supported files under it are transcribed).
- **`--language ar`**: use for Arabic; change for other languages.
- **`--base-model large-v3-turbo`**: must match the model you trained (default is correct for SADA22 config).
- **`--output-dir`**: optional; if set, writes one `.txt` per audio file with the transcript.

---

## Troubleshooting: WER goes to 100% or stays very high (model collapse)

If **WER jumps to 1.0 (100%)** and **NLL / log-prob / entropy are almost identical** at every eval step, the model has likely **collapsed**: it is outputting the same (or nearly same) prediction for every input.

- **Cause:** Learning rate **too high** for full fine-tuning of large-v3-turbo (~807M params). `lr: 2e-4` is often used for LoRA or small models but can destabilize full fine-tuning.
- **Fix:** Lower the learning rate in `configs/sada22_full.yaml`:
  - Set **`optimizer.params.lr`** to **`5.0e-5`** (or try `1e-5` if still unstable).
  - Optionally increase **`lr_scheduler.warmup_steps`** to **`0.15`** (15% warmup).
- **Then:** Restart training from scratch (do not resume the collapsed run). You should see WER start around ~0.77 and **decrease** over steps instead of jumping to 1.0.

Language code (`ar`) and Arabic normalization are already correct; the issue is training stability, not language or eval.

---

## Troubleshooting: Dataset download fails (IncompleteRead / ChunkedEncodingError)

If the first run fails with **IncompleteRead** or **ChunkedEncodingError** while downloading the dataset (e.g. connection dropped mid-download):

1. **Retry** – The script retries up to 3 times with backoff. Often a second run succeeds.
2. **Clear the cache and retry** – A partial download can leave bad files in the Hugging Face cache. Remove the cached dataset and run again:
   ```bash
   rm -rf ~/.cache/huggingface/datasets/MahmoudIbrahim___60k-SADA22_Saudi
   python src/whisper_finetune/scripts/finetune.py --config configs/sada22_full.yaml
   ```
   (Cache path may vary; check `HF_DATASETS_CACHE` in `.env` if you set it.)
3. **Stable network** – Use a stable connection or VPN; large datasets (e.g. SADA22 ~5GB) are sensitive to drops.

---

## Summary

1. Install deps: `pip install -e .`
2. Set **`.env`** with **`HF_TOKEN`** (and optionally W&B).
3. In **`configs/sada22_full.yaml`**, set **`upload.repo_id`** to your Hub repo (e.g. `athenasaurav/whisper-large-v3-turbo-sada22`). Same name is used for auto-push.
4. Use dataset: `MahmoudIbrahim/60k-SADA22_Saudi`, columns `audio` + `cleaned_text`.
5. Run: `python src/whisper_finetune/scripts/finetune.py --config configs/sada22_full.yaml` (or via SLURM).
6. Best model is saved locally and, if `upload.push_to_hub` and `HF_TOKEN` are set, pushed automatically to `https://huggingface.co/<upload.repo_id>`.
