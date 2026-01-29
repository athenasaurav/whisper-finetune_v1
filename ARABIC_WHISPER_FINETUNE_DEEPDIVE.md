# Arabic Whisper Fine-Tuning: Deep Dive

End-to-end guide for fine-tuning **whisper-large-v3-turbo** on **Arabic** using HF datasets, on a **2× A100 80GB** server, comparing **full**, **LoRA**, and **QLoRA** with **WER/CER/NLL/ECE** and **latency**.

---

## 1. Codebase Architecture (End-to-End)

### 1.1 Entry Point and Config Flow

```
finetune.py --config configs/example_config.yaml
    → read_config() → YAML dict
    → main(config)
```

- **`utils.read_config`**: Loads YAML. Required keys: `model`, `dataset`, `lr_scheduler`, `optimizer`, `training`, `augmentation`, `seed`, `save_dir`.
- **`model.init_name`**: Passed to `whisper.load_model(init_name)` (openai-whisper). For large-v3-turbo use `large-v3-turbo` if supported by your openai-whisper version; otherwise a local `.pt` path.
- **`n_mels`**: In `get_dataloader` and in `AudioDataset`: `128 if "v3" in config["model"]["init_name"] else 80`. large-v3-turbo uses **128** mels.

### 1.2 Model Loading (`finetune.py` + `model_utils.py`)

1. **`whisper.load_model(config["model"]["init_name"], device="cpu")`**  
   - openai-whisper: downloads or loads from cache. Names like `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. `large-v3-turbo` may exist in recent openai-whisper (e.g. 20240930); if not, use a path to a converted `.pt`.

2. **Precision**  
   - `config["model"]["bfloat16"]` is deprecated. Use `training.mixed_precision_training` and `training.mp_dtype` (fp16/bf16). Model stays in FP32; autocast uses `mp_dtype` in the forward.

3. **Gradient checkpointing**  
   - If `gradient_checkpointing_encoder`: replace `model.encoder` with `CheckpointedStochasticAudioEncoder` (stochastic depth + `torch.utils.checkpoint.checkpoint`).
   - If `gradient_checkpointing_decoder`: replace `model.decoder` with `CheckpointedStochasticTextDecoder`.
   - After replacing, **reload** weights with `load_model_and_set_heads(model, init_name)` because the new modules are randomly initialized.

4. **Frozen encoder/decoder**  
   - `train_only_decoder=True` → `disable_all_grads(whisper_model.encoder)`.
   - `train_only_encoder=True` → `disable_all_grads(whisper_model.decoder)`.

5. **LoRA**  
   - If `config["model"]["lora"]`:
     - `apply_lora(model, lora_config, train_only_decoder, train_only_encoder)`:
       - Uses **minLoRA**: `add_lora(module, lora_config)` where `lora_config` maps `whisper.model.Linear` to `LoRAParametrization.from_linear(rank=, lora_alpha=, lora_dropout_p=)`.
       - `lora_dropout` in YAML is renamed to `lora_dropout_p` for minLoRA.
     - `disable_all_but_parametrized_grads(model)`: only parameters with `"lora"` in the name keep `requires_grad=True`.

6. **Device**  
   - `model.to("cuda")`.

7. **Deep SpecAugment**  
   - If `augmentation.deep_spec_augment.apply`: `register_deep_spec_augment_hooks` on encoder `attn_ln` (all blocks except last). In the hook: TimeMasking + FrequencyMasking on the normalized features.

### 1.3 Data Pipeline

#### `data/utils.process_dataset(dataset_names, select_n_per_ds, split_name, groupby_col, ..., return_sizes=False)`

- **Load**: `load_hf_dataset(path_or_name)` → `load_from_disk` if path exists, else `load_dataset` (HF Hub).
- **Split**: `dataset[split_name]` (e.g. `"train"`, `"validation"`). Falls back to `"train"` or first key if `split_name` missing.
- **Columns**: renames `sentence` → `text`. If `language` missing, `dataset.map(add_fixed_value, fn_kwargs={"col_name":"language","fixed_value": default_language})`. For Arabic set **`dataset.default_language: "ar"`**.
- **Sampling**:  
  - If `N = select_n_per_ds[i]` and `groupby_col[i]` in columns: groupby on that column, sample `N` per group (with replace if group size &lt; N).  
  - Else if `N`: `dataset.select(np.arange(min(N, len(dataset))))` (deterministic; for true random, `dataset.shuffle().select` would need to be added).  
  - If `N` is `null`: use all.
- **Output**: `concatenate_datasets(processed_datasets)`. If `return_sizes=True`, also returns `dataset_sizes` for `WarmupDatasetSampler`.

#### `data/data_loader.py`

- **`AudioDataset`**  
  - **Required columns**: `audio`, `text`, `language`.  
  - **`audio`**: `record["audio"]["array"]` (numpy, 16 kHz assumed by `log_mel_spectrogram`).  
  - **Validation**: For each index, checks `audio` loads and `text` is `str`; invalid indices are skipped (`valid_indices`).

- **`__getitem__`**:
  1. `no_timestamps = no_timestamp_training or (random < no_timestamps_rate)`
  2. **Prompt**: With `prompt_use_rate`, build `[sot_prev] + encode(prompt)[-max_prompt_length:]`; else `[]`.
  3. **Text**: `_encode_text_with_timestamps` or `_encode_text_without_timestamps` (timestamp tokens `<|0.00|>`…`<|30.00|>`). `_get_partial_segment_start` for possible audio trim.
  4. **Special**: `[sot, lang_token, transcribe, no_timestamps?]` or `[sot, no_speech]` if text empty.
  5. **Decoder input**: `prompt + special + text_tokens`; **decoder output**: `[-100]*len(prompt)-1 + special[1:] + text_tokens + [eot]` (prompt part ignored in CE).
  6. **Audio**: Pad to `N_SAMPLES` (30 s), `log_mel_spectrogram(..., n_mels)` (80 or 128), `pad_or_trim` to `N_FRAMES`. Optional: audio aug (`apply_baseline_aug`, `apply_office_aug`, `apply_advanced_aug`), SpecAugment (time/freq mask, time warp), extremes freq masking.
  7. Return `(mel, decoder_input, decoder_output)`.

- **`collate_fn`**: Pads `decoder_input` with 0, `decoder_output` with -100.

- **`WarmupDatasetSampler`**: If `warmup_dataset_idx` is set, for the first `warmup_steps * batch_size` samples it yields only indices from that concatenated sub-dataset; then switches to all indices.

#### Tokenizer

- **`get_tokenizer(multilingual=True, language="...", task="transcribe")`** (openai-whisper).  
- For Arabic, **`language` must be `"ar"`**. This is used for the `&lt;|ar|&gt;` token in the decoder. The codebase previously hardcoded `"de"`; it should come from **`config["dataset"].get("language","ar")`** or a dedicated `model.language`.

### 1.4 Training Step (`model_utils.train_step`)

- **Accumulation**: `accum_grad_steps`; loss scaled by `1/accum_grad_steps`.
- **Autocast**: `device_type="cuda"`, `dtype=mp_dtype` (fp16 or bf16). **GradScaler** only if fp16 (bf16 does not use scaler).
- **Forward**: `model.embed_audio(x)`, `model.logits(y_in, audio_features=audio_features)`, `F.cross_entropy(logits.transpose(1,2), y_out, label_smoothing=label_smoothing)`.
- **Backward**: `scaler.scale(loss).backward()` or `loss.backward()`.
- **LoRA debug**: At eval steps, `log_lora_debug_info` (grad norms, etc.) after backward, before `optimizer.step()`; `lora_tracker.snapshot()` before `optimizer.step()`.
- **Grad clip**: `clip_grad_norm_(model.parameters(), max_grad_norm)`.
- **Optimizer**: `scaler.step(optimizer)` or `optimizer.step()`; `lr_scheduler.step()` (for fp16, only when scale did not increase after `scaler.step`).
- **`optimizer.zero_grad()`** at end.

### 1.5 Optimizer (`model/optimizer.py`)

- **`get_optimizer(model, optimizer_conf, is_lora_run)`**  
  - Filters `model.parameters()` to `requires_grad`.  
  - **8-bit**: `optimizer.8bit=True` → `bitsandbytes.optim.AdamW8bit` or `Adam8bit`. **QLoRA**: The codebase does **not** implement 4-bit base model; bitsandbytes here is only for **8-bit optimizer**.  
  - With LoRA, 8-bit optimizer can quantize small grads to zero; a warning is printed; for stability, `8bit: false` is sometimes better for LoRA.

### 1.6 Scheduler (`model/scheduler.py`)

- **`get_scheduler(optimizer, s_conf, train_steps)`**  
  - `linear`, `cosine`, `cosine_with_restarts`, `cosine_with_warmup_restarts`, `cosine_with_warmup_restarts_chill`.  
  - `warmup_steps`: if &lt; 1, treated as ratio of `train_steps` in `finetune.py`.

### 1.7 Evaluation (`eval/evaluator.py`, `eval/metrics.py`)

- **`evaluate_single_dataset(model, dataloader, dataset_name, t_config, tokenizer)`**  
  - `model.eval()`, `torch.no_grad()`.  
  - For each batch: `model(x, y_in)` (full forward), `argmax` for pred token IDs.  
  - For each sample: decode pred/ref, **normalize** with `normalize_text(..., **VOCAB_SPECS["v0"])`. For **Arabic**, `VOCAB_SPECS["arabic"]` must be used so Arabic script is not stripped.  
  - **jiwer**: `compute_measures` (WER), `cer`.  
  - **`compute_token_metrics`**: NLL, avg log-prob, entropy, per-token confidence and correct.  
  - **`compute_ece`**: ECE from (confidences, correct) over all tokens.  
  - **`aggregate_dataset_metrics`** → `DatasetMetrics` (wer, cer, mean_token_nll, avg_log_prob, mean_token_entropy, ece, num_samples, per_utterance).

- **`evaluate_multiple_datasets`**  
  - Loops over `dataloaders`, collects `DatasetMetrics`.  
  - **`compute_macro_average`**: unweighted mean of each metric across datasets → `macro_wer`, `macro_cer`, etc.  
  - **`log_metrics_to_wandb`**: per-dataset and macro; `step` set explicitly.

- **Latency**  
  - Not implemented by default. We add **inference timing** (e.g. wall-clock for `model(x, y_in)` over the val set) and report **samples/sec** or **ms/sample** and **RTF** (real-time factor) if we have total audio duration.

### 1.8 Main Loop (`finetune.main_loop`)

- `wandb.watch(model, log="all")`.
- **Initial eval** at step 0; `min_wer = macro_metrics["macro_wer"]`.
- Loop `step = 1..train_steps`:
  - `train_step(...)` → `train_loss`; `wandb.log(Learning rate, Train loss)`.
  - Assert `train_loss < max_train_loss`.
  - Every `val_steps` or at last step: `evaluate_multiple_datasets` → save **best** if `macro_wer < min_wer`; optionally `save_all_checkpoints` → `step{step}.pt`.
- `save_model(model, f"{save_dir}/last_model.pt")`.
- If `upload_models_to_wandb`: `wandb.save` for last/best.

### 1.9 Save / Merge

- **`save_model`**: `model.half()`, `torch.save({model_state_dict, dims}, path)`.
- **`merge_lora_weights.py`**: Load base model, `apply_lora` with same `lora_config`, load checkpoint `state_dict` with `strict=False`, `merge_lora` (remove parametrizations, merge into weight), `save_model`. **Must** use the same `init_name` and `lora_config` as training. The script is currently hardcoded to `large-v3` and a fixed `lora_config`; it should be updated to accept `--config` and read `model.init_name` and `model.lora_config` (or path to `lora_config.json` saved by `finetune.py`).

---

## 2. What You Need for Arabic

### 2.1 Datasets (HF)

- **Mozilla Common Voice**: e.g. `mozilla-foundation/common_voice_17_0` (or your preferred MCV Arabic), split `train`/`validation`; columns: `audio`, `text`; `sentence`→`text`; often has `locale` or you add `language` = `"ar"`.
- **FLEURS**: `google/fleurs`, config `ar_eg` (or other Arabic); `transcription` → map to `text`; add `language: "ar"` if missing.
- **GigaSpeech-style or other Arabic ASR** in HF Datasets format with `audio` (dict with `array`, `sampling_rate`) and `text`.

Ensure `audio` is 16 kHz or resampled in the dataset. The code does not resample; `log_mel_spectrogram` assumes 16 kHz.

### 2.2 Config Additions

- **`dataset.default_language`**: `"ar"`. Used when `language` column is missing in `process_dataset`.
- **`dataset.language`** (or `model.language`): `"ar"`. Used for `get_tokenizer(..., language="ar")` and indirectly for the decoder lang token. Both `finetune.py` and `evaluator.py` must use this.

### 2.3 Text Normalization for Eval (WER/CER)

- **`eval/utils.VOCAB_SPECS`** is Latin-centric (`char_vocab` = ascii + äöü, etc.). Using `"v0"` would remove Arabic characters and break WER/CER.
- Add **`VOCAB_SPECS["arabic"]`**:
  - `char_vocab`: `None` to mean “do not filter by character set” (only apply `char_lookup` and whitespace).
  - `char_lookup`: optional (e.g. normalize alef variants, tatweel, etc.); can start as `{}`.
  - `transform_lowercase`: `False`.
- **`normalize_text`**: if `char_vocab is None`, skip the `"".join([c for c in text if c in char_vocab])` step.
- **`evaluator`**: take a `vocab_spec` or `language` from config; for `"ar"` use `VOCAB_SPECS["arabic"]`, else `"v0"` or another.

---

## 3. Full vs LoRA vs QLoRA

### 3.1 Full Fine-Tuning

- **Config**: `model.lora: false` (or absent). No `lora_config`. All parameters (or only encoder/decoder if `train_only_*`) are trained.
- **Memory**: Highest. For large-v3-turbo (encoder 32L, decoder 4L, ~1280d) on 1× A100 80GB: gradient checkpointing + fp16/bf16 + batch_size 16–32 and `accum_grad_steps` 4–8 is typically fine. 2× A100 80GB: with current code (single process, one GPU), only one GPU is used; to use both, **DDP** (or similar) must be implemented.

### 3.2 LoRA

- **Config**: `model.lora: true`, `model.lora_config: { rank: 16, lora_alpha: 32, lora_dropout: 0.1 }`.
- **Mechanics**: `apply_lora` adds LoRA to `Linear` in encoder, decoder, or both; freezes non-LoRA params. Trainable params are a small fraction of the full model.
- **Memory**: Lower than full; allows larger batch or less gradient checkpointing. 8-bit optimizer is optional; sometimes `8bit: false` is more stable for LoRA.

### 3.3 QLoRA (4-bit Base + LoRA)

- **Definition**: Base model in **4-bit** (NF4/FP4 via bitsandbytes), then LoRA on top. This repo uses **openai-whisper** and `whisper.load_model`, which does **not** support 4-bit. **QLoRA is not implemented.**
- **What exists**: Only **8-bit optimizer** (AdamW8bit) from bitsandbytes. That is **not** QLoRA.
- **To add QLoRA** you would need to:
  - Switch to **HuggingFace** `WhisperForConditionalGeneration` and `BitsAndBytesConfig(load_in_4bit=True, ...)`.
  - Re-implement or adapt: dataloader (HF Whisper uses different input format), `embed_audio`/`logits` interface, training step, eval, and possibly `merge_lora` for HF.
- **Practical choice for now**: Use **LoRA** (and optionally 8-bit optimizer) as the “light” option. We provide an **`arabic_qlora.yaml`** that sets LoRA + 8-bit optimizer as a “memory-reduced” setup, with a clear comment that **true QLoRA (4-bit base) is not supported** in this codebase.

---

## 4. 2× A100 80GB

- **`sc_sbatch.sh`**: `--partition=a100-80g`, `--gres=gpu:1`. For 2 GPUs: `--gres=gpu:2`. **`finetune.py` does not use DDP**; it only uses `cuda` (one device). With `gres=gpu:2`, both GPUs are allocated but only one is used unless you add DDP.
- **Recommendation**:
  - **1× A100 80GB**: Enough for large-v3-turbo full (with gradient checkpointing and moderate batch) and LoRA. Use `--gres=gpu:1`.
  - **2× A100 80GB**: Either (a) run two **independent** jobs (e.g. full and LoRA) in parallel, each with `--gres=gpu:1`, or (b) implement DDP and use `--gres=gpu:2` for a single distributed run. We provide an **`sc_sbatch_2gpu.sh`** template with `--gres=gpu:2` and a note that DDP is required to exploit both GPUs.

---

## 5. Metrics and Latency (as in README)

### 5.1 Error Metrics (Already in Code)

- **WER, CER**: jiwer, on normalized text. For Arabic use `VOCAB_SPECS["arabic"]`.
- **NLL**: `mean_token_nll` (mean CE per token over valid positions).
- **Log-prob, entropy, ECE**: `compute_token_metrics` and `compute_ece` in `metrics.py`.  
All are computed per dataset and **macro-averaged** across validation sets.

### 5.2 Latency (To Be Added)

- **Inference time**: In `evaluate_single_dataset`, around the `model(x, y_in)` call, accumulate **wall-clock time** (excluding dataloader). Report:
  - **Total eval time** (s)
  - **Samples / s**
  - **ms / sample**
  - If we have total duration of the val audios (e.g. from `N_FRAMES` or fixed 30 s per chunk): **RTF** = (inference_time / total_audio_duration).

We add an optional `t_config["measure_inference_latency"]` and, when True, log `val/{dataset}_samples_per_sec`, `val/{dataset}_ms_per_sample`, `val/macro_ms_per_sample`, and optionally RTF.

---

## 6. Config Templates

### 6.1 `configs/arabic_full.yaml`

- `model.init_name: large-v3-turbo`, `lora` absent.
- `dataset`: `train_datasets`, `val_datasets` (placeholders for your Arabic HF sets), `default_language: "ar"`, `language: "ar"`.
- `training`: `gradient_checkpointing_encoder/decoder: true`, `mixed_precision_training: true`, `mp_dtype: fp16` (or bf16), `train_only_decoder/encoder: false`.
- `optimizer.8bit: false` or `true` (your choice).
- `batch_size`/`accum_grad_steps` tuned so that 1× A100 80GB does not OOM (e.g. 16–24 and 4–6 for full).

### 6.2 `configs/arabic_lora.yaml`

- Same as above, plus `model.lora: true`, `model.lora_config: { rank: 16, lora_alpha: 32, lora_dropout: 0.1 }`.
- Can use somewhat larger `batch_size` than full. `optimizer.8bit` optional.

### 6.3 `configs/arabic_qlora.yaml`

- Same as LoRA. **Comment in file**: true QLoRA (4-bit base) is not supported; this is **LoRA + 8-bit optimizer** for reduced memory. `optimizer.8bit: true`.

---

## 7. Running on the 2× A100 80GB Node

### 7.1 One GPU (recommended if no DDP)

```bash
# .env: WANDB_*, HF_*, etc.
sbatch sc_sbatch.sh configs/arabic_full.yaml
sbatch sc_sbatch.sh configs/arabic_lora.yaml
sbatch sc_sbatch.sh configs/arabic_qlora.yaml
```

Adjust `sc_sbatch.sh` so `--partition` and `--gres` match your cluster (e.g. `a100-80g` and `gpu:1`).

### 7.2 Two GPUs (when DDP is implemented)

- Use `sc_sbatch_2gpu.sh` with `--gres=gpu:2` and `srun`/`torchrun` to start the script with `CUDA_VISIBLE_DEVICES` or `RANK`/`WORLD_SIZE`. Not done in this codebase yet.

### 7.3 Merging LoRA (for LoRA/QLoRA configs)

After training, `merge_lora_weights.py` should be called with the **same** `init_name` and `lora_config` as in the run. With the planned `--config`:

```bash
python src/whisper_finetune/scripts/merge_lora_weights.py \
  --input output/<job_id>/best_model.pt \
  --config configs/arabic_lora.yaml \
  --output output/<job_id>/best_model_merged.pt
```

---

## 8. Comparison Table (What to Fill After Runs)

| Setup        | WER ↓ | CER ↓ | NLL ↓ | ECE ↓ | ms/sample ↓ | Notes                    |
|-------------|-------|-------|-------|-------|-------------|--------------------------|
| Full        |       |       |       |       |             | 1× A100 80GB, grad ckpt  |
| LoRA        |       |       |       |       |             | Same hardware            |
| LoRA+8bit   |       |       |       |       |             | “arabic_qlora” config    |

True QLoRA (4-bit) would require HF Whisper + 4-bit loading; not in this repo.

---

## 9. File-Level Reference

| Feature              | File                         | Symbol / logic |
|----------------------|------------------------------|----------------|
| Config, seed, steps  | `utils.py`                   | `read_config`, `set_seed`, `calculate_training_steps`, `calculate_val_steps` |
| Model load, LoRA     | `model/model_utils.py`, `model/lora.py` | `load_model_and_set_heads`, `apply_lora`, `disable_all_but_parametrized_grads` |
| Data                 | `data/utils.py`, `data/data_loader.py` | `process_dataset`, `load_hf_dataset`, `add_fixed_value`, `AudioDataset`, `get_dataloader` |
| Train step           | `model/model_utils.py`       | `train_step` |
| Eval, metrics        | `eval/evaluator.py`, `eval/metrics.py`, `eval/utils.py` | `evaluate_multiple_datasets`, `evaluate_single_dataset`, `compute_wer`, `compute_token_metrics`, `compute_ece`, `aggregate_dataset_metrics`, `compute_macro_average`, `normalize_text`, `VOCAB_SPECS` |
| Optimizer / Scheduler| `model/optimizer.py`, `model/scheduler.py` | `get_optimizer`, `get_scheduler` |
| Main loop            | `scripts/finetune.py`        | `main`, `main_loop` |
| Merge LoRA           | `scripts/merge_lora_weights.py` | `main` (needs `--config` and `lora_config` from config or JSON) |
| SLURM                | `sc_sbatch.sh`, `multi_submit.sh` | Partition, gres, conda, `finetune.py --config` |

---

## 10. Summary of Code/Config Changes

1. **`dataset.default_language`**: In `process_dataset`, when adding `language`, use `default_language` (from config, default `"ar"` for Arabic).
2. **`dataset.language`** (or `model.language`): Pass into `get_tokenizer(..., language=...)` in `finetune.py` and in `evaluate_single_dataset` (or derive from config).
3. **`eval/utils`**: Add `VOCAB_SPECS["arabic"]`; in `normalize_text`, if `char_vocab is None`, skip the character filter.
4. **Evaluator**: Use `VOCAB_SPECS[config.get("eval_vocab_spec","v0")]` or `"arabic"` when `language=="ar"`; add `measure_inference_latency` and timing to log `ms_per_sample`, `samples_per_sec`, and optionally RTF.
5. **`merge_lora_weights.py`**: `--config` to read `model.init_name` and `model.lora_config` (or `save_dir/lora_config.json`); remove hardcoded `large-v3` and `lora_config`.
6. **Configs**: `arabic_full.yaml`, `arabic_lora.yaml`, `arabic_qlora.yaml` with Arabic `train_datasets`/`val_datasets` placeholders, `default_language`/`language`, and LoRA only for lora/qlora.
7. **`sc_sbatch.sh`**: Keep `--gres=gpu:1` for single-GPU; add `sc_sbatch_2gpu.sh` with `--gres=gpu:2` and a note about DDP.

Once these are in place, you can run the three regimes on 1× (or 2× with two jobs) A100 80GB and fill the comparison table with WER, CER, NLL, ECE, and latency.
