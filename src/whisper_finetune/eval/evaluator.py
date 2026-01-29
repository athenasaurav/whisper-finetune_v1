"""
Multi-dataset evaluator for Whisper fine-tuning.
Evaluates model on multiple validation datasets and computes comprehensive metrics.
"""

import time
from typing import Dict, List, Tuple

import torch

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from whisper import Whisper
from whisper.tokenizer import get_tokenizer

from whisper_finetune.eval.metrics import (
    DatasetMetrics,
    PerUtteranceMetrics,
    aggregate_dataset_metrics,
    compute_cer_batch,
    compute_macro_average,
    compute_token_metrics,
    compute_wer,
)
from whisper_finetune.eval.utils import VOCAB_SPECS, normalize_text


def _print_eval_metrics(
    dataset_metrics: List[DatasetMetrics],
    macro_metrics: Dict[str, float],
    step: int | None = None,
) -> None:
    """Print comprehensive metrics (WER, CER, NLL, log-prob, entropy, ECE) with rich tables if available."""
    step_label = f" — Step {step}" if step is not None else ""
    title = f"Comprehensive Metrics 🆕{step_label}"
    if _RICH_AVAILABLE:
        console = Console()
        # Per-dataset table
        table = Table(
            title=f"[bold cyan]{title}[/] — Per-dataset",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
        )
        table.add_column("Dataset", style="cyan")
        table.add_column("Samples", justify="right", style="dim")
        table.add_column("WER ↓", justify="right", style="red")
        table.add_column("CER ↓", justify="right", style="red")
        table.add_column("NLL ↓", justify="right", style="yellow")
        table.add_column("Log-prob ↑", justify="right", style="green")
        table.add_column("Entropy", justify="right", style="dim")
        table.add_column("ECE ↓", justify="right", style="yellow")
        table.add_column("ms/sample", justify="right", style="dim")
        for dm in dataset_metrics:
            ms = f"{dm.inference_ms_per_sample:.1f}" if dm.inference_ms_per_sample is not None else "—"
            table.add_row(
                dm.dataset_name,
                str(dm.num_samples),
                f"{dm.wer:.4f}",
                f"{dm.cer:.4f}",
                f"{dm.mean_token_nll:.4f}",
                f"{dm.avg_log_prob:.4f}",
                f"{dm.mean_token_entropy:.4f}",
                f"{dm.ece:.4f}",
                ms,
            )
        console.print(table)
        # Macro panel
        macro_lines = []
        for k, v in macro_metrics.items():
            if v is not None:
                macro_lines.append(f"  [bold]{k}[/]: [cyan]{v:.4f}[/]")
            else:
                macro_lines.append(f"  [bold]{k}[/]: N/A")
        console.print(
            Panel(
                "\n".join(macro_lines),
                title="[bold green]Macro averages (unweighted)[/]",
                border_style="green",
            )
        )
    else:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        for dm in dataset_metrics:
            print(f"\n[Dataset: {dm.dataset_name}]")
            print(f"  Samples: {dm.num_samples}")
            print(f"  WER: {dm.wer:.4f}  CER: {dm.cer:.4f}")
            print(f"  NLL: {dm.mean_token_nll:.4f}  Log-prob: {dm.avg_log_prob:.4f}")
            print(f"  Entropy: {dm.mean_token_entropy:.4f}  ECE: {dm.ece:.4f}")
            if dm.inference_ms_per_sample is not None:
                print(f"  Inference: {dm.inference_ms_per_sample:.2f} ms/sample")
        print(f"\n{'='*60}")
        print("MACRO AVERAGES")
        print(f"{'='*60}")
        for k, v in macro_metrics.items():
            print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")


@torch.no_grad()
def evaluate_single_dataset(
    model: Whisper,
    dataloader: DataLoader,
    dataset_name: str,
    t_config: dict,
    tokenizer=None,
) -> DatasetMetrics:
    """
    Evaluate model on a single dataset with comprehensive metrics.

    Args:
        model: Whisper model to evaluate
        dataloader: DataLoader for the validation dataset
        dataset_name: Name of the dataset (for logging)
        t_config: Training configuration dict
        tokenizer: Tokenizer (if None, creates a new one)

    Returns:
        DatasetMetrics object with all computed metrics
    """
    model.eval()

    # Read variables from t_config
    mixed_precision_training = t_config.get("mixed_precision_training", True)
    mp_dtype = torch.float16 if t_config.get("mp_dtype", "fp16") == "fp16" else torch.bfloat16

    # Get tokenizer
    ds_language = t_config.get("dataset_language", "en")
    if tokenizer is None:
        tokenizer = get_tokenizer(multilingual=True, language=ds_language, task="transcribe")

    vocab_spec = t_config.get("eval_vocab_spec", "v0")
    vocab_spec = VOCAB_SPECS.get(vocab_spec, VOCAB_SPECS["v0"])

    measure_latency = t_config.get("measure_inference_latency", True)
    total_forward_s = 0.0
    n_forward = 0

    per_utterance_metrics = []

    for x, y_in, y_out in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        x, y_in, y_out = x.to(model.device), y_in.to(model.device), y_out.to(model.device)

        if measure_latency:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=mixed_precision_training, dtype=mp_dtype):
            logits = model(x, y_in)
        if measure_latency:
            torch.cuda.synchronize()
            total_forward_s += time.perf_counter() - t0
            n_forward += x.size(0)

        # Convert logits to token IDs
        pred_token_ids = torch.argmax(logits, dim=-1)

        # Process each sample in the batch
        for i in range(logits.size(0)):
            sample_logits = logits[i]
            sample_pred_ids = pred_token_ids[i]
            sample_true_ids = y_out[i]

            # Decode predictions and references
            pred_tokens = [
                id
                for id in sample_pred_ids.cpu().tolist()
                if id not in tokenizer.special_tokens.values() and id != -100
            ]
            true_tokens = [
                id
                for id in sample_true_ids.cpu().tolist()
                if id not in tokenizer.special_tokens.values() and id != -100
            ]

            pred_text = tokenizer.decode(pred_tokens)
            true_text = tokenizer.decode(true_tokens)

            # Skip empty references
            if true_text.strip() == "":
                continue

            # Normalize texts (use vocab_spec: v0 for Latin, arabic for Arabic)
            pred_normalized = normalize_text(pred_text, **vocab_spec)
            true_normalized = normalize_text(true_text, **vocab_spec)

            # Compute WER and CER
            from jiwer import compute_measures

            wer_measures = compute_measures(true_normalized, pred_normalized)
            wer_val = wer_measures["wer"]

            from jiwer import cer

            cer_val = cer(true_normalized, pred_normalized)

            # Compute token-level metrics
            mean_nll, avg_log_prob, mean_entropy, confidences, correct = compute_token_metrics(
                sample_logits, sample_true_ids, sample_pred_ids
            )

            # Store per-utterance metrics
            per_utterance_metrics.append(
                PerUtteranceMetrics(
                    prediction=pred_normalized,
                    reference=true_normalized,
                    wer=wer_val,
                    cer=cer_val,
                    token_nll=mean_nll,
                    avg_log_prob=avg_log_prob,
                    token_entropy=mean_entropy,
                    token_confidences=confidences,
                    token_correct=correct,
                )
            )

    # Aggregate metrics for the dataset
    dataset_metrics = aggregate_dataset_metrics(per_utterance_metrics, dataset_name)

    if measure_latency and n_forward > 0:
        dataset_metrics.inference_ms_per_sample = 1000.0 * total_forward_s / n_forward
        dataset_metrics.inference_samples_per_sec = n_forward / total_forward_s

    return dataset_metrics


def evaluate_multiple_datasets(
    model: Whisper,
    dataloaders: Dict[str, DataLoader],
    t_config: dict,
    step: int | None = None,
) -> Tuple[List[DatasetMetrics], Dict[str, float]]:
    """
    Evaluate model on multiple validation datasets.

    Args:
        model: Whisper model to evaluate
        dataloaders: Dictionary mapping dataset names to their DataLoaders
        t_config: Training configuration dict
        step: Optional training step (for display: 0=initial, N=periodic, final=last step)

    Returns:
        Tuple of:
            - List of DatasetMetrics (one per dataset)
            - Dictionary of macro-averaged metrics
    """
    tokenizer = get_tokenizer(
        multilingual=True, language=t_config.get("dataset_language", "en"), task="transcribe"
    )

    all_dataset_metrics = []

    for dataset_name, dataloader in dataloaders.items():
        if _RICH_AVAILABLE:
            Console().print(f"[dim]Evaluating dataset: {dataset_name}[/]")
        else:
            print(f"\nEvaluating dataset: {dataset_name}")

        dataset_metrics = evaluate_single_dataset(model, dataloader, dataset_name, t_config, tokenizer)
        all_dataset_metrics.append(dataset_metrics)

    # Compute macro averages and print comprehensive metrics (rich table + panel or plain)
    macro_metrics = compute_macro_average(all_dataset_metrics)
    _print_eval_metrics(all_dataset_metrics, macro_metrics, step=step)

    return all_dataset_metrics, macro_metrics


def log_metrics_to_wandb(
    dataset_metrics: List[DatasetMetrics],
    macro_metrics: Dict[str, float],
    step: int,
    prefix: str = "val",
):
    """
    Log all metrics to Weights & Biases.

    Args:
        dataset_metrics: List of metrics for each dataset
        macro_metrics: Dictionary of macro-averaged metrics
        step: Current training step
        prefix: Prefix for metric names (default: "val")
    """
    import wandb

    log_dict = {}

    # Log per-dataset metrics
    for dm in dataset_metrics:
        ds_name = dm.dataset_name
        log_dict[f"{prefix}/{ds_name}_wer"] = dm.wer
        log_dict[f"{prefix}/{ds_name}_cer"] = dm.cer
        log_dict[f"{prefix}/{ds_name}_loss"] = dm.mean_token_nll
        log_dict[f"{prefix}/{ds_name}_mean_token_nll"] = dm.mean_token_nll
        log_dict[f"{prefix}/{ds_name}_avg_log_prob"] = dm.avg_log_prob
        log_dict[f"{prefix}/{ds_name}_mean_token_entropy"] = dm.mean_token_entropy
        log_dict[f"{prefix}/{ds_name}_ece"] = dm.ece
        log_dict[f"{prefix}/{ds_name}_num_samples"] = dm.num_samples
        if dm.inference_ms_per_sample is not None:
            log_dict[f"{prefix}/{ds_name}_ms_per_sample"] = dm.inference_ms_per_sample
            log_dict[f"{prefix}/{ds_name}_samples_per_sec"] = dm.inference_samples_per_sec

    # Log macro averages (skip None, e.g. macro_ms_per_sample when no dataset has latency)
    for metric_name, metric_value in macro_metrics.items():
        if metric_value is not None:
            log_dict[f"{prefix}/{metric_name}"] = metric_value

    # Use explicit step= parameter instead of putting step in log_dict
    # This avoids W&B step mismatch warnings
    wandb.log(log_dict, step=step)
