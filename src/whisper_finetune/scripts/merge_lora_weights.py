"""
Merge LoRA weights into a Whisper model and save a plain .pt checkpoint.
Use --config with the same YAML used for training so init_name and lora_config match.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import whisper

from whisper_finetune.model.lora import apply_lora, is_lora_enabled, merge_lora
from whisper_finetune.model.model_utils import save_model
from whisper_finetune.utils import read_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into the base model and save a standard Whisper .pt."
    )
    parser.add_argument("--input", required=True, help="Path to input .pt checkpoint")
    parser.add_argument("--output", help="Path to output merged .pt checkpoint")
    parser.add_argument(
        "--config",
        help="Path to training config YAML (uses model.init_name and model.lora_config)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Model name if --config not given (e.g. large-v3-turbo, large-v3)",
    )
    parser.add_argument("--test_merge", action="store_true", help="Test merging before saving")
    return parser.parse_args()


def _load_lora_config(args) -> tuple[str, dict]:
    init_name = "large-v3"
    lora_config = {"rank": 16, "lora_alpha": 32, "lora_dropout": 0.0}

    if args.config:
        cfg = read_config(args.config)
        init_name = cfg["model"]["init_name"]
        lora_config = cfg["model"].get("lora_config") or lora_config

    if not args.config:
        init_name = args.model

    # If lora_config still default/empty, try save_dir/lora_config.json (written by finetune.py)
    if lora_config == {"rank": 16, "lora_alpha": 32, "lora_dropout": 0.0}:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        json_path = os.path.join(input_dir, "lora_config.json")
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                lora_config = json.load(f)
            print(f"Using lora_config from {json_path}")

    return init_name, lora_config


def main() -> None:
    args = parse_args()

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_merged{ext if ext else '.pt'}"

    init_name, lora_config = _load_lora_config(args)
    print(f"Loading base model: {init_name}, lora_config: {lora_config}")

    model = whisper.load_model(init_name, device="cpu")
    apply_lora(model, lora_config=lora_config)

    ckpt = torch.load(args.input, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("missing:", missing[:10], "..." if len(missing) > 10 else "")
        print("unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
        raise ValueError(
            f"State dict keys do not match model keys."
        )

    if is_lora_enabled(model):
        merge_lora(model)
        print("Merged LoRA weights into base model")
        
        if is_lora_enabled(model):
            print("Warning: LoRA parameters still detected after merge.")
    else:
        print("No LoRA parameters found; saving as-is")

    if args.test_merge:
        weights_changed = False
        for name, _ in model.named_parameters():
            if "lora" in name:
                raise ValueError(f"LoRA parameter {name} still present after merge.")
        original_model = whisper.load_model(init_name, device="cpu")
        for (name1, param1), (_, param2) in zip(
            model.named_parameters(), original_model.named_parameters()
        ):
            if not torch.allclose(param1, param2, atol=1e-5):
                print("Parameter", name1, "differs from original model after merge; LORA did change weights.")
                weights_changed = True
                break
        if not weights_changed:
            raise ValueError("No weights changed after merge; something went wrong.")
        print("Merge test passed: weights successfully merged.")

    save_model(model, output_path)
    print(f"Saved merged checkpoint to {output_path}")


if __name__ == "__main__":
    main()
