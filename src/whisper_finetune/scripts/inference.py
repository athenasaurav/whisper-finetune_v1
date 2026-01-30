"""
Run inference with a fine-tuned Whisper checkpoint (e.g. best_model.pt from training).

Example:
    python -m whisper_finetune.scripts.inference \\
        --checkpoint output/20260130_090017/best_model.pt \\
        --audio path/to/audio.mp3 \\
        --language ar

    # Multiple files
    python -m whisper_finetune.scripts.inference \\
        --checkpoint output/.../best_model.pt \\
        --audio dir/with/wavs/ \\
        --language ar \\
        --output-dir ./transcripts
"""

import argparse
from pathlib import Path

import torch
import whisper

from whisper_finetune.model.model_utils import load_model_and_set_heads


def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Whisper checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. output/.../best_model.pt)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file or directory of audio files",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ar",
        help="Language code for transcription (default: ar)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="large-v3-turbo",
        help="Base model name used for training (must match checkpoint; default: large-v3-turbo)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, write one .txt transcript per audio file into this directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use fp16 for inference (default: True)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir():
        audio_files = sorted(
            f for f in audio_path.rglob("*")
            if f.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
        )
        if not audio_files:
            raise FileNotFoundError(f"No audio files found under {audio_path}")
    else:
        raise FileNotFoundError(f"Audio path not found: {audio_path}")

    print(f"Loading base model: {args.base_model}")
    model = whisper.load_model(args.base_model, device="cpu")
    print(f"Loading fine-tuned weights: {checkpoint_path}")
    load_model_and_set_heads(model, str(checkpoint_path), device=args.device)
    model.eval()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for f in audio_files:
        print(f"\nTranscribing: {f}")
        result = model.transcribe(
            str(f),
            language=args.language,
            fp16=args.fp16,
        )
        text = result["text"].strip()
        print(text)

        if args.output_dir:
            out_txt = Path(args.output_dir) / (f.stem + ".txt")
            out_txt.write_text(text, encoding="utf-8")
            print(f"  -> {out_txt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
