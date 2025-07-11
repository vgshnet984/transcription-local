#!/usr/bin/env python3
"""
Download or convert faster-whisper models with quantization (int8/int4/float16/float32).

Usage:
    python scripts/download_quantized_model.py --model large-v3 --quantization int8 --output ./models

Supported quantization: float16, float32, int8, int4
"""
import argparse
import os
from pathlib import Path

try:
    from faster_whisper import download_model
except ImportError:
    download_model = None
    print("[WARN] faster-whisper not installed. Only CTranslate2 conversion will be available.")


def main():
    parser = argparse.ArgumentParser(description="Download or convert faster-whisper models with quantization.")
    parser.add_argument('--model', required=True, help='Model name (e.g., large-v3, base, medium)')
    parser.add_argument('--quantization', choices=['float16', 'float32', 'int8', 'int4'], default='float16', help='Quantization type')
    parser.add_argument('--output', default='./models', help='Output directory')
    parser.add_argument('--hf-token', default=None, help='HuggingFace token (if required)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading/converting model: {args.model} with quantization: {args.quantization}")
    print(f"[INFO] Output directory: {output_dir}")

    if args.quantization in ('float16', 'float32', 'int8'):
        if download_model is None:
            print("[ERROR] faster-whisper not installed. Please install it for direct download.")
            return 1
        download_model(
            args.model,
            output_dir=str(output_dir),
            quantization=args.quantization,
            token=args.hf_token
        )
        print(f"[SUCCESS] Model downloaded to {output_dir}")
        return 0
    elif args.quantization == 'int4':
        print("[INFO] INT4 quantization requires CTranslate2 converter.")
        print("[INFO] Run the following command (requires ctranslate2):")
        print(f"ct2-transformers-converter --model openai/whisper-{args.model} --output_dir {output_dir}/ct2-{args.model}-int4 --quantization int4")
        print("[INFO] See: https://opennmt.net/CTranslate2/quantization.html#int4-quantization")
        return 0
    else:
        print(f"[ERROR] Unsupported quantization: {args.quantization}")
        return 1

if __name__ == "__main__":
    exit(main()) 