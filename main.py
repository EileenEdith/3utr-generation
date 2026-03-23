import argparse
from pathlib import Path
from typing import Optional

from src.generation.generate_gemorna import generate_sequences, run_conditional_generation


def validate_file(path_str: Optional[str], name: str) -> None:
    if path_str is None:
        return

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main entry point for GEMORNA-based 3'UTR generation."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/pretrained/gemorna_3utr.pt",
        help="Path to pretrained or fine-tuned model checkpoint."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/generated_sequences/gemorna_3utr_generations.txt",
        help="Path to save generated sequences."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cpu, mps, or cuda."
    )

    # Original GEMORNA generation mode
    parser.add_argument(
        "--utr_length",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Use original GEMORNA generation path with length bucket."
    )
    parser.add_argument(
        "--shared_library",
        type=str,
        default="src/models/gemorna_shared/libg2m.so",
        help="Path to GEMORNA shared generation library for original generation mode."
    )

    # Conditional generation mode
    parser.add_argument(
        "--pgk",
        type=str,
        default=None,
        choices=["high", "low"],
        help="Conditional PGK tag for finetuned generation."
    )
    parser.add_argument(
        "--len",
        dest="length_tag",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="Conditional length tag for finetuned generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Explicit control prompt, e.g. "<pgk_high> <len_short>".'
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens for conditional decoding."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for conditional decoding."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling for conditional decoding."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_file(args.checkpoint, "Checkpoint file")

    use_conditional = any([args.pgk, args.length_tag, args.prompt])

    if use_conditional:
        result = run_conditional_generation(
            checkpoint_path=args.checkpoint,
            pgk=args.pgk,
            length=args.length_tag,
            prompt=args.prompt,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print("[INFO] Conditional prompt:", result['prompt'])
        print("[INFO] Generated token length:", result['generated_token_length'])
        print("[INFO] Decoded sequence length:", result['decoded_sequence_length'])
        print("[INFO] Generated sequence:")
        print(result['sequence'])
        return

    if args.utr_length is None:
        raise ValueError(
            "Provide either conditional arguments (--pgk/--len/--prompt) or the original --utr_length mode."
        )

    validate_file(args.shared_library, "Shared library")
    generate_sequences(
        checkpoint_path=args.checkpoint,
        utr_length=args.utr_length,
        output_path=args.output,
        shared_library_path=args.shared_library,
        device=args.device,
    )


if __name__ == "__main__":
    main()
