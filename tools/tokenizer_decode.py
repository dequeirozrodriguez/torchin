#!/usr/bin/env python3
"""
tokenizer_decode.py - Decode token IDs back to text using Qwen3 tokenizer

This script takes token IDs (from RefTorch inference output) and converts
them back to human-readable text.

Input formats:
  - CSV: One token ID per line
  - JSON: {"tokens": [...]} or just [...]
  - Binary: int32 count followed by int32 array
  - Stdin: Space or newline separated integers

Usage:
    # From file
    python tokenizer_decode.py --model Qwen/Qwen3-0.6B --input output_tokens.csv
    
    # From command line
    python tokenizer_decode.py --model Qwen/Qwen3-0.6B --tokens "151644 872 151645"
    
    # Streaming mode (for real-time output)
    python tokenizer_decode.py --model Qwen/Qwen3-0.6B --stream --input tokens.csv
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found")
    print("Install with: pip install transformers")
    sys.exit(1)


class Qwen3Decoder:
    """Wrapper for Qwen3 tokenizer decoding."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Loading tokenizer: {model_name}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model_name = model_name
        
        # Get special token IDs for filtering
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        
        # Tokens to potentially skip in output
        self.skip_tokens = set()
        if self.bos_token_id is not None:
            self.skip_tokens.add(self.bos_token_id)
    
    def decode(self, tokens: list, skip_special_tokens: bool = False) -> str:
        """Decode a list of token IDs to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID."""
        return self.tokenizer.decode([token_id])
    
    def is_eos(self, token_id: int) -> bool:
        """Check if token is end-of-sequence."""
        return token_id == self.eos_token_id
    
    def decode_streaming(self, tokens: list, skip_special_tokens: bool = True):
        """
        Generator that yields text incrementally as tokens are processed.
        Useful for displaying output as it's generated.
        """
        prev_text = ""
        for i in range(1, len(tokens) + 1):
            current_text = self.decode(tokens[:i], skip_special_tokens=skip_special_tokens)
            new_text = current_text[len(prev_text):]
            if new_text:
                yield new_text
            prev_text = current_text


def load_tokens_csv(path: str) -> list:
    tokens = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    tokens.append(int(line))
                except ValueError:
                    pass  # Skip non-integer lines
    return tokens


def load_tokens_json(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "tokens" in data:
        return data["tokens"]
    else:
        raise ValueError("JSON must be a list or have 'tokens' key")


def load_tokens_binary(path: str) -> list:
    tokens = []
    with open(path, "rb") as f:
        count_bytes = f.read(4)
        if len(count_bytes) < 4:
            raise ValueError("Invalid binary file")
        count = struct.unpack("<I", count_bytes)[0]
        
        for _ in range(count):
            token_bytes = f.read(4)
            if len(token_bytes) < 4:
                break
            tokens.append(struct.unpack("<i", token_bytes)[0])
    
    return tokens


def load_tokens_auto(path: str) -> list:
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        return load_tokens_json(path)
    elif suffix == ".bin":
        return load_tokens_binary(path)
    else:  # Default to CSV
        return load_tokens_csv(path)


def parse_token_string(token_str: str) -> list:
    tokens = []
    for part in token_str.replace(",", " ").split():
        try:
            tokens.append(int(part))
        except ValueError:
            pass
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Decode Qwen3 token IDs to text"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file with token IDs"
    )
    parser.add_argument(
        "--tokens", "-t",
        type=str,
        help="Token IDs as space-separated string"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token"
    )
    parser.add_argument(
        "--keep-special",
        action="store_true",
        help="Keep special tokens in output"
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show token ID to text mapping"
    )
    parser.add_argument(
        "--extract-response",
        action="store_true",
        help="Extract only the assistant's response (after last <|im_start|>assistant)"
    )
    
    args = parser.parse_args()
    
    decoder = Qwen3Decoder(args.model)
    
    if args.tokens:
        tokens = parse_token_string(args.tokens)
    elif args.input:
        tokens = load_tokens_auto(args.input)
    else:
        print("Reading tokens from stdin (space or newline separated)...", file=sys.stderr)
        stdin_content = sys.stdin.read()
        tokens = parse_token_string(stdin_content)
    
    if not tokens:
        print("Error: No tokens to decode", file=sys.stderr)
        sys.exit(1)
    
    print(f"Decoding {len(tokens)} tokens...", file=sys.stderr)
    
    if args.show_tokens:
        print("\nToken breakdown:", file=sys.stderr)
        for i, tok in enumerate(tokens):
            decoded = decoder.decode_token(tok)
            is_eos = " [EOS]" if decoder.is_eos(tok) else ""
            print(f"  {i:4d}: {tok:6d} -> {repr(decoded)}{is_eos}", file=sys.stderr)
        print(file=sys.stderr)
    
    skip_special = not args.keep_special
    
    if args.stream:
        print("\n--- Output ---", file=sys.stderr)
        for text_chunk in decoder.decode_streaming(tokens, skip_special):
            print(text_chunk, end="", flush=True)
        print()  # Final newline
    else:
        text = decoder.decode(tokens, skip_special_tokens=skip_special)
        
        if args.extract_response:
            marker = "<|im_start|>assistant\n"
            end_marker = "<|im_end|>"
            
            if marker in text:
                text = text.split(marker)[-1]
                if end_marker in text:
                    text = text.split(end_marker)[0]
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
            print(f"Saved to: {args.output}", file=sys.stderr)
        else:
            print("\n--- Decoded Text ---")
            print(text)
            print("--- End ---")


if __name__ == "__main__":
    main()

