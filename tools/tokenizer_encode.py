#!/usr/bin/env python3
"""
tokenizer_encode.py - Encode text to token IDs using Qwen3 tokenizer

This script takes text input and outputs token IDs that can be fed to
the RefTorch inference engine.

Output formats:
  - CSV: One token ID per line (default)
  - JSON: List of token IDs
  - Binary: Raw int32 array

Usage:
    # From command line argument
    python tokenizer_encode.py --model Qwen/Qwen3-0.6B --text "Hello, world!"
    
    # From file
    python tokenizer_encode.py --model Qwen/Qwen3-0.6B --input prompt.txt --output tokens.csv
    
    # Interactive mode
    python tokenizer_encode.py --model Qwen/Qwen3-0.6B --interactive

Special tokens for Qwen3:
    <|im_start|>  : Start of message
    <|im_end|>    : End of message
    <|endoftext|> : End of text (EOS)
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


class Qwen3Tokenizer:
    """Wrapper for Qwen3 tokenizer with chat template support."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model_name = model_name
        
        # Get special token IDs
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"  Vocab size: {self.tokenizer.vocab_size}")
        print(f"  BOS token ID: {self.bos_token_id}")
        print(f"  EOS token ID: {self.eos_token_id}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def encode_chat(self, messages: list, add_generation_prompt: bool = True) -> list:
        """
        Encode a chat conversation using the model's chat template.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            add_generation_prompt: Add tokens to prompt assistant response
        
        Returns:
            List of token IDs
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def encode_simple_prompt(self, user_message: str, system_message: str = None) -> list:
        """
        Encode a simple user prompt with optional system message.
        
        This creates the proper chat format for Qwen3.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        return self.encode_chat(messages, add_generation_prompt=True)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> dict:
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }


def save_tokens_csv(tokens: list, output_path: str):
    with open(output_path, "w") as f:
        for token in tokens:
            f.write(f"{token}\n")
    print(f"Saved {len(tokens)} tokens to: {output_path}")


def save_tokens_json(tokens: list, output_path: str):
    with open(output_path, "w") as f:
        json.dump({"tokens": tokens, "length": len(tokens)}, f, indent=2)
    print(f"Saved {len(tokens)} tokens to: {output_path}")


def save_tokens_binary(tokens: list, output_path: str):
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(tokens)))
        for token in tokens:
            f.write(struct.pack("<i", token))
    print(f"Saved {len(tokens)} tokens to: {output_path}")


def interactive_mode(tokenizer: Qwen3Tokenizer):
    """Interactive encoding mode."""
    print("\n" + "="*50)
    print("Interactive Tokenizer (Ctrl+C to exit)")
    print("="*50)
    print("\nCommands:")
    print("  /raw <text>  - Encode raw text without chat template")
    print("  /system <text> - Set system prompt")
    print("  /clear       - Clear system prompt")
    print("  /quit        - Exit")
    print()
    
    system_prompt = None
    
    while True:
        try:
            text = input("Enter text> ").strip()
            
            if not text:
                continue
            
            if text.startswith("/"):
                cmd = text.split(maxsplit=1)
                command = cmd[0].lower()
                
                if command == "/quit":
                    break
                elif command == "/clear":
                    system_prompt = None
                    print("System prompt cleared.")
                    continue
                elif command == "/system" and len(cmd) > 1:
                    system_prompt = cmd[1]
                    print(f"System prompt set: {system_prompt[:50]}...")
                    continue
                elif command == "/raw" and len(cmd) > 1:
                    tokens = tokenizer.encode(cmd[1], add_special_tokens=False)
                    print(f"Tokens ({len(tokens)}): {tokens}")
                    continue
                else:
                    print(f"Unknown command: {command}")
                    continue
            
            tokens = tokenizer.encode_simple_prompt(text, system_prompt)
            
            print(f"\nTokens ({len(tokens)}):")
            print(tokens)
            
            if len(tokens) <= 50:
                print("\nToken breakdown:")
                for i, tok in enumerate(tokens):
                    decoded = tokenizer.tokenizer.decode([tok])
                    print(f"  {i:3d}: {tok:6d} -> {repr(decoded)}")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Encode text to Qwen3 token IDs"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to encode"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file containing text"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tokens.csv",
        help="Output file (default: tokens.csv)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["csv", "json", "binary"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        help="System prompt to prepend"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Encode raw text without chat template"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print tokens to stdout"
    )
    
    args = parser.parse_args()
    
    tokenizer = Qwen3Tokenizer(args.model)
    
    if args.interactive:
        interactive_mode(tokenizer)
        return
    
    if args.text:
        text = args.text
    elif args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        print("Error: Provide --text, --input, or use --interactive")
        sys.exit(1)
    
    print(f"\nInput text ({len(text)} chars):")
    print(f"  {text[:100]}{'...' if len(text) > 100 else ''}")
    
    if args.raw:
        tokens = tokenizer.encode(text, add_special_tokens=True)
    else:
        tokens = tokenizer.encode_simple_prompt(text, args.system)
    
    print(f"\nEncoded to {len(tokens)} tokens")
    
    if args.show_tokens:
        print(f"Tokens: {tokens}")
    
    if args.format == "csv":
        save_tokens_csv(tokens, args.output)
    elif args.format == "json":
        save_tokens_json(tokens, args.output)
    elif args.format == "binary":
        save_tokens_binary(tokens, args.output)
    
    special_path = Path(args.output).parent / "special_tokens.json"
    with open(special_path, "w") as f:
        json.dump(tokenizer.get_special_tokens(), f, indent=2)
    print(f"Saved special tokens to: {special_path}")


if __name__ == "__main__":
    main()

