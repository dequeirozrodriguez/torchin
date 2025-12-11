#!/usr/bin/env python3
import argparse
import sys
import re
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--stdin", action="store_true", help="Read from pipe")
    parser.add_argument("--tokens", type=str, help="Space separated tokens")
    args = parser.parse_args()

    # 1. Get Token List
    token_str = ""
    
    if args.stdin:
        # Read everything from pipe
        full_output = sys.stdin.read()
        # Look for the marker we printed in Run.ref
        if "FINAL_SEQUENCE:" in full_output:
            token_str = full_output.split("FINAL_SEQUENCE:")[1]
        else:
            # Fallback: try to find any numbers in the last line
            token_str = full_output
    elif args.tokens:
        token_str = args.tokens
    else:
        print("Usage: ./Run | python tokenizer_decode.py --stdin")
        return

    # 2. Parse Integers (Robustly)
    # This finds all sequences of digits, ignoring text/logs
    token_ids = [int(x) for x in re.findall(r'\d+', token_str)]

    if not token_ids:
        print("No tokens found to decode.")
        return

    # 3. Decode
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print("\n=== GENERATED TEXT ===")
        print(text)
        print("======================")
    except Exception as e:
        print(f"Decoding Error: {e}")

if __name__ == "__main__":
    main()
