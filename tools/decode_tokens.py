#!/usr/bin/env python3
"""
decode_tokens.py - Decode Qwen3 token IDs to text

Usage:
    python decode_tokens.py 21 28 35 42 49 56 63 70 77 84
    
    # Or pipe from your Refal output:
    echo "21 28 35 42" | python decode_tokens.py
    
    # Or with a file:
    python decode_tokens.py < tokens.txt
"""

import sys

def main():
    # Try to import transformers
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: Please install transformers: pip install transformers")
        sys.exit(1)
    
    # Load Qwen3 tokenizer
    print("Loading Qwen3 tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Get token IDs from arguments or stdin
    if len(sys.argv) > 1:
        # From command line arguments
        tokens_str = ' '.join(sys.argv[1:])
    else:
        # From stdin
        tokens_str = sys.stdin.read()
    
    # Parse token IDs (handle various formats)
    tokens_str = tokens_str.replace(',', ' ').replace('\n', ' ')
    token_ids = []
    for part in tokens_str.split():
        try:
            token_ids.append(int(part))
        except ValueError:
            continue  # Skip non-numeric parts like "GENERATED:"
    
    if not token_ids:
        print("No token IDs found!")
        sys.exit(1)
    
    print(f"Token IDs ({len(token_ids)}): {token_ids}", file=sys.stderr)
    
    # Decode to text
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    print("\n" + "="*50)
    print("DECODED TEXT:")
    print("="*50)
    print(text)
    print("="*50)
    
    # Also show individual tokens for debugging
    print("\nToken breakdown:", file=sys.stderr)
    for tid in token_ids:
        token_text = tokenizer.decode([tid])
        print(f"  {tid:6d} -> {repr(token_text)}", file=sys.stderr)

if __name__ == "__main__":
    main()
