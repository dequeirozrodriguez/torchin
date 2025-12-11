#!/usr/bin/env python3
"""
check_binary_format.py - Verify binary files match TLoadBinary format

Format expected by TLoadBinary:
  - int64_t ndim          (8 bytes, little-endian)
  - int64_t shape[ndim]   (8 bytes each)
  - float32 data[numel]   (4 bytes each)
"""

import struct
import numpy as np
import os

BASE = "/home/dieguez/qwen3_perl/torchin/torchin/qwen_data"

def check_binary(path, desc=""):
    if not os.path.exists(path):
        print(f"❌ {desc}: FILE NOT FOUND")
        return False
    
    try:
        with open(path, 'rb') as f:
            # Read ndim (int64_t)
            ndim_bytes = f.read(8)
            if len(ndim_bytes) != 8:
                print(f"❌ {desc}: Can't read ndim (got {len(ndim_bytes)} bytes)")
                return False
            ndim = struct.unpack('<q', ndim_bytes)[0]
            
            # Read shape
            shape = []
            for i in range(ndim):
                dim_bytes = f.read(8)
                if len(dim_bytes) != 8:
                    print(f"❌ {desc}: Can't read dim {i}")
                    return False
                shape.append(struct.unpack('<q', dim_bytes)[0])
            
            # Calculate numel
            numel = 1
            for s in shape:
                numel *= s
            
            # Read data
            data_bytes = f.read(numel * 4)
            if len(data_bytes) != numel * 4:
                print(f"❌ {desc}: Data size mismatch. Expected {numel*4}, got {len(data_bytes)}")
                return False
            
            data = np.frombuffer(data_bytes, dtype=np.float32)
            
            print(f"✓ {desc}: ndim={ndim}, shape={shape}, numel={numel}, "
                  f"mean={data.mean():.4f}, std={data.std():.4f}")
            return True
            
    except Exception as e:
        print(f"❌ {desc}: ERROR - {e}")
        return False

def main():
    print("="*60)
    print("Checking binary format for TLoadBinary compatibility")
    print("="*60)
    
    files = [
        ("model.embed_tokens.weight.bin", "embed_tokens"),
        ("model.norm.weight.bin", "final_norm"),
        ("layer_0_packed.bin", "layer_0"),
        ("layer_1_packed.bin", "layer_1"),
        ("prompt.bin", "prompt"),
    ]
    
    all_ok = True
    for filename, desc in files:
        path = os.path.join(BASE, filename)
        if not check_binary(path, desc):
            all_ok = False
    
    print()
    if all_ok:
        print("✓ All files are in correct binary format!")
    else:
        print("❌ Some files have issues. Re-run pack_qwen_weights.py")

if __name__ == "__main__":
    main()

