"""
调试 captions.txt 格式
"""

import os
from pathlib import Path

captions_file = Path("data/captions.txt")

print("=" * 60)
print("Checking captions.txt file...")
print("=" * 60)

if not captions_file.exists():
    print("❌ captions.txt not found!")
else:
    print(f"✓ captions.txt found at: {captions_file}")
    print(f"  File size: {captions_file.stat().st_size} bytes")

    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nTotal lines: {len(lines)}")
    print("\nFirst 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"  Line {i}: {repr(line)}")

    print("\nChecking line format...")
    for i, line in enumerate(lines[:10]):
        parts = line.strip().split('\t')
        print(f"  Line {i}: {len(parts)} parts -> {parts}")

print("\n" + "=" * 60)
print("Checking Images directory...")
print("=" * 60)

images_dir = Path("data/Images")
if images_dir.exists():
    image_files = list(images_dir.glob("*.jpg"))
    print(f"✓ Found {len(image_files)} JPG files")
    if image_files:
        print(f"  Example filenames: {[f.name for f in image_files[:5]]}")
else:
    print("❌ Images directory not found!")
