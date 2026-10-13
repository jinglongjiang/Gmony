#!/usr/bin/env python3
import re
import sys

def optimize_code(lines):
    output = []
    skip_until = None
    in_skip_bc_block = False
    block_depth = 0

    for i, line in enumerate(lines):
        if skip_until and i < skip_until:
            continue
        skip_until = None

        stripped = line.strip()

        if 'if train_cfg.skip_bc_training:' in line or 'if self.skip_bc_training' in line:
            in_skip_bc_block = True
            block_depth = len(line) - len(line.lstrip())
            continue

        if in_skip_bc_block:
            curr_depth = len(line) - len(line.lstrip())
            if curr_depth <= block_depth and stripped and not stripped.startswith('#'):
                in_skip_bc_block = False
            else:
                continue

        if re.match(r'\s*logging\.(info|debug)\(f?\[', line):
            continue
        if 'logging.info("="' in line:
            continue
        if re.search(r'logging\.info.*✓|✗|⚠', line):
            continue

        if any(x in line for x in [
            'Skip reading',
            'detected old format',
            'Detected format',
            'Missing key',
            'Ignored',
            'partial load',
            'Loading checkpoint',
            'Found existing',
            'Will train from scratch',
            'Starting.*epochs',
            'Training completed',
            'Checkpoint saved',
            'Saving checkpoint',
            'Final',
        ]):
            continue

        if re.match(r'\s*(try|except\s+Exception|pass)\s*:', line):
            next_i = i + 1
            while next_i < len(lines) and (lines[next_i].strip() == '' or lines[next_i].strip().startswith('#')):
                next_i += 1
            if next_i < len(lines) and lines[next_i].strip() in ['pass', 'continue']:
                skip_until = next_i + 1
                continue

        if not stripped or stripped == 'pass':
            continue

        output.append(line)

    return output

if __name__ == '__main__':
    with open('train_clean.py', 'r') as f:
        lines = f.readlines()

    optimized = optimize_code(lines)

    with open('train_optimized.py', 'w') as f:
        f.writelines(optimized)

    print(f"Original: {len(lines)} lines")
    print(f"Optimized: {len(optimized)} lines")
    print(f"Removed: {len(lines) - len(optimized)} lines")
