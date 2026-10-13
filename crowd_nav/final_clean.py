import re

with open('train_ultra_clean.py', 'r') as f:
    lines = f.readlines()

output = []
in_docstring = False
skip_next = 0

for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    s = line.strip()
    
    if s.startswith('"""') or s.startswith("'''"):
        if in_docstring:
            in_docstring = False
        else:
            in_docstring = True
        continue
    
    if in_docstring:
        continue
    
    if not s:
        continue
    
    if 'from tqdm' in line or 'import tqdm' in line:
        continue
    
    if re.match(r'\s*pbar\s*=', line):
        continue
    
    if 'pbar.set_postfix' in line or 'pbar.update' in line:
        continue
    
    output.append(line)

with open('train_final.py', 'w') as f:
    f.writelines(output)

print(f"Final: {len(output)} lines")
