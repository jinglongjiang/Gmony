with open('train_compact.py', 'r') as f:
    lines = f.readlines()

seen_imports = set()
output = []

for line in lines:
    s = line.strip()
    
    if s.startswith('import ') or s.startswith('from '):
        if s in seen_imports:
            continue
        seen_imports.add(s)
    
    if 'from typing import' in line and 'Optional' not in line and 'Dict' not in line:
        continue
    
    if s == 'pass':
        continue
    
    if s.startswith('assert ') and 'shape' not in s:
        continue
    
    output.append(line)

with open('train.py', 'w') as f:
    f.writelines(output)

print(f"Ultra compact: {len(output)} lines")
