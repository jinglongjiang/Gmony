with open('train.py', 'r') as f:
    lines = f.readlines()

output = []
for line in lines:
    s = line.strip()
    
    if 'logging.warning(' in line and any(x in line for x in ['partial', 'Missing', 'skip', 'fallback', 'deprecated']):
        continue
    
    if 'print(' in line and 'debug' in line.lower():
        continue
    
    output.append(line)

with open('train.py', 'w') as f:
    f.writelines(output)

print(f"Trimmed to {len(output)} lines")
