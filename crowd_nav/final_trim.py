with open('train.py', 'r') as f:
    lines = f.readlines()

output = []
for i, line in enumerate(lines):
    s = line.strip()
    
    if s.startswith('if __name__') or s == 'else:' or s == 'elif:':
        output.append(line)
        continue
    
    if len(output) > 0 and output[-1].rstrip() == line.rstrip():
        continue
    
    if 'getattr(' in line and 'None' in line and i < len(lines) - 1:
        next_line = lines[i+1].strip()
        if 'if' in next_line and 'is None' in next_line:
            continue
    
    output.append(line)

with open('train.py', 'w') as f:
    f.writelines(output[: 1999])

print(f"Trimmed to {min(len(output), 1999)} lines")
