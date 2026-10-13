with open('train.py', 'r') as f:
    lines = f.readlines()

output = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 6:
        i += 1
        continue
    
    if '#' in line and not line.strip().startswith('#'):
        comment_pos = line.find('#')
        if comment_pos > 0:
            code_part = line[:comment_pos].rstrip()
            if code_part:
                output.append(code_part + '\n')
                i += 1
                continue
    
    if i < len(lines) - 2:
        if 'try:' in stripped and i + 1 < len(lines):
            next_line = lines[i+1].strip()
            if next_line in ['pass', ''] or next_line.startswith('except'):
                i += 2
                while i < len(lines) and lines[i].strip() in ['pass', 'except Exception:', 'except:']:
                    i += 1
                continue
    
    output.append(line)
    i += 1

with open('train.py', 'w') as f:
    f.writelines(output)

print(f"Cleaned to {len(output)} lines")
