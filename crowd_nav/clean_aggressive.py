import re

with open('train_optimized.py', 'r') as f:
    lines = f.readlines()

output = []
for line in lines:
    s = line.strip()
    if not s:
        continue
    if s.startswith('#'):
        continue
    if any(x in line for x in [
        'logging.info',
        'logging.debug', 
        'logging.warning',
        'print(',
        '"""',
        "'''",
    ]):
        if any(k in line for k in ['ERROR', 'CRITICAL', 'exception', 'Failed', 'error']):
            output.append(line)
        continue
    if re.match(r'\s*(try|except Exception|except\s+\w+):\s*$', line):
        continue
    if line.strip() in ['pass', 'continue']:
        continue
    output.append(line)

with open('train_ultra_clean.py', 'w') as f:
    f.writelines(output)

print(f"Reduced from {len(lines)} to {len(output)} lines")
