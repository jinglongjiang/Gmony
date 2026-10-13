#!/usr/bin/env python3
import re

with open('train.py', 'r') as f:
    content = f.read()

content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

lines = content.split('\n')
output = []

for line in lines:
    stripped = line.strip()
    
    if stripped.startswith('#'):
        if 'HACK' in stripped or 'BUG' in stripped or 'TODO' in stripped or 'FIXME' in stripped:
            output.append(line)
        continue
    
    if not stripped:
        continue
    
    if 'logging.info(' in line and not any(x in line for x in ['ERROR', 'WARNING', 'CRITICAL', 'Exception', 'Failed', 'Error', 'failed']):
        continue
    
    if 'logging.debug(' in line:
        continue
    
    output.append(line)

with open('train.py', 'w') as f:
    f.write('\n'.join(output))

print(f"Reduced to {len(output)} lines")
