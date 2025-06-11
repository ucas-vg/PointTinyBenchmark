import os
c = 0
for root, dirs, files in os.walk('../mmdet'):
    for file in files:
        if not file.endswith('.py'):
            continue
        filepath = os.path.join(root, file)
        print(filepath)
        c += len([f for f in open(filepath).readlines() if len(f.strip()) > 0])

print(c)
