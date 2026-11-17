from pathlib import Path
from collections import Counter

folder = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3\valid\labels")
names = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]
stats = {name: Counter() for name in names}

for file in folder.glob("*.txt"):
    for line in file.read_text().splitlines():
        v = line.split()
        if len(v) != 5:
            continue

        cls = int(float(v[0]))
        if not 0 <= cls < len(names):
            continue

        area = float(v[3]) * float(v[4])
        size = "Small" if area < 0.01 else "Medium" if area < 0.05 else "Large"
        stats[names[cls]][size] += 1

print("VALIDATION OBJECT SIZE DISTRIBUTION")
print(f"{'Class':<12}{'Small':>10}{'Medium':>10}{'Large':>10}")

for name, values in stats.items():
    print(f"{name:<12}{values['Small']:>10}{values['Medium']:>10}{values['Large']:>10}")
