from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "xray_error_analysis" / "advanced_diagnostic.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

bins = Counter()
class_bins = {}

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row["Diagnostic"] != "Unmatched prediction":
        continue

    cls = row["Predicted Class"]
    value = row["IoU"]

    if not value:
        continue

    overlap = float(value)

    if overlap < 0.10:
        category = "0.00 - 0.09"
    elif overlap < 0.25:
        category = "0.10 - 0.24"
    elif overlap < 0.50:
        category = "0.25 - 0.49"
    else:
        category = "0.50 - 1.00"

    bins[category] += 1
    class_bins.setdefault(cls, Counter())[category] += 1

print("\nLOCALIZATION ANALYSIS")
print("=" * 55)
print(f"{'Best IoU overlap':<22}{'Count':>10}")

for category in [
    "0.00 - 0.09",
    "0.10 - 0.24",
    "0.25 - 0.49",
    "0.50 - 1.00"
]:
    print(f"{category:<22}{bins[category]:>10}")

print("\nCLASS-WISE LOCALIZATION")
print("=" * 75)
print(f"{'Class':<12}{'0-.09':>10}{'.10-.24':>12}{'.25-.49':>12}{'.50-1':>10}")

for cls, values in class_bins.items():
    print(
        f"{cls:<12}"
        f"{values['0.00 - 0.09']:>10}"
        f"{values['0.10 - 0.24']:>12}"
        f"{values['0.25 - 0.49']:>12}"
        f"{values['0.50 - 1.00']:>10}"
    )

print("\nTotal unmatched predictions:", sum(bins.values()))
