from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent.parent
image_dir = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3\test\images")
csv_path = root / "results" / "xray_error_analysis" / "detection_errors.csv"
output_path = root / "results" / "xray_error_analysis" / "top_error_images.jpg"

with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

counts = {}
for row in rows:
    name = row["Image"]
    counts[name] = counts.get(name, 0) + 1

top_images = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]

cell_w, cell_h = 320, 270
sheet = Image.new("RGB", (cell_w * 3, cell_h * 5), "white")
draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()

for index, (filename, error_count) in enumerate(top_images):
    path = image_dir / filename
    if not path.exists():
        print("Image not found:", filename)
        continue

    img = Image.open(path).convert("RGB")
    img.thumbnail((cell_w - 16, cell_h - 55))

    x = (index % 3) * cell_w
    y = (index // 3) * cell_h

    sheet.paste(img, (x + (cell_w - img.width) // 2, y + 5))
    draw.text((x + 8, y + cell_h - 45), filename[:39], fill="black", font=font)
    draw.text((x + 8, y + cell_h - 25), f"Total errors: {error_count}", fill="red", font=font)

sheet.save(output_path, quality=95)
print("Contact sheet saved:")
print(output_path)
