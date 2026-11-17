from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(r"C:\Users\hp\Downloads\archive (13)\sixray_v3")
MODEL = ROOT / "results" / "xray_detector_exp_recall" / "weights" / "best.pt"

NAMES = ["Gun", "Knife", "Pliers", "Scissors", "Wrench"]

FILES = [
    "P07722_jpg.rf.07731b99691d2ad100b5f3d3249db3d4.jpg",
    "P06287_jpg.rf.db9a29c1d09d056998a3c185ef28a137.jpg",
    "P00779_jpg.rf.ddf99f90c728937a4b689a4ac34e8535.jpg"
]

OUT = ROOT / "results" / "xray_error_analysis" / "comparisons"
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL))

def draw_box(draw, box, label, color):
    draw.rectangle(box, outline=color, width=3)
    x1, y1, _, _ = box
    draw.text((x1, max(0, y1 - 16)), label, fill=color)

def main():
    for filename in FILES:
        image_path = DATA / "test" / "images" / filename
        label_path = DATA / "test" / "labels" / (Path(filename).stem + ".txt")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        gt_image = image.copy()
        pred_image = image.copy()

        gt_draw = ImageDraw.Draw(gt_image)

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    continue

                cls = int(float(values[0]))
                x, y, bw, bh = map(float, values[1:])

                if not 0 <= cls < len(NAMES):
                    continue

                box = (
                    int((x - bw / 2) * w),
                    int((y - bh / 2) * h),
                    int((x + bw / 2) * w),
                    int((y + bh / 2) * h)
                )
                draw_box(gt_draw, box, NAMES[cls], "lime")

        result = model.predict(
            source=str(image_path),
            imgsz=416,
            conf=0.10,
            iou=0.45,
            device="cpu",
            max_det=100,
            verbose=False
        )[0]

        pred_draw = ImageDraw.Draw(pred_image)

        if result.boxes is not None:
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
                result.boxes.conf.cpu().numpy()
            ):
                if not 0 <= cls < len(NAMES):
                    continue

                coords = tuple(int(v) for v in box)
                draw_box(
                    pred_draw,
                    coords,
                    f"{NAMES[cls]} {conf:.2f}",
                    "red"
                )

        panel = Image.new("RGB", (w * 2, h + 45), "white")
        panel.paste(gt_image, (0, 45))
        panel.paste(pred_image, (w, 45))

        draw = ImageDraw.Draw(panel)
        draw.text((10, 12), "GROUND TRUTH - GREEN", fill="black")
        draw.text((w + 10, 12), "MODEL PREDICTIONS - RED", fill="black")

        output = OUT / f"comparison_{Path(filename).stem}.jpg"
        panel.save(output, quality=95)
        print("Saved:", output)

if __name__ == "__main__":
    main()
