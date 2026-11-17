
from pathlib import Path
from PIL import Image

# Dataset location
DATASET_PATH = Path(
    r"C:\Users\hp\Downloads\archive (13)\sixray_v3"
)

# Dataset classes
CLASS_NAMES = [
    "Gun",
    "Knife",
    "Pliers",
    "Scissors",
    "Wrench"
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


def validate_split(split_name):

    print("\n" + "=" * 55)
    print(f"VALIDATING {split_name.upper()} DATASET")
    print("=" * 55)

    image_dir = DATASET_PATH / split_name / "images"
    label_dir = DATASET_PATH / split_name / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print("ERROR: Image or label folder not found!")
        return

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = {
        p.stem: p
        for p in label_dir.glob("*.txt")
    }

    image_names = {p.stem for p in images}

    missing_labels = []
    missing_images = []
    invalid_labels = []
    empty_labels = []
    corrupt_images = []

    total_boxes = 0

    # Check images and matching labels
    for image_path in images:

        if image_path.stem not in labels:
            missing_labels.append(image_path.name)

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupt_images.append(image_path.name)

    # Check labels
    for label_name, label_path in labels.items():

        if label_name not in image_names:
            missing_images.append(label_path.name)

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            if not lines:
                empty_labels.append(label_path.name)
                continue

            for line_number, line in enumerate(lines, start=1):

                parts = line.split()

                if len(parts) != 5:
                    invalid_labels.append(
                        f"{label_name}: line {line_number}, "
                        "expected 5 values"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    valid = (
                        0 <= class_id < NUM_CLASSES
                        and 0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 < w <= 1
                        and 0 < h <= 1
                        and x - w / 2 >= -1e-6
                        and y - h / 2 >= -1e-6
                        and x + w / 2 <= 1 + 1e-6
                        and y + h / 2 <= 1 + 1e-6
                    )

                    if not valid:
                        invalid_labels.append(
                            f"{label_name}: line {line_number}, "
                            "invalid class or coordinates"
                        )
                    else:
                        total_boxes += 1

                except ValueError:
                    invalid_labels.append(
                        f"{label_name}: line {line_number}, "
                        "non-numeric value"
                    )

        except Exception as error:
            invalid_labels.append(
                f"{label_name}: {error}"
            )

    # Display report
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Valid bounding boxes:", total_boxes)

    print("Missing labels:", len(missing_labels))
    print("Missing images:", len(missing_images))
    print("Empty labels:", len(empty_labels))
    print("Invalid labels:", len(invalid_labels))
    print("Corrupt images:", len(corrupt_images))

    if invalid_labels:
        print("\nFirst invalid labels:")
        for item in invalid_labels[:5]:
            print(item)

    if missing_labels:
        print("\nFirst missing labels:")
        for item in missing_labels[:5]:
            print(item)

    if corrupt_images:
        print("\nFirst corrupt images:")
        for item in corrupt_images[:5]:
            print(item)

    return {
        "split": split_name,
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_labels": len(empty_labels),
        "invalid_labels": len(invalid_labels),
        "corrupt_images": len(corrupt_images)
    }


if __name__ == "__main__":

    print("SIXRAY DATASET VALIDATION")
    print("Classes:", CLASS_NAMES)

    reports = []

    for split in ["train", "valid", "test"]:
        report = validate_split(split)

        if report:
            reports.append(report)

    print("\n" + "=" * 55)
    print("DATASET VALIDATION COMPLETED")
    print("=" * 55)

    for report in reports:
        print(report)
