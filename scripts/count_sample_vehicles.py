import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

input_path = Path("data/interim/vehicle_count_pilot.csv")
# output_dir = Path("data/processed/yolo_sample")
output_dir = Path("data/processed/yolo_sample_s_conf010")
# output_dir = Path("data/processed/yolo_sample_s_conf015")
# output_dir = Path("data/processed/yolo_sample_s_conf015_img960")
output_dir.mkdir(parents=True, exist_ok=True)

with input_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

if len(rows) != 40:
    raise ValueError(f"Expected 40 sample rows, found {len(rows)}")

model = YOLO("yolo11s.pt")
vehicle_names = {"car", "motorcycle", "bus", "truck", "bicycle"}
class_ids = [
    class_id
    for class_id, name in model.names.items()
    if name in vehicle_names
]

output_path = output_dir / "counts.csv"
extra_fields = ["predicted_count", "status", "preview_path", "error"]

with output_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames + extra_fields)
    writer.writeheader()

    for index, row in enumerate(rows, start=1):
        row.update(predicted_count="", status="", preview_path="", error="")
        notes = (row.get("notes") or "").strip()

        # ใช้หมายเหตุที่นับมือระบุไว้เพื่อข้ามภาพสีขาวสองภาพ
        if notes == "ภาพสีขาว":
            row["status"] = "invalid_image"
            writer.writerow(row)
            print(f"[{index}/40] skipped white image")
            continue

        # เปิดผ่าน Pillow เพื่ออ่านพาธชื่อภาษาไทย
        try:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
        except OSError as error:
            row["status"] = "read_error"
            row["error"] = str(error)
            writer.writerow(row)
            print(f"[{index}/40] read error")
            continue

        result = model.predict(
            source=image,
            classes=class_ids,
            conf=0.10,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        preview_path = output_dir / f"{index:02d}_{row['camera_id']}.jpg"
        result.save(filename=str(preview_path))

        row["predicted_count"] = len(result.boxes)
        row["status"] = "ok"
        row["preview_path"] = preview_path.as_posix()
        writer.writerow(row)

        print(
            f"[{index}/40] camera={row['camera_id']} "
            f"manual={row['manual_count']} "
            f"YOLO={row['predicted_count']}"
        )

print(f"Saved results to {output_path}")