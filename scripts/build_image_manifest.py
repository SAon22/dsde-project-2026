import csv
from datetime import datetime
from pathlib import Path

images_dir = Path("data/raw/cctv_images")

with open("data/raw/cctv_info.csv", encoding="utf-8-sig", newline="") as file:
    camera_ids = {row["camera_id"].strip() for row in csv.DictReader(file)}

rows = []
for camera_dir in sorted(images_dir.iterdir()):
    if not camera_dir.is_dir():
        continue

    camera_id = camera_dir.name
    if camera_id not in camera_ids:
        raise ValueError(f"Camera {camera_id} is missing from cctv_info.csv")

    for image in sorted(camera_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # ดึงวันที่และเวลาจากสองส่วนท้ายของชื่อรูป
        try:
            _, date, time = image.stem.rsplit("_", 2)
            timestamp = datetime.strptime(date + time, "%Y%m%d%H%M%S")
        except ValueError as error:
            raise ValueError(f"Invalid timestamp in {image}") from error

        rows.append((camera_id, timestamp.isoformat(sep=" "), image.as_posix()))

output_path = Path("data/processed/image_manifest.csv")
with output_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["camera_id", "timestamp", "image_path"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to image_manifest.csv")