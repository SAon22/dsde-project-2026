import pandas as pd
from PIL import Image
from ultralytics import YOLO

pilot = pd.read_csv("data/interim/vehicle_count_pilot.csv")
image_path = pilot.iloc[0]["image_path"]

model = YOLO("yolo11s.pt")

with Image.open(image_path) as source:
    image = source.convert("RGB")

result = model.predict(
    source=image,
    classes=None,
    conf=0.05,
    imgsz=640,
    device="cpu",
    verbose=False,
)[0]

# display(Image.fromarray(result.plot()[..., ::-1]))
Image.fromarray(result.plot()[..., ::-1]).show()

for box in result.boxes:
    name = result.names[int(box.cls.item())]
    print(f"{name}: {box.conf.item():.3f}")