import json
import os
from glob import glob

IMG_DIR = r"D:\PyImage\final_project\dataset\plate_train\images\val"
OUT_DIR = r"D:\PyImage\final_project\dataset\plate_train\labels\val"

CLASS_MAP = {
    "plate": 0
}

os.makedirs(OUT_DIR, exist_ok=True)

for json_file in glob(os.path.join(IMG_DIR, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue

        (x1, y1), (x2, y2) = shape["points"]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

        yolo_lines.append(
            f"{CLASS_MAP[label]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )

    out_name = os.path.basename(json_file).replace(".json", ".txt")
    out_path = os.path.join(OUT_DIR, out_name)

    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))
