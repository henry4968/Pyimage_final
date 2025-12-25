import os
import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, output_dir):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect(self, image, filename):
        """
        return:
        [
            {
                'seq': 1,
                'bbox': (x1, y1, x2, y2),
                'conf': 0.92,
                'plate_img': ndarray,
                'plate_file': 'xxx_1.jpg'
            },
            ...
        ]
        """
        results = self.model(image)[0]
        detections = []

        seq = 1
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            plate_filename = f"{os.path.splitext(filename)[0]}_{seq}.jpg"
            plate_path = os.path.join(self.output_dir, plate_filename)

            # 存車牌截圖
            cv2.imwrite(plate_path, plate_img)

            detections.append({
                "seq": seq,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "plate_img": plate_img,
                "plate_file": plate_filename
            })

            seq += 1

        return detections
