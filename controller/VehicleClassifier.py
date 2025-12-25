from ultralytics import YOLO
import math


class VehicleClassifier:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        # YOLO COCO 類別
        self.VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle",
            7: "truck",
        }

    # =========================
    # 基本工具
    # =========================
    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _overlap_ratio(self, plate_box, vehicle_box):
        ax1, ay1, ax2, ay2 = plate_box
        bx1, by1, bx2, by2 = vehicle_box

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        plate_area = (ax2 - ax1) * (ay2 - ay1)

        return inter_area / plate_area

    # =========================
    # 主分類流程
    # =========================
    def classify(self, image, plate_bbox):
        px1, py1, px2, py2 = plate_bbox
        plate_center = self._center(plate_bbox)

        results = self.model(image, conf=0.3, verbose=False)[0]
        best = None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.VEHICLE_CLASSES:
                continue

            vehicle_type = self.VEHICLE_CLASSES[cls_id]
            conf = float(box.conf[0])
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
            vehicle_bbox = (vx1, vy1, vx2, vy2)
            vehicle_center = self._center(vehicle_bbox)

            if vehicle_type in ("car", "truck"):
                overlap = self._overlap_ratio((px1, py1, px2, py2), vehicle_bbox)
                if overlap < 0.3:
                    continue

                dist = self._distance(plate_center, vehicle_center)
                if dist > max(vx2 - vx1, vy2 - vy1) * 1.8:
                    continue

            elif vehicle_type == "motorcycle":
                # 一定要有 YOLO motorcycle box
                if px2 < vx1 or px1 > vx2:
                    continue

                dist = self._distance(plate_center, vehicle_center)
                if dist > max(vx2 - vx1, vy2 - vy1) * 2.5:
                    continue

            if best is None or conf > best["conf"]:
                best = {
                    "type": vehicle_type,
                    "conf": conf,
                    "bbox": vehicle_bbox,
                }

        if best:
            return best["type"], best["conf"], best["bbox"]

        return None, None, None

