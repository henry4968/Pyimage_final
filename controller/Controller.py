import os
import cv2

from controller.YoloDetector import YoloDetector
from controller.OcrRecognizer import OcrRecognizer
from controller.VehicleClassifier import VehicleClassifier
from controller.DBService import DBService
from controller.Logger import Logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


class ALPRController:
    def __init__(self):
        self.logger = Logger()

        # DB 路徑
        self.db_path = os.path.join(PROJECT_DIR, "Dataset", "database.db")
        self.db = DBService(self.db_path, self.logger)

        # plate 模型路徑
        self.plate_model_path = os.path.join(
            PROJECT_DIR, "runs", "detect", "train4", "weights", "best.pt"
        )

        # input / output
        self.input_dir = os.path.join(PROJECT_DIR, "Dataset", "test_images")
        self.plate_output_dir = os.path.join(PROJECT_DIR, "output", "plates")
        self.final_output_dir = os.path.join(PROJECT_DIR, "output", "final")

        os.makedirs(self.plate_output_dir, exist_ok=True)
        os.makedirs(self.final_output_dir, exist_ok=True)

        # modules
        self.yolo = YoloDetector(self.plate_model_path, self.plate_output_dir)
        self.ocr = OcrRecognizer()
        self.vehicle_classifier = VehicleClassifier()  # 你若要帶 model_path 再改這裡

        self.logger.info("ALPRController initialized")

    def run(self):
        if not os.path.isdir(self.input_dir):
            self.logger.error(f"input_dir not found: {self.input_dir}")
            return

        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.logger.info(f"Found {len(files)} images")

        for filename in files:
            img_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                self.logger.error(f"Read failed: {filename}")
                continue

            self.logger.info(f"Processing {filename}")

            # 1) YOLO 偵測 + 存 plate crop（由 YoloDetector 負責存檔）
            plates = self.yolo.detect(img, filename)  # list of dicts: seq,bbox,conf,plate_img,plate_file

            plate_texts_for_final_name = []
            annotated = img.copy()

            for p in plates:
                seq = p["seq"]
                x1, y1, x2, y2 = p["bbox"]
                yolo_conf = p["conf"]
                plate_img = p["plate_img"]

                # 2) OCR（防呆在 OcrRecognizer 內）
                plate_text, ocr_conf = self.ocr.recognize(plate_img)
                plate_texts_for_final_name.append(plate_text)

                # 3) DB 寫入：車牌資訊
                self.db.insert_alpr_info(
                    filename=filename,
                    seq=seq,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    yolo_conf=yolo_conf,
                    plate_text=plate_text,
                    ocr_conf=ocr_conf,
                )

                # 4) 車種辨識（只針對「有 plate」的才做）
                # 你目前車種辨識器怎麼實作我不知道，所以這裡先用 try 包住
                vehicle_type, vehicle_conf, vehicle_box = None, None, None
                try:
                    vehicle_type, vehicle_conf, vehicle_box = self.vehicle_classifier.classify(img, (x1, y1, x2, y2))
                    # 寫回 DB
                    self.db.update_vehicle_info(filename, seq, vehicle_type, vehicle_conf)
                except Exception as e:
                    self.logger.error(f"[Vehicle] classify failed: {e}")

                # 5) 畫框：綠色車牌 + 藍色車（若有）
                # 綠色車牌框
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{plate_text} ({ocr_conf:.2f})",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # 藍色車框（如果 vehicle_box 有回傳）
                if vehicle_box:
                    vx1, vy1, vx2, vy2 = vehicle_box
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    if vehicle_type:
                        cv2.putText(
                            annotated,
                            f"{vehicle_type} ({vehicle_conf:.2f})",
                            (vx1, max(30, vy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            2
                        )

                elif vehicle_box:
                    vx1, vy1, vx2, vy2 = vehicle_box
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{vehicle_type} ({vehicle_conf:.2f})",
                        (vx1, max(30, vy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )
            # 6) final 檔名：用車牌組合用 _ 串接（照你規則）
            if not plate_texts_for_final_name:
                final_name = "UNKNOWN.jpg"
            else:
                final_name = "_".join(plate_texts_for_final_name) + ".jpg"

            final_path = os.path.join(self.final_output_dir, final_name)
            cv2.imwrite(final_path, annotated)
            self.logger.info(f"[FINAL] saved: {final_name}")
