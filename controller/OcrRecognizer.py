from paddleocr import PaddleOCR
import re
from controller.Logger import Logger


class OcrRecognizer:
    def __init__(self):
        self.logger = Logger()
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False
        )

        # 7 碼新式車牌（去掉 - 後的結構）
        self.pattern_7 = re.compile(
            r"^[ABCEKLMNPRTV][A-Z]{2}[0-9]{4}$"
        )

        # 6 碼（寬鬆）
        self.pattern_6 = re.compile(
            r"^[A-Z0-9]{6}$"
        )

        # OCR 常見混淆
        self.confusion_map = {
            "0": "O",
            "O": "0",
            "1": "I",
            "I": "1",
            "2": "Z",
            "Z": "2",
            "5": "S",
            "S": "5",
            "8": "B",
            "B": "8",
        }

    # ==================================================
    # 工具
    # ==================================================

    def normalize(self, text):
        """
        保留 OCR 原本的 -，其餘只留英數
        """
        return re.sub(r"[^A-Z0-9-]", "", text.upper())

    def strip_dash(self, text):
        """
        移除 - 只用來驗證
        """
        return text.replace("-", "")

    def is_valid_plate_core(self, core):
        """
        core = 去掉 - 的車牌
        """
        if len(core) == 7:
            return bool(self.pattern_7.match(core))
        if len(core) == 6:
            return bool(self.pattern_6.match(core))
        return False

    def try_confusion_fix(self, core):
        """
        單字元 OCR 混淆修正（不處理 -）
        """
        for i, ch in enumerate(core):
            if ch in self.confusion_map:
                fixed = core[:i] + self.confusion_map[ch] + core[i + 1:]
                if self.is_valid_plate_core(fixed):
                    self.logger.info(
                        f"[OCR] confusion fix {core} -> {fixed}"
                    )
                    return fixed
        return None

    # ==================================================
    # 主流程
    # ==================================================

    def recognize(self, img):
        if img is None:
            return "UNKNOWN", 0.0

        result = self.ocr.ocr(img, cls=False)

        best_plate = "UNKNOWN"
        best_conf = 0.0

        if not result or result[0] is None:
            self.logger.info("[OCR] no text detected")
            return "UNKNOWN", 0.0

        for line in result:
            if line is None:
                continue

            for word in line:
                raw_txt, conf = word[1]

                txt = self.normalize(raw_txt)
                if not txt:
                    continue

                core = self.strip_dash(txt)

                # 原樣 OCR 就符合結構
                if self.is_valid_plate_core(core):
                    if conf > best_conf:
                        best_plate = txt        # ← 不改格式
                        best_conf = conf
                    continue

                # 嘗試混淆修正（只修英數，不動 -）
                fixed_core = self.try_confusion_fix(core)
                if fixed_core:
                    # 將修正後的 core 套回原本有沒有 -
                    if "-" in txt:
                        # 保留 OCR 的 -
                        idx = txt.index("-")
                        fixed_txt = fixed_core[:idx] + "-" + fixed_core[idx:]
                    else:
                        fixed_txt = fixed_core

                    if conf > best_conf:
                        best_plate = fixed_txt
                        best_conf = conf * 0.95

        if best_plate == "UNKNOWN":
            self.logger.info("[OCR] no valid TW plate matched")

        return best_plate, best_conf
