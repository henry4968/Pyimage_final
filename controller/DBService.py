import sqlite3


class DBService:
    def __init__(self, db_path, logger=None):
        self.logger = logger
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        if self.logger:
            self.logger.info("DBService connected")

    def insert_alpr_info(self, filename, seq, x1, y1, x2, y2, yolo_conf, plate_text, ocr_conf):
        sql = """
        INSERT OR REPLACE INTO alpr_info
        (filename, seq, x1, y1, x2, y2, yolo_conf, plate_text, ocr_conf)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(sql, (filename, seq, x1, y1, x2, y2, yolo_conf, plate_text, ocr_conf))
        self.conn.commit()

        if self.logger:
            self.logger.info(f"[DB] insert plate | file={filename} seq={seq} plate={plate_text}")

    def update_vehicle_info(self, filename, seq, vehicle_type, vehicle_conf):
        sql = """
        UPDATE alpr_info
        SET vehicle_type = ?, vehicle_conf = ?
        WHERE filename = ? AND seq = ?
        """
        self.conn.execute(sql, (vehicle_type, vehicle_conf, filename, seq))
        self.conn.commit()

        if self.logger:
            self.logger.info(f"[DB] update vehicle | file={filename} seq={seq} type={vehicle_type}")
