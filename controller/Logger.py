# Controller/Logger.py
import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir="log"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = os.path.join(self.log_dir, f"{self.today}.txt")

    def _write(self, level: str, message: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] [{level}] {message}"

        # console
        print(line)

        # file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, message: str):
        self._write("INFO", message)

    def warn(self, message: str):
        self._write("WARN", message)

    def error(self, message: str):
        self._write("ERROR", message)
