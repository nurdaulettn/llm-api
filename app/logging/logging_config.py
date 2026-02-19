import logging
import json
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("logs/app.log")
    handler.setFormatter(JsonFormatter())
    handler.stream.reconfigure(line_buffering=True)
    logger.addHandler(handler)