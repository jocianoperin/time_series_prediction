import logging
import os

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(processName)s | %(levelname)s | %(message)s")

    # --- FILE HANDLER -----------------------------------------
    log_path = log_file or "logs/pipeline.log"
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(fmt)

    # --- CONSOLE HANDLER (opcional) ---------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger