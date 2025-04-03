import logging
import os

def get_logger(name: str) -> logging.Logger:
    # Cria diretório de logs se não existir
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formato do log
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Handler para log em arquivo
    file_handler = logging.FileHandler("logs/pipeline.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Handler para log no console (opcional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Evita adicionar múltiplos handlers ao mesmo logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
