import logging

def get_logger(name="quantbot"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        file_handler = logging.FileHandler("quantbot.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
