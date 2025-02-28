import logging

import colorlog

logger = logging.getLogger("aido")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(name)s: %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red"
        }
    )
)
logger.addHandler(console_handler)
