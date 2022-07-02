import logging

# TODO - move this to executable scripts, not package level

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logger.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
