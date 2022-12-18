import os


def is_debug():
    return True if os.getenv('DEBUG_DETECTOR') else False
