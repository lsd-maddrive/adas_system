import os


def is_debug():
    return True if os.getenv('DEBUG') else False
