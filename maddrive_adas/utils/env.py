import os
import pathlib


def is_debug():
    return True if os.getenv('DEBUG_DETECTOR') else False


def get_project_root() -> pathlib.Path:
    assert 'VIRTUAL_ENV' in os.environ, f'Run {__file__} in venv to resolve project root'
    return pathlib.Path(os.getenv('VIRTUAL_ENV')).parent.resolve()
