# Interesting tricks:
#   https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094

try:
    from importlib import metadata as importlib_metadata
    __version__ = importlib_metadata.version(__package__)
except Exception as ex:
    print(f"Failed to import package through `importlib_metadata`: {ex}")  # noqa
    __version__ = "0.0.0"
