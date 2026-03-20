"""
Internal package state, initialized by invertedai/__init__.py at startup.

Internal modules should import this module to access shared session, logger,
and configuration rather than importing the top-level invertedai package,
which creates circular imports.

Usage inside implementation files:

    import invertedai._state as _state
    # ...
    _state.session.request(...)
    _state.logger.warning(...)
"""
import importlib.metadata

__version__ = importlib.metadata.version("invertedai")

commercial_url = "https://api.inverted.ai/v0/aws/m1"
academic_url = "https://api.inverted.ai/v0/academic/m1"
dev = False
dev_url = None

model_resources = {
    "initialize": ("post", "/initialize"),
    "blame": ("post", "/blame"),
    "drive": ("post", "/drive"),
    "location_info": ("get", "/location_info"),
    "light": ("get", "/light"),
    "test": ("get", "/test"),
}

# Set by invertedai/__init__.py during package initialization.
session = None
logger = None
debug_logger = None
