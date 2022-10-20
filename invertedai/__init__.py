import os
from distutils.util import strtobool

from invertedai.api_resources import (
    drive,
    initialize,
    location_info,
)
from invertedai.cosimulation import BasicCosimulation
from invertedai.utils import Jupyter_Render, IAILogger, Session

dev = os.environ.get("IAI_DEV", False)
if dev:
    dev_url = os.environ.get("IAI_DEV_URL", "http://localhost:8000")
log_level = os.environ.get("IAI_LOG_LEVEL", "WARNING")
log_console = os.environ.get("IAI_LOG_CONSOLE", 1)
log_file = os.environ.get("IAI_LOG_FILE", 0)
api_key = os.environ.get("IAI_API_KEY", "")

logger = IAILogger(level=log_level, consoel=bool(log_console), log_file=bool(log_file))

session = Session(api_key)
add_apikey = session.add_apikey
use_mock_api = session.use_mock_api

if strtobool(os.environ.get("IAI_MOCK_API", "false")):
    use_mock_api()

model_resources = {
    "initialize": ("post", "/initialize"),
    "drive": ("post", "/drive"),
    "location_info": ("get", "/location_info"),
}
__all__ = [
    "BasicCosimulation",
    "drive",
    "initialize",
    "location_info",
    "Jupyter_Render",
    "logger",
    "session",
    "add_apikey",
    "use_mock_api",
    "location_info",
]
