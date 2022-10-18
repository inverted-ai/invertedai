import os
from distutils.util import strtobool

from dotenv import load_dotenv
from invertedai.api_resources import (
    drive,
    initialize,
    location_info,
)
from invertedai.utils import Jupyter_Render, IAILogger, Session

load_dotenv()
dev = os.environ.get("DEV", False)
if dev:
    dev_url = os.environ.get("DEV_URL", "http://localhost:8000")
log_level = os.environ.get("LOG_LEVEL", "WARNING")
log_console = os.environ.get("LOG_CONSOLE", 1)
log_file = os.environ.get("LOG_FILE", 0)
api_key = os.environ.get("API_KEY", "")

logger = IAILogger(level=log_level, consoel=bool(log_console), log_file=bool(log_file))

session = Session(api_key)
add_apikey = session.add_apikey
use_mock_api = session.use_mock_api

if strtobool(os.environ.get("IAI_MOCK_API", "false")):
    use_mock_api()

model_resources = {
    "initialize": ("get", "/initialize"),
    "drive": ("post", "/drive"),
    "location_info": ("get", "/location_info"),
    "available_locations": ("get", "/available_locations"),
}
__all__ = [
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
