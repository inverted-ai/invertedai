import os
from dotenv import load_dotenv
from invertedai.api_resources import drive, initialize, get_map, available_locations
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
model_resources = {
    "initialize": ("get", "/initialize"),
    "drive": ("post", "/drive"),
    "get_map": ("get", "/map"),
    "available_locations": ("get", "/available_locations"),
}
try:
    from invertedai.simulators import CarlaEnv, CarlaSimulationConfig
except:
    logger.warning(
        "Cannot import CarlaEnv\n"
        + "Carla Python API is not installed\n"
        + "Ignore these warnings if you are not running Carla"
    )


__all__ = [
    "drive",
    "initialize",
    "get_map",
    "Jupyter_Render",
    "logger",
    "session",
    "add_apikey",
    "get_map",
    "available_locations",
]
