import os
from dotenv import load_dotenv
from invertedai.drive import drive, initialize
from invertedai.utils import Jupyter_Render, IAILogger, Session
from invertedai.simulators import CarlaEnv, CarlaSimulationConfig


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

__all__ = [
    "drive",
    "initialize",
    "Jupyter_Render",
    "CarlaEnv",
    "CarlaSimulationConfig",
    "logger",
    "session",
    "add_apikey",
]
