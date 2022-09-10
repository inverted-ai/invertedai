import os
from dotenv import load_dotenv
from invertedai_drive.drive import Drive, Config
from invertedai_drive.utils import Jupyter_Render, IAI_Logger, Client
from invertedai_drive.simulators import CarlaEnv, CarlaSimulationConfig


load_dotenv()
dev = os.environ.get("DEV", False)
if dev:
    dev_url = os.environ.get("DEV_URL", "http://localhost:8000")
log_level = os.environ.get("LOG_LEVEL", "WARNING")
log_console = os.environ.get("LOG_CONSOLE", 1)
log_file = os.environ.get("LOG_FILE", 0)
api_key = os.environ.get("API_KEY", "")

logger = IAI_Logger(level=log_level, consoel=bool(log_console), log_file=bool(log_file))

client = Client(api_key)

__all__ = [
    Drive,
    Config,
    Jupyter_Render,
    CarlaEnv,
    CarlaSimulationConfig,
    logger,
    client,
]
