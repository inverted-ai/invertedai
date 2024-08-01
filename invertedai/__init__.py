import os
import importlib.metadata
__version__ = importlib.metadata.version("invertedai")
from distutils.util import strtobool

from invertedai.api.light import light
from invertedai.api.location import location_info
from invertedai.api.initialize import initialize, async_initialize
from invertedai.api.drive import drive, async_drive
from invertedai.api.blame import blame, async_blame
from invertedai.cosimulation import BasicCosimulation
from invertedai.utils import Jupyter_Render, IAILogger, Session
from invertedai.large.initialize import get_regions_in_grid, get_number_of_agents_per_region_by_drivable_area, insert_agents_into_nearest_region, get_regions_default, large_initialize
from invertedai.large.drive import large_drive

dev = strtobool(os.environ.get("IAI_DEV", "false"))
if dev:
    dev_url = os.environ.get("IAI_DEV_URL", "http://localhost:8000")
commercial_url = "https://api.inverted.ai/v0/aws/m1"
academic_url = "https://api.inverted.ai/v0/academic/m1"

log_level = os.environ.get("IAI_LOG_LEVEL", "WARNING")
log_console = strtobool(os.environ.get("IAI_LOG_CONSOLE", "true"))
log_file = strtobool(os.environ.get("IAI_LOG_FILE", "false"))
api_key = os.environ.get("IAI_API_KEY", "")


logger = IAILogger(level=log_level, consoel=bool(log_console), log_file=bool(log_file))

session = Session()
if api_key:
    session.add_apikey(api_key)
add_apikey = session.add_apikey
use_mock_api = session.use_mock_api

if strtobool(os.environ.get("IAI_MOCK_API", "false")):
    use_mock_api()

model_resources = {
    "initialize": ("post", "/initialize"),
    "blame": ("post", "/blame"),
    "drive": ("post", "/drive"),
    "location_info": ("get", "/location_info"),
    "light": ("get", "/light"),
    "test": ("get", "/test"),
}
__all__ = [
    "BasicCosimulation",
    "Jupyter_Render",
    "logger",
    "session",
    "add_apikey",
    "use_mock_api",
    "blame",
    "drive",
    "initialize",
    "location_info",
    "light",
    "async_initialize",
    "async_drive",
    "async_blame",
]