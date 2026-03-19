"""
InvertedAI Python SDK for AI-powered NPC control in autonomous driving simulations.

All public symbols are available directly from this top-level package. Always import
from here rather than from internal submodules:

    # Correct
    import invertedai as iai
    from invertedai import WaypointManager, AgentType, ScenePlotter

    # Avoid — internal paths are not part of the public API and may change
    from invertedai.helpers.waypoints import WaypointManager
    from invertedai.common import AgentType

Internal module paths are subject to change between versions without notice.
"""
import os
import warnings
import importlib.metadata

import invertedai._state as _state

__version__ = importlib.metadata.version("invertedai")

# --- Core API functions ---
from invertedai.api.light import light
from invertedai.api.location import location_info
from invertedai.api.initialize import initialize, async_initialize
from invertedai.api.drive import drive, async_drive
from invertedai.api.blame import blame, async_blame

# --- High-level simulation wrappers ---
from invertedai.cosimulation import BasicCosimulation

# --- Session and utilities ---
from invertedai.utils import Jupyter_Render, IAILogger, Session, ScenePlotter, get_default_agent_properties

# --- Large-scale simulation ---
from invertedai.large.initialize import (
    get_regions_in_grid,
    get_number_of_agents_per_region_by_drivable_area,
    get_regions_default,
    large_initialize,
)
from invertedai.large.drive import large_drive

# --- Logging and diagnostics ---
from invertedai.logs.logger import LogWriter, LogReader
from invertedai.logs.diagnostics import DiagnosticTool
from invertedai.logs.debug_logger import DebugLogger

# --- Helpers ---
from invertedai.helpers.waypoints import WaypointManagerConfig, WaypointManager
from invertedai.helpers.simulation_manager import SimulationAgentDict, SimulationManager

# --- Common data types ---
from invertedai.common import (
    AgentType,
    AgentProperties,
    AgentState,
    AgentAttributes,
    AgentData,
    RecurrentState,
    Point,
    Origin,
    LocationMap,
    Image,
    TrafficLightState,
    LightRecurrentState,
    InfractionIndicators,
    StaticMapActor,
)

warnings.filterwarnings(action="once", message=".*agent_attributes.*")


def strtobool(value: str) -> bool:
    value = value.lower()
    if value in ("y", "yes", "on", "1", "true", "t"):
        return True
    return False


dev = strtobool(os.environ.get("IAI_DEV", "false"))
_state.dev = dev
if dev:
    dev_url = os.environ.get("IAI_DEV_URL", "http://localhost:8000")
    _state.dev_url = dev_url
commercial_url = _state.commercial_url
academic_url = _state.academic_url

log_level = os.environ.get("IAI_LOG_LEVEL", "WARNING")
log_console = strtobool(os.environ.get("IAI_LOG_CONSOLE", "true"))
log_file = strtobool(os.environ.get("IAI_LOG_FILE", "false"))
api_key = os.environ.get("IAI_API_KEY", "")
debug_logger_path = os.environ.get("IAI_LOGGER_PATH", None)

debug_logger = None
if debug_logger_path is not None:
    debug_logger = DebugLogger(os.path.join(debug_logger_path))
logger = IAILogger(level=log_level, consoel=bool(log_console), log_file=bool(log_file))

session = Session(debug_logger)
if api_key:
    session.add_apikey(api_key)
add_apikey = session.add_apikey
use_mock_api = session.use_mock_api

_state.session = session
_state.logger = logger
_state.debug_logger = debug_logger

if strtobool(os.environ.get("IAI_MOCK_API", "false")):
    use_mock_api()

model_resources = _state.model_resources

# Deprecated name aliases. When an existing public name is renamed or moved,
# add an entry here instead of breaking users' existing scripts:
#   "OldName": ("new_attr_name", "human-readable replacement for the warning message")
_DEPRECATED: dict[str, tuple[str, str]] = {
    # Example (uncomment and adapt when a rename happens):
    # "OldClassName": ("new_class_name", "invertedai.NewClassName"),
}


def __getattr__(name: str):
    """Intercept attribute access to handle deprecated or moved names."""
    if name in _DEPRECATED:
        new_attr, display_name = _DEPRECATED[name]
        warnings.warn(
            f"'invertedai.{name}' is deprecated and will be removed in a future version. "
            f"Use '{display_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_attr]
    raise AttributeError(f"module 'invertedai' has no attribute {name!r}")


__all__ = [
    # Core API
    "initialize",
    "async_initialize",
    "drive",
    "async_drive",
    "blame",
    "async_blame",
    "location_info",
    "light",
    # High-level wrappers
    "BasicCosimulation",
    # Large-scale simulation
    "large_initialize",
    "large_drive",
    "get_regions_default",
    "get_regions_in_grid",
    "get_number_of_agents_per_region_by_drivable_area",
    # Helpers
    "WaypointManager",
    "WaypointManagerConfig",
    "SimulationAgentDict",
    "SimulationManager",
    # Session and utilities
    "session",
    "add_apikey",
    "use_mock_api",
    "Jupyter_Render",
    "IAILogger",
    "Session",
    "ScenePlotter",
    "get_default_agent_properties",
    "logger",
    # Logging and diagnostics
    "LogWriter",
    "LogReader",
    "DiagnosticTool",
    "DebugLogger",
    # Common data types
    "AgentType",
    "AgentProperties",
    "AgentState",
    "AgentAttributes",
    "AgentData",
    "RecurrentState",
    "Point",
    "Origin",
    "LocationMap",
    "Image",
    "TrafficLightState",
    "LightRecurrentState",
    "InfractionIndicators",
    "StaticMapActor",
]
