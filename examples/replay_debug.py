from invertedai.logs.debug_logger import DebugLogger

log_path = "iai_debug_logs/iai_log_2025-09-25_00:23:36:978301_UTC.json"

DebugLogger.read_log_from_path(
    debug_log_path=log_path,
    #is_visualize_log=True,   # create a GIF
    #gif_name="my_debug_vis.gif",
    fov=200
)