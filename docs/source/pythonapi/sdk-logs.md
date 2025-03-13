[scenario-log-example-link]: https://github.com/inverted-ai/invertedai/blob/master/examples/scenario_log_example.py
[diagnostic-log-example-link]: https://github.com/inverted-ai/invertedai/blob/master/invertedai/logs/diagnostics_logger.py

# Logs & Debugging

## Overview
Capturing data to and from the API can be useful for several purposes. The IAI SDK provides features for producing logs that can help in several use cases.

# Debug Logs

## Description
Debug logs are relatively simple in comparison to scenario logs. These logs capture the raw data of all requests and response to and from the API. In the case of the {ref}`LARGE_DRIVE` and {ref}`LARGE_INITIALIZE` tools, the data is formatted into its serializable form before being divided into individual API calls.

```{eval-rst}
.. autoclass:: invertedai.logs.debug_logger.DebugLogger
   :members:
```

## Example Usage

### Capturing Debug Logs
Capturing debug logs is a simple process. Simply set the IAI_LOGGER_PATH environment variable. For example, in a Linux terminal, run the following command to set the path of a directory at which the debug logs will be written:

```bash
export IAI_LOGGER_PATH="<INSERT_DIRECTORY_PATH_HERE>"
```

If the directory does not exist, the python script will attempt to create the directory so that JSON debug logs may be written to that path.

### Running Diagnostics
While debug logs can be useful in capturing implementation issues, parsing the raw data can be difficult. The diagnostic tool can be used to check for common mistakes that MIGHT cause potential issues. The diagnostic tool will parse a debug log and print information on the command line regarding what could be causing degradation in performance. The diagnostic tool can be run directly by calling the [diagnostic script][diagnostic-log-example-link] with a path to the debug log file.

```{eval-rst}
.. autoclass:: invertedai.logs.diagnostics.DiagnosticTool
   :members:
```

# Scenario Logs

## Description
This log type is designed primarily to capture scenarios and simulation rollouts of interest. These scenarios can then be loaded, replayed, and modified using 
the same tool. The data format and the specific tools with their primary functions are shown below.

```{eval-rst}
.. autoclass:: invertedai.logs.logger.ScenarioLog
   :members:
   :undoc-members:
   :exclude-members: model_config, model_fields
```
---
```{eval-rst}
.. autoclass:: invertedai.logs.logger.LogWriter
   :members:
```
---
```{eval-rst}
.. autoclass:: invertedai.logs.logger.LogReader
   :members:
```

## Example Usage
Please follow the following link to see an example of how to run a [scenario log example][scenario-log-example-link]. This example demonstrates running a sample scenario then writing to a log file, loading the sample log and visualizing it, then replaying the log but modifying it at a time step of interest.


