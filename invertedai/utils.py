import json
import os
import re
import csv
import math
import logging
import random
import time
import numpy as np
import warnings

from typing import Dict, Optional, List, Tuple, Union, Any
from copy import deepcopy
from pydantic import validate_call, validate_arguments

import requests
from requests import Response
from requests.auth import AuthBase
from requests.adapters import HTTPAdapter, Retry

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib import transforms

import invertedai as iai
import invertedai.api
import invertedai.api.config
from invertedai import error
from invertedai.future import to_thread
from invertedai.error import InvertedAIError
from invertedai.common import (
    AgentState, 
    AgentAttributes, 
    AgentProperties, 
    AgentType,
    RecurrentState,
    StaticMapActor,
    TrafficLightState, 
    TrafficLightStatesDict 
)

H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7
TIMEOUT_SECS = 600
MAX_RETRIES = 10
AGENT_SCOPE_FOV = 120

logger = logging.getLogger(__name__)

STATUS_MESSAGE = {
    403: "Access denied. Please check the provided API key.",
    429: "Throttled",
    502: "The server is having trouble communicating. This is usually a temporary issue. Please try again later.",
    504: "The server took too long to respond. Please try again later.",
    500: "The server encountered an unexpected issue. We're working to resolve this. Please try again later.",
}

Color = Tuple[float,float,float]
ColorList = List[Optional[Color]]

class Session:
    def __init__(self,debug_logger=None):
        self.session = requests.Session()
        self.session.mount(
            "https://",
            requests.adapters.HTTPAdapter(),
        )
        self.session.mount(
            "http://",
            requests.adapters.HTTPAdapter(),
        )
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "x-client-version": iai.__version__,
            }
        )
        self._base_url = self._get_base_url()
        self._max_retries = float("inf")
        self._status_force_list = [403, 408, 429, 500, 502, 503, 504]
        self._base_backoff = 1  # Base backoff time in seconds
        self._backoff_factor = 2
        self._jitter_factor = 0.5
        self._current_backoff = self._base_backoff
        self._max_backoff = None

        self._debug_logger = debug_logger

    @property
    def base_url(self):
        return self._base_url

    @property
    def max_retries(self):
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value):
        self._max_retries = value

    @property
    def status_force_list(self):
        return self._status_force_list

    @status_force_list.setter
    def status_force_list(self, value):
        self._status_force_list = value.copy()

    @property
    def base_backoff(self):
        return self._base_backoff

    @base_backoff.setter
    def base_backoff(self, value):
        self._base_backoff = value
        self.current_backoff = (
            self._base_backoff
        )  # Reset current_backoff when base_backoff changes

    @property
    def backoff_factor(self):
        return self._backoff_factor

    @backoff_factor.setter
    def backoff_factor(self, value):
        self._backoff_factor = value

    
    @property
    def current_backoff(self):
        return self._current_backoff
    
    @current_backoff.setter
    def current_backoff(self, value):
        self._current_backoff = value

    @property
    def max_backoff(self):
        return self._max_backoff
    
    @max_backoff.setter
    def max_backoff(self, value):
        self._max_backoff = value

    @property
    def jitter_factor(self):
        return self._jitter_factor
    
    @jitter_factor.setter
    def jitter_factor(self, value):
        self._jitter_factor = value


    def should_log(self, retry_count):
        return retry_count == 0 or math.log2(retry_count).is_integer()

    @base_url.setter
    def base_url(self, value):
        self._base_url = value

    def _verify_api_key(
        self, 
        api_token: str, 
        verifying_url: str
    ):
        """
        Verifies the API key by making a request to the verifying URL.

        Args:
            api_token (str): The API token to be used for authentication.
            verifying_url (str): The URL to be used for verification.

        Returns:
            str: The final verifying URL after fallback (if applicable).

        Raises:
            error.AuthenticationError: If access is denied due to an invalid API key.
        """
        self.session.auth = APITokenAuth(api_token)
        response = self.session.request(method="get", url=verifying_url)
        if verifying_url == iai.commercial_url and response.status_code != 200:
            # Check for academic access in case the previous call to the commercial server fails.
            logger.warning(
                "Commercial access denied and fallback to check for academic access."
            )
            verifying_url = iai.academic_url
            response_acd = self.session.request(method="get", url=verifying_url)
            if response_acd.status_code == 200:
                self.base_url = verifying_url
                response = response_acd
            elif response_acd.status_code != 403:
                response = response_acd
        if response.status_code == 403:
            raise error.AuthenticationError(
                "Access denied. Please check the provided API key."
            )
        return verifying_url

    def add_apikey(
        self,
        api_token: str = "",
        key_type: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """
        Bind an API key to the session for authentication.

        Args:
            api_token (str): The API key to be added. Defaults to an empty string.
            key_type (str, optional): The type of API key. Defaults to None. When passed, the base_url will be set according to the key_type.
            url (str, optional): The URL to be used for the request. Defaults to None. When passed, the base_rul will be set to the passed value 
            and the key_type will be ignored.

        Raises:
            InvalidAPIKeyError: If the API key is empty and not in development mode.
            InvalidAPIKeyError: If the key_type is invalid.
            AuthenticationError: If access is denied due to an invalid API key.
            APIError: If the server encounters an error or is unable to perform the requested method.
        """
        if not iai.dev and not api_token:
            raise error.InvalidAPIKeyError("Empty API key received.")
        if url is None:
            request_url = self._get_base_url()
        if key_type is not None and key_type not in ["commercial", "academic"]:
            raise error.InvalidAPIKeyError(f"Invalid API key type: {key_type}.")
        if key_type == "academic":
            request_url = iai.academic_url
        elif key_type == "commercial":
            request_url = iai.commercial_url
        if url is not None:
            request_url = url
        self.base_url = self._verify_api_key(api_token, request_url)

    def use_mock_api(
        self, 
        use_mock: bool = True
    ) -> None:
        invertedai.api.config.mock_api = use_mock
        if use_mock:
            iai.logger.warning(
                "Using mock Inverted AI API - predictions will be trivial"
            )

    async def async_request(
        self, 
        *args, 
        **kwargs
    ):
        return await to_thread(self.request, *args, **kwargs)

    def request(
        self, 
        model: str, 
        params: Optional[dict] = None, 
        data: Optional[dict] = None
    ):
        method, relative_path = iai.model_resources[model]
        
        if self._debug_logger is not None:
            request_data = data
            if params is not None:
                request_data = params
            self._debug_logger.append_request(model,request_data)

        response = self._request(
            method=method,
            relative_path=relative_path,
            params=params,
            json_body=data,
        )

        if self._debug_logger is not None:
            self._debug_logger.append_response(model,response)

        return response

    def _request(
        self,
        method,
        relative_path: str = "",
        params=None,
        headers=None,
        json_body=None,
        data=None,
    ) -> Dict:
        try:
            retries = 0
            while retries < self.max_retries:
                try:
                    response = self.session.request(
                        method=method,
                        params=params,
                        url=self.base_url + relative_path,
                        headers=headers,
                        data=data,
                        json=json_body,
                    )
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    logger.warning("Error communicating with IAI, will retry.")
                    response = None
                if response is not None and response.status_code not in self.status_force_list:
                    self.current_backoff = max(
                        self.base_backoff, self.current_backoff / self.backoff_factor
                    )
                    response.raise_for_status()
                    break
                else:
                    if self.jitter_factor is not None:
                        jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
                    else:
                        jitter = 0
                    if self.should_log(retries):
                        if response is not None:
                            logger.warning(
                                f"Retrying {relative_path}: Status {response.status_code}, Message {STATUS_MESSAGE.get(response.status_code, response.text)} Retry #{retries + 1}, Backoff {self.current_backoff} seconds"
                            )
                        else:
                            logger.warning(f"Retrying {relative_path}: No response received, Retry #{retries + 1}, Backoff {self.current_backoff} seconds")
                    time.sleep(min(self.current_backoff * (1 + jitter), self.max_backoff if self.max_backoff is not None else float("inf")))
                    self.current_backoff *= self.backoff_factor
                    if self.max_backoff is not None:
                        self.current_backoff = min(
                            self.current_backoff, self.max_backoff
                        )
                    retries += 1
            else:
                if response is not None:
                    response.raise_for_status()
                else:
                    error.APIConnectionError(
                        "Error communicating with IAI", should_retry=True)


        except requests.exceptions.ConnectionError as e:
            raise error.APIConnectionError(
                "Error communicating with IAI", should_retry=True
            ) from None
        except requests.exceptions.Timeout as e:
            raise error.APIConnectionError("Error communicating with IAI") from None
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 403:
                raise error.AuthenticationError(STATUS_MESSAGE[403]) from None
            elif e.response.status_code in [400, 422]:
                raise error.InvalidRequestError(e.response.text, param="") from None
            elif e.response.status_code == 404:
                raise error.ResourceNotFoundError(e.response.text) from None
            elif e.response.status_code == 408:
                raise error.RequestTimeoutError(e.response.text) from None
            elif e.response.status_code == 413:
                raise error.RequestTooLarge(e.response.text) from None
            elif e.response.status_code == 429:
                raise error.RateLimitError(STATUS_MESSAGE[429]) from None
            elif e.response.status_code == 502:
                raise error.APIError(STATUS_MESSAGE[502]) from None
            elif e.response.status_code == 503:
                raise error.RequestTimeoutError(e.response.text) from None
            elif e.response.status_code == 504:
                raise error.ServiceUnavailableError(STATUS_MESSAGE[504]) from None
            elif 400 <= e.response.status_code < 500:
                raise error.APIError(e.response.text) from None
            else:
                raise error.APIError(STATUS_MESSAGE[500]) from None
        iai.logger.info(
            iai.logger.logfmt(
                "IAI API response",
                path=self.base_url,
                response_code=response.status_code,
            )
        )
        try:
            data = json.loads(response.content)
        except json.decoder.JSONDecodeError:
            raise error.APIError(
                f"HTTP code {response.status_code} from API ({response.content})",
                response.content,
                response.status_code,
                headers=response.headers,
            )
        return data

    def _get_base_url(self) -> str:
        """
        This function returns the endpoint for API calls, which includes the
        version and other endpoint specifications.
        The method path should be appended to the base_url
        """
        if not iai.dev:
            base_url = iai.commercial_url  # Default to commercial when initializing.
        else:
            base_url = iai.dev_url
        # TODO: Add endpoint option and versioning to base_url
        return base_url

    def _handle_error_response(
        self, 
        rbody, 
        rcode, 
        resp, 
        rheaders, 
        stream_error=False
    ):
        try:
            error_data = resp["error"]
        except (KeyError, TypeError):
            raise error.APIError(
                "Invalid response object from API: %r (HTTP response code "
                "was %d)" % (rbody, rcode),
                rbody,
                rcode,
                resp,
            )

        if "internal_message" in error_data:
            error_data["message"] += "\n\n" + error_data["internal_message"]

        iai.logger.info(
            iai.logger.logfmt(
                "IAI API error received",
                error_code=error_data.get("code"),
                error_type=error_data.get("type"),
                error_message=error_data.get("message"),
                error_param=error_data.get("param"),
                stream_error=stream_error,
            )
        )

        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            return error.RateLimitError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode in [400, 404, 415]:
            return error.InvalidRequestError(
                error_data.get("message"),
                error_data.get("param"),
                error_data.get("code"),
                rbody,
                rcode,
                resp,
                rheaders,
            )
        elif rcode == 401:
            return error.AuthenticationError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 403:
            return error.PermissionError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 409:
            return error.TryAgain(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        else:
            return error.APIError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )

    def _interpret_response_line(
        self, 
        result
    ):
        rbody = result.content
        rcode = result.status_code
        rheaders = result.headers

        if rcode == 503:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            data = json.loads(rbody)
        except BaseException:
            raise error.APIError(
                f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders
            )
        if "error" in data or not 200 <= rcode < 300:
            raise self._handle_error_response(rbody, rcode, data, rheaders)

        return data


@validate_call
def get_default_agent_properties(
    agent_count_dict: Dict[AgentType,int],
    use_agent_properties: Optional[bool] = True
) -> List[Union[AgentAttributes,AgentProperties]]:
    """
    Function that outputs a list a AgentAttributes with minimal default settings. 
    Mainly meant to be used to pad a list of AgentAttributes to send as input to
    initialize(). This list is created by reading a dictionary containing the
    desired agent types with the agent count for each type respectively.
    If desired to use deprecate AgentAttributes instead of AgentProperties, set the
    use_agent_properties flag to False.
    """

    agent_attributes_list = []

    for agent_type, agent_count in agent_count_dict.items():
        for _ in range(agent_count):
            if use_agent_properties:
                agent_properties = AgentProperties(agent_type=agent_type)
                agent_attributes_list.append(agent_properties)
            else:
                agent_attributes_list.append(AgentAttributes.fromlist([agent_type]))

    return agent_attributes_list


@validate_call
def convert_attributes_to_properties(
    attributes: AgentAttributes
) -> AgentProperties:
    """
    Convert deprecated AgentAttributes data type to AgentProperties.
    """

    properties = AgentProperties(
        length=attributes.length,
        width=attributes.width,
        rear_axis_offset=attributes.rear_axis_offset,
        agent_type=attributes.agent_type,
        waypoint=attributes.waypoint
    )

    return properties


@validate_call
def iai_conditional_initialize(
    location: str, 
    agent_type_count: Dict[str,int],
    location_of_interest: Tuple[float] = (0,0),
    recurrent_states: Optional[List[RecurrentState]] = None,
    agent_properties: Optional[List[AgentProperties]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None, 
    get_birdview: Optional[bool] = False,
    get_infractions: Optional[bool] = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None
):
    """
    A utility function to run initialize with conditional agents located at arbitrary distances from the location
    of interest. Only agents within a defined distance of the location of interest are passed to initialize as 
    conditional. Agents outisde of this distance are padded on to the initialize response, including their reccurent
    states. Recurrent states must be provided for all agents, otherwise this function behaves like :func:`initialize`.
    Please refer to the documentation for :func:`initialize` for more information.

    Arguments
    ----------
    location:
        Location name in IAI format.

    agent_type_count:
        A dictionary containing valid AgentType strings as keys mapped to an integer value specifying the desired
        number of agents of that type to initialize.

    location_of_interest:
        Optional coordinates for spawning agents with the given location as center instead of the default map center
    
    See Also
    --------
    :func:`initialize`
    """

    conditional_agent_properties = []
    conditional_agent_states_indexes = []
    conditional_recurrent_states = []
    outside_agent_states = []
    outside_agent_properties = []
    outside_recurrent_states = []

    current_agent_states = states_history[-1]
    conditional_agent_type_count = deepcopy(agent_type_count)
    for i in range(len(current_agent_states)):
        agent_state = current_agent_states[i]
        dist = math.dist(location_of_interest, (agent_state.center.x, agent_state.center.y))
        if dist < AGENT_SCOPE_FOV:
            conditional_agent_states_indexes.append(i)
            conditional_agent_properties.append(agent_properties[i])
            conditional_recurrent_states.append(recurrent_states[i])

            conditional_agent_type = agent_properties[i].agent_type
            if conditional_agent_type in conditional_agent_type_count:
                conditional_agent_type_count[conditional_agent_type] -= 1
                if conditional_agent_type_count[conditional_agent_type] <= 0:
                    del conditional_agent_type_count[conditional_agent_type]

        else:
            outside_agent_states.append(agent_state)
            outside_agent_properties.append(agent_properties[i])
            outside_recurrent_states.append(recurrent_states[i])

    if not conditional_agent_type_count: #The dictionary is empty.
        iai.logger.warning("Agent count requirement already satisfied, no new agents initialized.")

    padded_agent_properties = get_default_agent_properties(conditional_agent_type_count)
    conditional_agent_properties.extend(padded_agent_properties)

    conditional_agent_states = [[]*len(conditional_agent_states_indexes)]
    for ts in range(len(conditional_agent_states)):
        for agent_index in conditional_agent_states_indexes:
            conditional_agent_states[ts].append(states_history[ts][agent_index])

    response = invertedai.api.initialize(
        location = location,
        agent_properties = conditional_agent_properties,
        states_history = conditional_agent_states,
        location_of_interest = location_of_interest,
        traffic_light_state_history = traffic_light_state_history,
        get_birdview = get_birdview,
        get_infractions = get_infractions,
        random_seed = random_seed,
        api_model_version = api_model_version
    )
    response.agent_properties = response.agent_properties + outside_agent_properties
    response.agent_states = response.agent_states + outside_agent_states
    response.recurrent_states = response.recurrent_states + outside_recurrent_states

    return response


class APITokenAuth(AuthBase):
    def __init__(
        self, 
        api_token
    ):
        self.api_token = api_token

    def __call__(
        self, 
        r
    ):
        r.headers["x-api-key"] = self.api_token
        r.headers["api-key"] = self.api_token
        return r


def Jupyter_Render():
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np

    class Jupyter_Render(widgets.HBox):
        def __init__(self):
            super().__init__()
            output = widgets.Output()
            self.buffer = [np.zeros([128, 128, 3], dtype=np.uint8)]

            with output:
                self.fig, self.ax = plt.subplots(
                    constrained_layout=True, figsize=(5, 5)
                )
            self.im = self.ax.imshow(self.buffer[0])
            self.ax.set_axis_off()

            self.fig.canvas.toolbar_position = "bottom"

            self.max = 0
            # define widgets
            self.play = widgets.Play(
                value=0,
                min=0,
                max=self.max,
                step=1,
                description="Press play",
                disabled=False,
            )
            self.int_slider = widgets.IntSlider(
                value=0, min=0, max=self.max, step=1, description="Frame"
            )

            controls = widgets.HBox(
                [
                    self.play,
                    self.int_slider,
                ]
            )
            controls.layout = self._make_box_layout()
            widgets.jslink((self.play, "value"), (self.int_slider, "value"))
            output.layout = self._make_box_layout()

            self.int_slider.observe(self.update, "value")
            self.children = [controls, output]

        def update(
            self, 
            change
        ):
            self.im.set_data(self.buffer[self.int_slider.value])
            self.fig.canvas.draw()

        def add_frame(
            self, 
            frame
        ):
            self.buffer.append(frame)
            self.int_slider.max += 1
            self.play.max += 1
            self.int_slider.value = self.int_slider.max
            self.play.value = self.play.max

        def _make_box_layout(self):
            return widgets.Layout(
                border="solid 1px black",
                margin="0px 10px 10px 0px",
                padding="5px 5px 5px 5px",
            )

    return Jupyter_Render()


class IAILogger(logging.Logger):
    def __init__(
        self,
        name: str = "IAILogger",
        level: str = "WARNING",
        consoel: bool = True,
        log_file: bool = False,
    ) -> None:

        level = logging.getLevelName(level)
        log_level = level if isinstance(level, int) else 30
        super().__init__(name, log_level)
        if consoel:
            consoel_handler = logging.StreamHandler()
            self.addHandler(consoel_handler)
        if log_file:
            file_handler = logging.FileHandler("iai.log")
            self.addHandler(file_handler)

    @staticmethod
    def logfmt(
        message, 
        **params
    ):
        props = dict(message=message, **params)

        def fmt(key, val):
            # Handle case where val is a bytes or bytesarray
            if hasattr(val, "decode"):
                val = val.decode("utf-8")
            # Check if val is already a string to avoid re-encoding into ascii.
            if not isinstance(val, str):
                val = str(val)
            if re.search(r"\s", val):
                val = repr(val)
            # key should already be a string
            if re.search(r"\s", key):
                key = repr(key)
            return f"{key}={val}"

        return " ".join([fmt(key, val) for key, val in sorted(props.items())])


def rot(rot):
    """Rotate in 2d"""
    return np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])


class ScenePlotter():
    """
    A class providing features for handling the data visualization of a scene involving IAI data.

    Arguments
    ----------
    map_image:
        An image used as the background for the visualization decoded from the birdview map taken from location info.
    fov:
        A single float value representing the field of view of the visualization that can be taken from location info.
    xy_offset:
        A tuple coordinate of the center of the map in metres that can be taken from location info.
    static_actors:
        A list of StaticMapActor objects representing objects such as traffic lights that can be taken from location info.
    open_drive: 
        If using an ASAM OpenDRIVE format map for visualization, this string parameter is used to indicate the path to the corresponding CSV file.
    resolution: 
        The desired resolution of the map image expressed as a Tuple with two integers for the width and height respectively.
    dpi:
        Dots per inch to define the level of detail in the image.
    left_hand_coordinates:
        Boolean flag dictating whether the X-coordinates of all agents and actors should be reversed to fit a left hand coordinate system.

    Keyword Arguments
    -----------------
    map_image:
        Base image onto which the scene is visualized. This parameter must be provided if using an ASAM OpenDRIVE format map.
    fov: float
        The field of view in meters corresponding to the map_image attribute. This parameter must be provided if using an ASAM OpenDRIVE format map.
    xy_offset:
        The left-hand offset for the center of the map image. This parameter must be provided if using an ASAM OpenDRIVE format map.
    static_actors:
        A list of static actor agents (e.g. traffic lights) represented as StaticMapActor objects, in the scene. This parameter must be provided 
        if using an ASAM OpenDRIVE format map.
    
    See Also
    --------
    :func:`location_info`
    """
    def __init__(
        self,
        map_image: Optional[np.array] = None,
        fov: Optional[float] = None,
        xy_offset: Optional[Tuple[float,float]] = None,
        static_actors: Optional[List[StaticMapActor]] = None,
        open_drive: Optional[str] = None, 
        resolution: Tuple[int,int] = (640, 480), 
        dpi: float = 100,
        left_hand_coordinates: bool = False,
        **kwargs
    ):

        self._left_hand_coordinates = left_hand_coordinates
        
        self._open_drive = open_drive
        self._dpi = dpi
        self._resolution = resolution
        
        self.map_image = map_image
        self.fov = fov
        self.xy_offset = xy_offset
        self.static_actors = static_actors

        self.traffic_lights = {static_actor.actor_id: static_actor for static_actor in self.static_actors if static_actor.agent_type == 'traffic_light'}

        if self._open_drive is None:
            self.extent = (- self.fov / 2 + self.xy_offset[0], self.fov / 2 + self.xy_offset[0]) + \
                (- self.fov / 2 + self.xy_offset[1], self.fov / 2 + self.xy_offset[1])

        self.traffic_light_colors = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "yellow": (1.0, 0.8, 0.0),
        }

        self.agent_c = (0.125,0.29,0.529)
        self.agent_ped_c = (1.0, 0.75, 0.8)
        self.cond_c = (0.78, 0.0, 0.0)
        self.dir_c = (0.392,1.0,1.0)
        self.v_c = (0.2, 0.75, 0.2)

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.frame_label = None
        self.current_ax = None

        self.numbers = None

        self.reset_recording()

        self.waypoint_markers = {}
        
    def reset_recording(self):
        """
        Explicitly reset the recording and remove the previous agent state, agent attribute, traffic light, and agent style data.
        """

        self.agent_states_history = None
        self.traffic_lights_history = None
        self.agent_properties = None
        self.waypoint_marker_history = None
        
        self.agent_face_colors = None 
        self.agent_edge_colors = None 

    @validate_arguments
    def initialize_recording(
        self,
        agent_states: List[AgentState], 
        agent_attributes: Optional[List[AgentAttributes]] = None, 
        agent_properties: Optional[List[AgentProperties]] = None,
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None 
    ):
        """
        Record the initial state of the scene to be visualized. This function also acts as an implicit reset of the recording and removes previous 
        agent state, agent attribute, traffic light, and agent style data.

        Arguments
        ----------
        agent_states:
            A list of AgentState objects corresponding to the initial time step to be visualized.
        agent_attributes:
            Static attributes of the agents present in the initial step of the simulation. We assume every agent is a rectangle obeying a kinematic 
            bicycle model. The attributes of each agent respectively may not change but if agents are added or removed from the simulation, this 
            list will change.
        agent_properties:
            Static properties of the agent (with the AgentProperties data type), present in the initial step of the simulation. We assume every 
            agent is a rectangle obeying a kinematic bicycle model. The properties of each agent respectively may not change but if agents are added
            or removed from the simulation, this list will change.
        traffic_light_states:
            Optional parameter containing the state of the traffic lights corresponding to the initial time step to be visualized. This parameter 
            should only be used if the corresponding map contains traffic light static actors.
        """

        assert (agent_attributes is not None) ^ (agent_properties is not None), \
            "Either agent_attributes or agent_properties is populated. Populating both or neither field is invalid."

        if agent_attributes is not None:
            self.agent_properties = [[convert_attributes_to_properties(attr) for attr in agent_attributes]]
            warnings.warn('agent_attributes is deprecated. Please use agent_properties.',category=DeprecationWarning)
        else:
            self.agent_properties = [agent_properties]

        self._validate_timestep_agents(
            agent_states=agent_states,
            agent_properties=self.agent_properties[0]
        )

        self.agent_states_history = [agent_states]
        self.traffic_lights_history = [traffic_light_states]

        self.agent_face_colors = None
        self.agent_edge_colors = None

    @validate_arguments
    def record_step(
        self,
        agent_states: List[AgentState], 
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None,
        agent_properties: Optional[List[AgentProperties]] = None
    ):
        """
        Record a single timestep of scene data to be used in a visualization.

        Arguments
        ----------
        agent_states:
            A list of AgentState objects corresponding to the initial time step to be visualized.
        traffic_light_states:
            Optional parameter containing the state of the traffic lights corresponding to the initial time step to be visualized. This parameter should
            only be used if the corresponding map contains traffic light static actors.
        agent_properties:
            A list of AgentProperties for the agents present during this time step. The indexes of these properties will be matched with corresponding
            indexes of the states given in the agent_states parameter. If no argument is given, it is assumed the agent properties have not changed since
            the previous time step, including which agents are present.
        """

        self.agent_states_history.append(agent_states)
        self.traffic_lights_history.append(traffic_light_states)

        if agent_properties is None:
            agent_properties = self.agent_properties[-1]
        self._validate_timestep_agents(
            agent_states=agent_states,
            agent_properties=agent_properties
        )
        self.agent_properties.append(agent_properties)

        if not hasattr(self, "waypoint_marker_history"):
            self.waypoint_marker_history = []

        frame_waypoints = [
            ap.waypoint if hasattr(ap, "waypoint") else None
            for ap in agent_properties
        ]
        self.waypoint_marker_history.append(frame_waypoints)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def plot_scene(
        self,
        agent_states: List[AgentState], 
        agent_attributes: Optional[List[AgentAttributes]] = None, 
        agent_properties: Optional[List[AgentProperties]] = None, 
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None, 
        ax: Optional[Axes] = None,
        numbers: Optional[List[int]] = None, 
        direction_vec: bool = True, 
        velocity_vec: bool = False,
        agent_face_colors: Optional[ColorList] = None,
        agent_edge_colors: Optional[ColorList] = None
    ):
        """
        Plot a single timestep of data then reset the recording. 

        Arguments
        ----------
        agent_states:
            A list of agents to be visualized in the image.
        agent_attributes: 
            Static attributes of the agent, which don’t change over the course of a simulation. We assume every agent is a rectangle obeying a kinematic
            bicycle model.
        agent_properties:
            Static attributes of the agent (with the AgentProperties data type), which don’t change over the course of a simulation. We assume every 
            agent is a rectangle obeying a kinematic bicycle model.
        traffic_light_states: 
            Optional parameter containing the state of the traffic lights to be visualized in the image. This parameter should only be used if the 
            corresponding map contains traffic light static actors.
        ax: 
            A matplotlib Axes object used to plot the image. By default, an Axes object is created if a value of None is passed.
        numbers: 
            A list of agent ID's that should be plotted in the image. By default this value is set to None.
        direction_vec:
            Flag to determine if a vector showing the vehicles direction should be plotted in the image. By default this flag is set to True.
        velocity_vec: 
            Flag to determine if the a vector showing the vehicles velocity should be plotted in the animation. By default this flag is set to False.
        agent_face_colors:
            An optional parameter containing a list of either RGB tuples indicating the desired color of the agent with the corresponding index ID. A value 
            of None in this list will use the default color.
        agent_edge_colors:
            An optional parameter containing a list of either RGB tuples indicating the desired color of a border around the agent with the corresponding 
            index ID. A value of None in this list will use the default color.

        """

        assert (agent_attributes is not None) ^ (agent_properties is not None), \
            "Either agent_attributes or agent_properties is populated. Populating both or neither field is invalid."

        if agent_attributes is not None:
            agent_properties = [convert_attributes_to_properties(attr) for attr in agent_attributes]
            warnings.warn('agent_attributes is deprecated. Please use agent_properties.',category=DeprecationWarning)

        self.initialize_recording(
            agent_states=agent_states, 
            agent_properties=agent_properties,
            traffic_light_states=traffic_light_states
        )

        self._validate_agent_style_data(
            agent_face_colors=agent_face_colors,
            agent_edge_colors=agent_edge_colors
        )

        self._plot_frame(
            idx=0, 
            ax=ax, 
            numbers=numbers, 
            direction_vec=direction_vec,
            velocity_vec=velocity_vec, 
            plot_frame_number=False
        )

        self.reset_recording()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def animate_scene(
        self,
        output_name: Optional[str] = None,
        start_idx: int = 0, 
        end_idx: int = -1,
        ax: Optional[Axes] = None,
        numbers: Optional[List[int]] = None, 
        direction_vec: bool = True, 
        velocity_vec: bool = False,
        plot_frame_number: bool = False, 
        mark_waypoints: bool = False,
        agent_face_colors: Optional[Union[ColorList,List[ColorList]]] = None,
        agent_edge_colors: Optional[Union[ColorList,List[ColorList]]] = None
    ) -> FuncAnimation:
        """
        Produce an animation of sequentially recorded steps. A matplotlib animation object can be returned and/or a gif saved of the scene.

        Parameters
        ----------
        output_name: 
            File name of the gif to which the animation will be saved.
        start_idx:
            The index of the time step from which the animation will begin. By default it is assumed all recorded steps are desired to be animated.
        end_idx:
            The index of the time step from which the animation will end. By default it is assumed all recorded steps are desired to be animated.
        ax: 
            A matplotlib Axes object used to plot the animation. By default, an Axes object is created if a value of None is passed.
        numbers: 
            A list of agent ID's that should be plotted in the image. By default this value is set to None.
        direction_vec: 
            Flag to determine if a vector showing the vehicles direction should be plotted in the animation. By default this flag is set to True.
        velocity_vec:
            Flag to determine if the a vector showing the vehicles velocity should be plotted in the animation. By default this flag is set to False.
        plot_frame_number: 
            Flag to determine if the frame numbers should be plotted in the animation. By default this flag is set to False.
        mark_waypoint: 
            Flag to determine if the waypoint should be marked in the animation. By default this flag is set to False.
        agent_face_colors:
            An optional parameter containing a list of RGB tuples indicating the desired color of the agent with the corresponding index ID. A value 
            of None in this list will use the default color. If the number of agents change throughout the simulation, the color of each agent must 
            be specified per time step.
        agent_edge_colors:
            An optional parameter containing a list of RGB tuples indicating the desired color of a border around the agent with the corresponding index 
            ID. A value of None in this list will use the default color. If the number of agents change throughout the simulation, the color of each agent 
            must be specified per time step.
        """

        self._validate_agent_style_data(
            agent_face_colors=agent_face_colors,
            agent_edge_colors=agent_edge_colors
        )

        self._initialize_plot(
            ax=ax, 
            numbers=numbers, 
            direction_vec=direction_vec,
            velocity_vec=velocity_vec, 
            plot_frame_number=plot_frame_number,
            mark_waypoints=mark_waypoints
        )
        end_idx = len(self.agent_states_history) if end_idx == -1 else end_idx
        fig = self.current_ax.figure
        fig.set_size_inches(self._resolution[0] / self._dpi, self._resolution[1] / self._dpi, True)

        def animate(i):
            self._update_frame_to(i)

        ani = FuncAnimation(
            fig, animate, np.arange(start_idx, end_idx), interval=100)
        if output_name is not None:
            ani.save(f'{output_name}', writer='pillow', dpi=self._dpi)
        return ani

    def _validate_agent_style_data_helper(self,agent_colors,agent_color_type):
        if agent_colors is None:
            agent_colors = [None]*len(self.agent_properties)
        else:
            if type(agent_colors) == ColorList:
                agent_colors = [agent_colors]*len(self.agent_properties)
            else:
                assert len(self.agent_properties) == len(agent_colors), f"Number of {agent_color_type} time steps does not match number of simulation time steps."

        for i, (props_ts, colors_ts) in enumerate(zip(self.agent_properties,agent_colors)):
            if colors_ts is not None:
                assert len(colors_ts) == len(props_ts), f"Number of {agent_color_type} does not match number of agents at time step {i}."

        return agent_colors

    def _validate_agent_style_data(self,agent_face_colors,agent_edge_colors):
        if self.agent_properties is not None: 
            self.agent_face_colors = self._validate_agent_style_data_helper(
                agent_colors=agent_face_colors,
                agent_color_type="agent face colors"
            )

            self.agent_edge_colors = self._validate_agent_style_data_helper(
                agent_colors=agent_edge_colors,
                agent_color_type="agent face colors"
            )

        else:
            raise Exception("No agent properties found, cannot validate agent face or edge colours.")

    def _validate_timestep_agents(
        self,
        agent_states: List[AgentState],
        agent_properties: List[AgentProperties]
    ): 
        assert len(agent_states) == len(agent_properties), "Number of given agent states and agent properties is unequal."

    def _transform_point_to_left_hand_coordinate_frame(self,x,orientation):
        t_x = 2*self.xy_offset[0] - x
        if orientation >= 0:
            t_orientation = -orientation + math.pi
        else:
            t_orientation = -orientation - math.pi

        return t_x, t_orientation

    def _plot_frame(
        self, 
        idx, 
        ax=None, 
        numbers=None, 
        direction_vec=True,
        velocity_vec=False, 
        plot_frame_number=False,
        mark_waypoints=False
    ):
        self._initialize_plot(
            ax=ax, 
            numbers=numbers, 
            direction_vec=direction_vec,
            velocity_vec=velocity_vec, 
            plot_frame_number=plot_frame_number,
            mark_waypoints=mark_waypoints
        )
        self._update_frame_to(idx)

    def _initialize_plot(
        self, 
        ax=None, 
        numbers=None, 
        direction_vec=True,
        velocity_vec=False, 
        plot_frame_number=False,
        mark_waypoints=False 
    ):
        if ax is None:
            plt.clf()
            ax = plt.gca()
        if self._open_drive is None:
            ax.imshow(self.map_image, extent=self.extent)
        else:
            self._draw_xodr_map(ax)
            self.extent = (self.xy_offset[0] - self.fov / 2, self.xy_offset[0] + self.fov / 2) +\
                (self.xy_offset[1] - self.fov / 2, self.xy_offset[1] + self.fov / 2)

            ax.set_xlim((self.extent[0], self.extent[1]))
            ax.set_ylim((self.extent[2], self.extent[3]))
        self.current_ax = ax

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.waypoint_markers = {}
        self.frame_label = None

        self.numbers = numbers
        self.direction_vec = direction_vec
        self.velocity_vec = velocity_vec
        self.plot_frame_number = plot_frame_number
        self.mark_waypoints = mark_waypoints

        self._update_frame_to(0)

    def _get_color(
        self,
        agent_idx,
        color_list
    ):
        c = None
        if color_list and color_list[agent_idx]:
            is_good_color_format = isinstance(color_list[agent_idx],tuple)
            for pc in color_list[agent_idx]:
                is_good_color_format *= isinstance(pc,float) and (0.0 <= pc <= 1.0)
            
            if not is_good_color_format:
                raise Exception(f"Expected color format is Tuple[float,float,float] with 0 <= float <= 1 but received {color_list[agent_idx]}.")
            c = color_list[agent_idx]

        return c

    def _update_frame_to(self, frame_idx):
        for rect in self.actor_boxes.values():
            rect.set_visible(False)
        if hasattr(self, "waypoint_markers"):
            for marker in self.waypoint_markers.values():
                marker.set_visible(False)
        for lines in self.dir_lines.values():
            for line in lines:
                line.set_visible(False)
        for lines in self.v_lines.values():
            for line in lines:
                line.set_visible(False)
        for label in self.box_labels.values():
            label.set_visible(False)

        for i in range(len(self.agent_properties[frame_idx])):
            self._update_agent(
                agent_idx=i,
                frame_idx=frame_idx
            )

        if self.traffic_lights_history[frame_idx] is not None:
            for light_id, light_state in self.traffic_lights_history[frame_idx].items():
                self._plot_traffic_light(light_id, light_state)
        # if self.waypoint_marker_history and frame_idx < len(self.waypoint_history):
        #     self._plot_waypoints(frame_idx)

        if self.plot_frame_number:
            if self.frame_label is None:
                self.frame_label = self.current_ax.text(
                    self.extent[0], 
                    self.extent[2], 
                    str(frame_idx), 
                    c="r", 
                    fontsize=18
                )
            else:
                self.frame_label.set_text(str(frame_idx))

        if self._open_drive is None:
            self.current_ax.set_xlim(*self.extent[0:2])
            self.current_ax.set_ylim(*self.extent[2:4])

    def _update_agent(
        self, 
        agent_idx, 
        frame_idx
    ):
        agent = self.agent_states_history[frame_idx][agent_idx]
        agent_properties = self.agent_properties[frame_idx][agent_idx]

        l, w = agent_properties.length, agent_properties.width
        if agent_properties.agent_type == "pedestrian":
            l, w = 1.5, 1.5
        x, y = agent.center.x, agent.center.y
        v = agent.speed
        psi = agent.orientation

        if self._left_hand_coordinates:
            x, psi = self._transform_point_to_left_hand_coordinate_frame(x,psi)

        box = np.array([
            [0, 0], [l * 0.5, 0],  # direction vector
            [0, 0], [v * 0.5, 0],  # speed vector at (0.5 m / s ) / m
        ])

        box = np.matmul(rot(psi), box.T).T + np.array([[x, y]])
        if self.direction_vec:
            marker_offset = agent_properties.length/4
            x_data = x + marker_offset*math.cos(psi)
            y_data = y + marker_offset*math.sin(psi)
            marker_data = (3, 0, (-90+180*psi/math.pi))

            if agent_idx not in self.dir_lines:
                self.dir_lines[agent_idx] = self.current_ax.plot(
                    x_data,
                    y_data,
                    marker=marker_data,
                    markersize=agent_properties.width*400/self.fov, 
                    linestyle='None',
                    c=self.dir_c
                )
            else:
                self.dir_lines[agent_idx][0].set_xdata([x_data])
                self.dir_lines[agent_idx][0].set_ydata([y_data])
                self.dir_lines[agent_idx][0].set_marker(marker_data)

            self.dir_lines[agent_idx][0].set_visible(True)

        if self.velocity_vec:
            if agent_idx not in self.v_lines:
                self.v_lines[agent_idx] = self.current_ax.plot(
                    box[2:4, 0], 
                    box[2:4, 1], 
                    lw=1.5, 
                    c=self.v_c
                )[0]  # plot the speed
            else:
                self.v_lines[agent_idx].set_xdata(box[2:4, 0])
                self.v_lines[agent_idx].set_ydata(box[2:4, 1])

            self.v_lines[agent_idx][0].set_visible(True)
        
        if self.numbers is not None and agent_idx in self.numbers:
            if agent_idx not in self.box_labels:
                self.box_labels[agent_idx] = self.current_ax.text(
                    x, 
                    y, 
                    str(agent_idx), 
                    c="r", 
                    fontsize=18
                )
                self.box_labels[agent_idx].set_clip_on(True)
            else:
                self.box_labels[agent_idx].set_x(x)
                self.box_labels[agent_idx].set_y(y)

            self.box_labels[agent_idx].set_visible(True)

        lw = 1
        fc = self._get_color(agent_idx,self.agent_face_colors[frame_idx])
        if fc is None:
            fc = self.agent_c
        ec = self._get_color(agent_idx,self.agent_edge_colors[frame_idx])
        if ec is None:
            lw = 0
            ec = fc

        rect = Rectangle(
            (x - l / 2, y - w / 2), 
            l, 
            w, 
            angle=psi * 180 / np.pi, 
            rotation_point='center', 
            fc=fc, 
            ec=ec, 
            lw=lw
        )
        if agent_properties.waypoint is not None:
            wp_x = agent_properties.waypoint.x
            wp_y = agent_properties.waypoint.y

            if self._left_hand_coordinates:
                wp_x, _ = self._transform_point_to_left_hand_coordinate_frame(wp_x, 0)

            # Initialize storage if missing
            if not hasattr(self, "waypoint_markers"):
                self.waypoint_markers = {}

            # Scale marker size relative to FOV (smaller for large maps)
            marker_size = max(2, 200 / self.fov)  # e.g. ~2–4 px typical

            # Create or update waypoint marker
            if agent_idx not in self.waypoint_markers:
                self.waypoint_markers[agent_idx] = self.current_ax.plot(
                    wp_x,
                    wp_y,
                    marker="o",
                    markersize=marker_size,
                    color=(0.36, 0.25, 0.2),  # brown tone 
                    linestyle="None",
                    zorder=6,
                )[0]
            else:
                m = self.waypoint_markers[agent_idx]
                m.set_xdata([wp_x])
                m.set_ydata([wp_y])
                m.set_markersize(marker_size)
                m.set_visible(True)

        if agent_idx in self.actor_boxes:
            self.actor_boxes[agent_idx].remove()
        self.actor_boxes[agent_idx] = rect
        self.actor_boxes[agent_idx].set_clip_on(True)
        self.current_ax.add_patch(self.actor_boxes[agent_idx])
        self.actor_boxes[agent_idx].set_visible(True)

        if self.mark_waypoints:
            wp = getattr(agent_properties, "waypoint", None)
            if wp is not None:
                # If not drawn before, create marker
                if agent_idx not in self.waypoint_markers:
                    self.waypoint_markers[agent_idx], = self.current_ax.plot(
                        wp.x, wp.y,
                        marker='o',
                        color='saddlebrown',
                        markersize=1.5,
                        zorder=6
                    )
                else:
                    # Update marker position
                    self.waypoint_markers[agent_idx].set_data([wp.x], [wp.y])
                self.waypoint_markers[agent_idx].set_visible(True)
            else:
                # Hide marker if this agent has no waypoint this frame
                if agent_idx in self.waypoint_markers:
                    self.waypoint_markers[agent_idx].set_visible(False)

    def _plot_waypoints(self, frame_idx):
            """Draws the waypoints for a single frame index."""
            waypoints = self.waypoint_history[frame_idx]
            
            # Clear/remove any previous markers that might be visible
            for marker in self.waypoint_markers.values():
                marker.remove()
            self.waypoint_markers = {} # Reset the dictionary

            # Plot the new waypoints
            ax = self.current_ax
            for i, wp in enumerate(waypoints):
                # Assuming waypoint object has an 'x' and 'y' location attribute
                x = wp.location.x if hasattr(wp, 'location') else wp.x # Adapt based on your waypoint format
                y = wp.location.y if hasattr(wp, 'location') else wp.y
                
                # Apply coordinate transformation if needed
                if self._left_hand_coordinates:
                    x = 2 * self.xy_offset[0] - x # Correct transformation for X

                # Reuse existing line objects or create new ones
                if i in self.waypoint_markers:
                    marker = self.waypoint_markers[i]
                    marker.set_data(x, y)
                    marker.set_visible(True)
                else:
                    marker, = ax.plot(
                        x, y, 
                        marker='.', # Small dot marker
                        markersize=5, 
                        color='lime', 
                        linestyle='', 
                        alpha=0.6,
                        zorder=10 # Ensure waypoints are drawn over the map/road
                    )
                    self.waypoint_markers[i] = marker
    def _plot_traffic_light(
        self, 
        light_id, 
        light_state
    ):
        light = self.traffic_lights[light_id]
        x, y = light.center.x, light.center.y
        psi = light.orientation
        l, w = max(light.length,1.0), max(light.width,1.0)

        if self._left_hand_coordinates:
            x, psi = self._transform_point_to_left_hand_coordinate_frame(x,psi)

        rect = Rectangle(
            (x - l / 2, y - w / 2),
            l,
            w,
            angle=psi * 180 / np.pi,
            rotation_point="center",
            fc=self.traffic_light_colors[light_state],
            lw=0,
        )
        if light_id in self.traffic_light_boxes:
            self.traffic_light_boxes[light_id].remove()
        self.current_ax.add_patch(rect)
        self.traffic_light_boxes[light_id] = rect

    def _draw_xodr_map(self, ax, extras=False):
        """
        This function plots the parsed xodr map
        the `odrplot` of `esmini` is used for plotting and parsing xodr
        https: // esmini.github.io/  # _tools_overview
        """
        with open(self._open_drive) as f:
            reader = csv.reader(f, skipinitialspace=True)
            positions = list(reader)

        ref_x = []
        ref_y = []
        ref_z = []
        ref_h = []

        lane_x = []
        lane_y = []
        lane_z = []
        lane_h = []

        border_x = []
        border_y = []
        border_z = []
        border_h = []

        road_id = []
        road_id_x = []
        road_id_y = []

        road_start_dots_x = []
        road_start_dots_y = []

        road_end_dots_x = []
        road_end_dots_y = []

        lane_section_dots_x = []
        lane_section_dots_y = []

        arrow_dx = []
        arrow_dy = []

        current_road_id = None
        current_lane_id = None
        current_lane_section = None
        new_lane_section = False

        for i in range(len(positions) + 1):

            if i < len(positions):
                pos = positions[i]

            # plot road id before going to next road
            if i == len(positions) or (
                pos[0] == "lane" and i > 0 and current_lane_id == "0"
            ):

                if current_lane_section == "0":
                    road_id.append(int(current_road_id))
                    index = int(len(ref_x[-1]) / 3.0)
                    h = ref_h[-1][index]
                    road_id_x.append(
                        ref_x[-1][index]
                        + (text_x_offset * math.cos(h) - text_y_offset * math.sin(h))
                    )
                    road_id_y.append(
                        ref_y[-1][index]
                        + (text_x_offset * math.sin(h) + text_y_offset * math.cos(h))
                    )
                    road_start_dots_x.append(ref_x[-1][0])
                    road_start_dots_y.append(ref_y[-1][0])
                    if len(ref_x) > 0:
                        arrow_dx.append(ref_x[-1][1] - ref_x[-1][0])
                        arrow_dy.append(ref_y[-1][1] - ref_y[-1][0])
                    else:
                        arrow_dx.append(0)
                        arrow_dy.append(0)

                lane_section_dots_x.append(ref_x[-1][-1])
                lane_section_dots_y.append(ref_y[-1][-1])

            if i == len(positions):
                break

            if pos[0] == "lane":
                current_road_id = pos[1]
                current_lane_section = pos[2]
                current_lane_id = pos[3]
                if pos[3] == "0":
                    ltype = "ref"
                    ref_x.append([])
                    ref_y.append([])
                    ref_z.append([])
                    ref_h.append([])

                elif pos[4] == "no-driving":
                    ltype = "border"
                    border_x.append([])
                    border_y.append([])
                    border_z.append([])
                    border_h.append([])
                else:
                    ltype = "lane"
                    lane_x.append([])
                    lane_y.append([])
                    lane_z.append([])
                    lane_h.append([])
            else:
                if ltype == "ref":
                    ref_x[-1].append(float(pos[0]))
                    ref_y[-1].append(float(pos[1]))
                    ref_z[-1].append(float(pos[2]))
                    ref_h[-1].append(float(pos[3]))

                elif ltype == "border":
                    border_x[-1].append(float(pos[0]))
                    border_y[-1].append(float(pos[1]))
                    border_z[-1].append(float(pos[2]))
                    border_h[-1].append(float(pos[3]))
                else:
                    lane_x[-1].append(float(pos[0]))
                    lane_y[-1].append(float(pos[1]))
                    lane_z[-1].append(float(pos[2]))
                    lane_h[-1].append(float(pos[3]))

        # plot driving lanes in blue
        for i in range(len(lane_x)):
            ax.plot(lane_x[i], lane_y[i], linewidth=1.0, color="#222222")

        # plot road ref line segments
        for i in range(len(ref_x)):
            ax.plot(ref_x[i], ref_y[i], linewidth=2.0, color="#BB5555")

        # plot border lanes in gray
        for i in range(len(border_x)):
            ax.plot(border_x[i], border_y[i], linewidth=1.0, color="#AAAAAA")

        if extras:
            # plot red dots indicating lane dections
            for i in range(len(lane_section_dots_x)):
                ax.plot(
                    lane_section_dots_x[i],
                    lane_section_dots_y[i],
                    "o",
                    ms=4.0,
                    color="#BB5555",
                )

            for i in range(len(road_start_dots_x)):
                # plot a yellow dot at start of each road
                ax.plot(
                    road_start_dots_x[i],
                    road_start_dots_y[i],
                    "o",
                    ms=5.0,
                    color="#BBBB33",
                )
                # and an arrow indicating road direction
                ax.arrow(
                    road_start_dots_x[i],
                    road_start_dots_y[i],
                    arrow_dx[i],
                    arrow_dy[i],
                    width=0.1,
                    head_width=1.0,
                    color="#BB5555",
                )
            # plot road id numbers
            for i in range(len(road_id)):
                ax.text(
                    road_id_x[i],
                    road_id_y[i],
                    road_id[i],
                    size=text_size,
                    ha="center",
                    va="center",
                    color="#3333BB",
                )

        return None

