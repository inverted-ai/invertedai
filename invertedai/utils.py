import requests
import asyncio
import json
import re
from typing import Dict, Optional, List, Tuple, Any
from requests.auth import AuthBase
from requests.adapters import HTTPAdapter, Retry
import invertedai as iai
import invertedai.api
import invertedai.api.config
from invertedai import error
from invertedai.common import Point
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
import numpy as np
import csv
import math
from tqdm.contrib import tmap
from itertools import product
from invertedai.common import AgentState, AgentAttributes, TrafficLightState, StaticMapActor, Image
from invertedai.light import LocationResponse
from matplotlib import transforms
from copy import deepcopy
from invertedai.future import to_thread
from pydantic import validate_arguments, BaseModel, ConfigDict

H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7
TIMEOUT_SECS = 600
MAX_RETRIES = 10
SLACK = 2
INITIALIZE_FOV = 120
AGENT_FOV = 35


class Session:
    def __init__(self, api_token: str = ""):
        self.session = requests.Session()
        self.session.auth = APITokenAuth(api_token)
        retries = Retry(total=5,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504],
                        raise_on_status=False)
        self.session.mount(
            "https://",
            requests.adapters.HTTPAdapter(max_retries=retries),
        )
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        )
        self.base_url = self._get_base_url()

    def add_apikey(self, api_token: str = ""):
        if not iai.dev and not api_token:
            raise error.InvalidAPIKeyError("Empty API key received.")
        self.session.auth = APITokenAuth(api_token)
        response = self.session.request(method="get", url=self.base_url)
        if response.status_code != 200:
            url_acd = "https://api.inverted.ai/v0/academic/m1"
            response_acd = self.session.request(method="get", url=url_acd)
            if response_acd.status_code == 200:
                self.base_url = url_acd
                response = response_acd
            elif response_acd.status_code != 403:
                response = response_acd
        if response.status_code == 403:
            raise error.AuthenticationError(
                "Access denied. Please check the provided API key."
            )
        elif response.status_code != 200:
            raise error.APIError("The server is aware that it has erred or is incapable of performing the requested"
                                 " method.")

    def use_mock_api(self, use_mock: bool = True) -> None:
        invertedai.api.config.mock_api = use_mock
        if use_mock:
            iai.logger.warning(
                'Using mock Inverted AI API - predictions will be trivial'
            )

    async def async_request(self, *args, **kwargs):
        return await to_thread(self.request, *args, **kwargs)

    def request(
        self, model: str, params: Optional[dict] = None, data: Optional[dict] = None
    ):
        method, relative_path = iai.model_resources[model]
        response = self._request(
            method=method,
            relative_path=relative_path,
            params=params,
            json_body=data,
        )

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
            result = self.session.request(
                method=method,
                params=params,
                url=self.base_url + relative_path,
                headers=headers,
                data=data,
                json=json_body,
            )
            result.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise error.APIConnectionError("Error communicating with IAI", should_retry=True)
        except requests.exceptions.Timeout as e:
            raise error.APIConnectionError("Error communicating with IAI")
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 403:
                raise error.AuthenticationError(
                    "Access denied. Please check the provided API key."
                )
            elif e.response.status_code in [400, 422]:
                raise error.InvalidRequestError(e.response.text, param="")
            elif e.response.status_code == 404:
                raise error.ResourceNotFoundError(e.response.text)
            elif e.response.status_code == 413:
                raise error.LargeRequestTimeout(e.response.text)
            elif e.response.status_code == 429:
                raise error.RateLimitError("Throttled")
            elif e.response.status_code == 503:
                raise error.ServiceUnavailableError("Service Unavailable")
            elif 400 <= e.response.status_code < 500:
                raise error.APIError("Invalid request. Please check the sent data again.")
            else:
                raise error.APIError("The server is aware that it has erred or is incapable of performing the requested"
                                     " method.")
        iai.logger.info(
            iai.logger.logfmt(
                "IAI API response",
                path=self.base_url,
                response_code=result.status_code,
            )
        )
        try:
            data = json.loads(result.content)
        except json.decoder.JSONDecodeError:
            raise error.APIError(
                f"HTTP code {result.status_code} from API ({result.content})", result.content, result.status_code,
                headers=result.headers
            )
        return data

    def _get_base_url(self) -> str:
        """
        This function returns the endpoint for API calls, which includes the
        version and other endpoint specifications.
        The method path should be appended to the base_url
        """
        if not iai.dev:
            base_url = "https://api.inverted.ai/v0/aws/m1"
        else:
            base_url = iai.dev_url
        # TODO: Add endpoint option and versioning to base_url
        return base_url

    def _handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
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

    def _interpret_response_line(self, result):
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


def get_centers(map_center, height, width, stride):
    def check_valid_center(center):
        return ((map_center[0] - width) < center[0] < (map_center[0] + width) and
                (map_center[1] - height) < center[1] < (map_center[1] + height))

    def get_neighbors(center):
        return [(center[0] + (i * stride), center[1] + (j * stride)) for i, j in list(product(*[(-1, 1), ] * 2))]

    queue, centers = [map_center], []

    while queue:
        center = queue.pop(0)
        neighbors = filter(check_valid_center, get_neighbors(center))
        queue.extend([neighbor for neighbor in neighbors if neighbor not in queue and neighbor not in centers])
        if center not in centers and check_valid_center(center):
            centers.append(center)
    return centers


async def async_area_re_initialization(location, agent_attributes, states_history, traffic_lights_states=None,
                                       random_seed=None, map_center=(0, 0), width=100, height=100,
                                       initialize_fov=INITIALIZE_FOV, get_birdview=False, birdview_path=None):
    def inside_fov(center: Point, initialize_fov: float, point: Point) -> bool:
        return ((center.x - (initialize_fov / 2) < point.x < center.x + (initialize_fov / 2)) and
                (center.y - (initialize_fov / 2) < point.y < center.y + (initialize_fov / 2)))

    async def reinit(reinitialize_agent_state, reinitialize_agent_attrs, area_center):
        try:
            # Initialize simulation with an API cal
            response = await iai.async_initialize(
                location=location,
                states_history=reinitialize_agent_state,
                agent_attributes=reinitialize_agent_attrs,
                agent_count=len(reinitialize_agent_attrs),
                get_infractions=False,
                traffic_light_state_history=traffic_lights_states,
                location_of_interest=(area_center.x, area_center.y),
                random_seed=random_seed,
                get_birdview=get_birdview,
            )
        except BaseException:
            return [], [], []
        SLACK = 0
        valid_agents = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov - SLACK, point=x[0].center),
            zip(response.agent_states, response.agent_attributes, response.recurrent_states)))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]
        if get_birdview:
            file_path = f"{birdview_path}-{(area_center.x, area_center.y)}.jpg"
            response.birdview.decode_and_save(file_path)

        return valid_agent_state, valid_agent_attrs, valid_agent_rs

    stride = initialize_fov / 2

    remaining_agents_states = states_history
    remaining_agents_attrs = agent_attributes
    new_agent_state = []
    new_attributes = []
    new_recurrent_states = []
    # # first = True
    centers = get_centers(map_center, height, width, stride)
    initialize_payload = []
    for area_center in tmap(Point.fromlist, centers, total=len(centers),
                            desc=f"Renewing Recurrent States {location.split(':')[1]}"):

        reinitialize_agent = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0][-1].center), zip(remaining_agents_states, remaining_agents_attrs)))
        remaining_agents = list(filter(lambda x: not inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0][-1].center), zip(remaining_agents_states, remaining_agents_attrs)))

        reinitialize_agent_state = [x[0] for x in reinitialize_agent]
        #: Reorder form list of agents to list of time steps
        reinitialize_agent_state = [list(st) for st in zip(*reinitialize_agent_state)]
        reinitialize_agent_attrs = [x[1] for x in reinitialize_agent]
        remaining_agents_states = [x[0] for x in remaining_agents]
        remaining_agents_attrs = [x[1] for x in remaining_agents]

        initialize_payload.append({"center": area_center, "state": reinitialize_agent_state,
                                  "attr": reinitialize_agent_attrs})

    results = await asyncio.gather(*[reinit(agnts["state"], agnts["attr"], agnts["center"]) for agnts in initialize_payload])

    for result in results:
        new_agent_state += result[0]
        new_attributes += result[1]
        new_recurrent_states += result[2]

    return invertedai.api.InitializeResponse(
        recurrent_states=new_recurrent_states,
        agent_states=new_agent_state,
        agent_attributes=new_attributes)


def area_re_initialization(location, agent_attributes, states_history, traffic_lights_states=None, random_seed=None,
                           map_center=(0, 0), width=100, height=100, initialize_fov=INITIALIZE_FOV, get_birdview=False,
                           birdview_path=None):
    def inside_fov(center: Point, initialize_fov: float, point: Point) -> bool:
        return ((center.x - (initialize_fov / 2) < point.x < center.x + (initialize_fov / 2)) and
                (center.y - (initialize_fov / 2) < point.y < center.y + (initialize_fov / 2)))
    stride = initialize_fov / 2

    remaining_agents_states = states_history
    remaining_agents_attrs = agent_attributes
    new_agent_state = []
    new_attributes = []
    new_recurrent_states = []
    # first = True

    centers = get_centers(map_center, height, width, stride)
    for area_center in tmap(Point.fromlist, centers, total=len(centers),
                            desc=f"Initializing {location.split(':')[1]}"):

        reinitialize_agent = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0][-1].center), zip(remaining_agents_states, remaining_agents_attrs)))
        remaining_agents = list(filter(lambda x: not inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0][-1].center), zip(remaining_agents_states, remaining_agents_attrs)))

        reinitialize_agent_state = [x[0] for x in reinitialize_agent]
        #: Reorder form list of agents to list of time steps
        reinitialize_agent_state = [list(st) for st in zip(*reinitialize_agent_state)]
        reinitialize_agent_attrs = [x[1] for x in reinitialize_agent]

        remaining_agents_states = [x[0] for x in remaining_agents]
        remaining_agents_attrs = [x[1] for x in remaining_agents]

        if not reinitialize_agent_state:
            continue

        for _ in range(1):
            try:
                # Initialize simulation with an API cal
                response = iai.initialize(
                    location=location,
                    states_history=reinitialize_agent_state,
                    agent_attributes=reinitialize_agent_attrs,
                    agent_count=len(reinitialize_agent_attrs),
                    get_infractions=False,
                    traffic_light_state_history=traffic_lights_states,
                    location_of_interest=(area_center.x, area_center.y),
                    random_seed=random_seed,
                    get_birdview=get_birdview,
                )
                break
            except BaseException:
                pass
        else:
            continue
        # Filter out agents that are not inside the ROI to avoid collision with other agents not passed as conditional
        # SLACK is for removing the agents that are very close to the boundary and
        # they may collide agents not filtered as conditional
        SLACK = 0
        valid_agents = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov - SLACK, point=x[0].center),
            zip(response.agent_states, response.agent_attributes, response.recurrent_states)))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]

        new_agent_state += valid_agent_state
        new_attributes += valid_agent_attrs
        new_recurrent_states += valid_agent_rs
        if get_birdview:
            file_path = f"{birdview_path}-{(area_center.x, area_center.y)}.jpg"
            response.birdview.decode_and_save(file_path)

    return invertedai.api.InitializeResponse(
        recurrent_states=new_recurrent_states,
        agent_states=new_agent_state,
        agent_attributes=new_attributes)


def area_initialization(location, agent_density, traffic_lights_states=None, random_seed=None, map_center=(0, 0),
                        width=100, height=100, stride=100, initialize_fov=INITIALIZE_FOV, get_birdview=False,
                        birdview_path=None):
    def inside_fov(center: Point, initialize_fov: float, point: Point) -> bool:
        return ((center.x - (initialize_fov / 2) < point.x < center.x + (initialize_fov / 2)) and
                (center.y - (initialize_fov / 2) < point.y < center.y + (initialize_fov / 2)))

    agent_states = []
    agent_attributes = []
    agent_rs = []
    first = True
    centers = get_centers(map_center, height, width, stride)
    for area_center in tmap(Point.fromlist, centers, total=len(centers),
                            desc=f"Initializing {location.split(':')[1]}"):

        conditional_agent = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0].center), zip(agent_states, agent_attributes,
                                                                                       agent_rs)))
        remaining_agents = list(filter(lambda x: not inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0].center), zip(agent_states, agent_attributes,
                                                                                       agent_rs)))

        con_agent_state = [x[0] for x in conditional_agent]
        con_agent_attrs = [x[1] for x in conditional_agent]
        con_agent_rs = [x[2] for x in conditional_agent]
        remaining_agents_states = [x[0] for x in remaining_agents]
        remaining_agents_attrs = [x[1] for x in remaining_agents]
        remaining_agents_rs = [x[2] for x in remaining_agents]

        if len(con_agent_state) > agent_density:
            continue

        for _ in range(1):
            try:
                # Initialize simulation with an API cal
                response = iai.initialize(
                    location=location,
                    states_history=[con_agent_state] if len(con_agent_state) > 0 else None,
                    agent_attributes=con_agent_attrs if len(con_agent_attrs) > 0 else None,
                    agent_count=agent_density,
                    get_infractions=False,
                    traffic_light_state_history=traffic_lights_states,
                    location_of_interest=(area_center.x, area_center.y),
                    random_seed=random_seed,
                    get_birdview=get_birdview,
                )
                break
            except BaseException as e:
                print(e)
        else:
            continue
        # Filter out agents that are not inside the ROI to avoid collision with other agents not passed as conditional
        # SLACK is for removing the agents that are very close to the boundary and
        # they may collide agents not filtered as conditional
        valid_agents = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov - SLACK, point=x[0].center),
            zip(response.agent_states, response.agent_attributes, response.recurrent_states)))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]

        agent_states = remaining_agents_states + valid_agent_state
        agent_attributes = remaining_agents_attrs + valid_agent_attrs
        agent_rs = remaining_agents_rs + valid_agent_rs

        if get_birdview:
            file_path = f"{birdview_path}-{(area_center.x, area_center.y)}.jpg"
            response.birdview.decode_and_save(file_path)

    return invertedai.api.InitializeResponse(
        recurrent_states=agent_rs,
        agent_states=agent_states,
        agent_attributes=agent_attributes)


class APITokenAuth(AuthBase):
    def __init__(self, api_token):
        self.api_token = api_token

    def __call__(self, r):
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

        def update(self, change):
            self.im.set_data(self.buffer[self.int_slider.value])
            self.fig.canvas.draw()

        def add_frame(self, frame):
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

    @ staticmethod
    def logfmt(message, **params):
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
    return np.array([[np.cos(rot), -np.sin(rot)],
                     [np.sin(rot), np.cos(rot)]])



class ScenePlotter():
    """
    A class providing features and handling the data regarding visualization of a scene involving IAI data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validate_arguments
    def __init__(
        self,
        location_response: Optional[LocationResponse] = None,
        open_drive: bool = False, 
        resolution: Tuple[int,int] = (640, 480), 
        dpi: float = 100,
        **kwargs
    ):
        """
        Arguments
        ----------
        location_response: Optional[LocationResponse] = None
            A LocationResponse object taken from calling iai.light() containing relevant data regarding the location of the scene including the map image.
        open_drive: bool = False
            A flag dictating whether the map is in the ASAM OpenDRIVE format (i.e. xodr maps)
        resolution: Tuple[int,int] = (640, 480)
            The desired resolution of the map image expressed as a Tuple with two integers for the width and height respectively.
        dpi: float = 100
            Dots per inch to define the level of detail in the image.

        Keyword Arguments
        -----------------
        map_image: [np.ndarray]
            Base image onto which the scene is visualized. Only use this argument if not using the respective information from the relevant LocationReponse object.
        fov: float
            The field of view in meters corresponding to the map_image attribute. Only use this argument if not using the respective information from the relevant LocationReponse object.
        xy_offset: Optional[Tuple[int,int]] = None
            The left-hand offset for the center of the map image. Only use this argument if not using the respective information from the relevant LocationReponse object.
        static_actors: Optional[List[StaticMapActor]] = None
            A list of static actor agents (e.g. traffic lights) represented as StaticMapActor objects, in the scene. Only use this argument if not using the respective information from the relevant LocationReponse object.


        """

        self.conditional_agents = None
        self.agent_attributes = None
        self.traffic_lights_history = None
        self.agent_states_history = None
        
        self.open_drive = open_drive
        
        if not self.open_drive
            self.map_image = location_response.birdview_image.decode()
            self.fov = location_response.map_fov
            self.xy_offset = (location_response.map_center.x, location_response.map_center.y)
            static_actor = location_response.static_actors
        else:
            self._validate_kwargs("map_image")
            self._validate_kwargs("fov")
            self._validate_kwargs("xy_offset")
            self._validate_kwargs("static_actor")

        self.traffic_lights = {static_actor.actor_id: static_actor for static_actor in static_actors if static_actor.agent_type == 'traffic-light'}


        self.dpi = dpi
        self.resolution = resolution
        self.fov = fov
        self.map_image = map_image
        if not open_drive:
            self.extent = (- self.fov / 2 + xy_offset[0], self.fov / 2 + xy_offset[0]) + \
                (- self.fov / 2 + xy_offset[1], self.fov / 2 + xy_offset[1])
        else:
            self.map_center = xy_offset

        

        self.traffic_light_colors = {
            'red': (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'yellow': (1.0, 0.8, 0.0)
        }

        self.agent_c = (0.2, 0.2, 0.7)
        self.cond_c = (0.75, 0.35, 0.35)
        self.dir_c = (0.9, 0.9, 0.9)
        self.v_c = (0.2, 0.75, 0.2)

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.frame_label = None
        self.current_ax = None

        self.numbers = False

        self.agent_face_colors = None 
        self.agent_edge_colors = None 

        self.reset_recording()

    @validate_arguments
    def initialize_recording(
        self,
        agent_states: List[AgentState], 
        agent_attributes: List[AgentAttributes], 
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None, 
        conditional_agents: Optional[List[int]] = None
    ):
        """
        Record the initial state of the scene to be visualized. This function also acts as an implicit reset of the recording and removes previous agent state, agent attribute, conditional agent, traffic light, and agent style data.

        Arguments
        ----------
        agent_states: List[AgentState]
            A list of AgentState objects corresponding to the initial time step to be visualized.
        agent_attributes: List[AgentState]
            Static attributes of the agent, which don’t change over the course of a simulation. We assume every agent is a rectangle obeying a kinematic bicycle model.
        traffic_light_states: Optional[Dict[int, TrafficLightState]]
            Optional parameter containing the state of the traffic lights corresponding to the initial time step to be visualized. This parameter should only be used if the corresponding map contains traffic light static actors.
        conditional_agents: List[int]
            Optional parameter containing a list of agent IDs corresponding to conditional agents to be visualized to distinguish themselves.
        """

        self.agent_states_history = [agent_states]
        self.traffic_lights_history = [traffic_light_states]
        self.agent_attributes = agent_attributes
        if conditional_agents is not None:
            self.conditional_agents = conditional_agents
        else:
            self.conditional_agents = []

        self.agent_face_colors = None
        self.agent_edge_colors = None

    def reset_recording(self):
        """
        Explicitly reset the recording and remove the previous agent state, agent attribute, conditional agent, traffic light, and agent style data.
        """
        self.agent_states_history = []
        self.traffic_lights_history = []
        self.agent_attributes = None
        self.conditional_agents = []
        self.agent_face_colors = None 
        self.agent_edge_colors = None 

    @validate_arguments
    def record_step(
        self,
        agent_states: List[AgentState], 
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None
    ):
        """
        Record a single timestep of scene data to be used in a visualization

        Arguments
        ----------
        agent_states: List[AgentState]
            A list of AgentState objects corresponding to the initial time step to be visualized.
        traffic_light_states: Optional[Dict[int, TrafficLightState]]
            Optional parameter containing the state of the traffic lights corresponding to the initial time step to be visualized. This parameter should only be used if the corresponding map contains traffic light static actors.
        """

        self.agent_states_history.append(agent_states)
        self.traffic_lights_history.append(traffic_light_states)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def plot_scene(
        self,
        agent_states: List[AgentState], 
        agent_attributes: List[AgentAttributes], 
        traffic_light_states: Optional[Dict[int, TrafficLightState]] = None, 
        conditional_agents: Optional[List[int]] = None,
        ax: Optional[Axes] = None,
        numbers: bool = False, 
        direction_vec: bool = True, 
        velocity_vec: bool = False,
        agent_face_colors: Optional[List[Optional[Tuple[float,float,float]]]] = None,
        agent_edge_colors: Optional[List[Optional[Tuple[float,float,float]]]] = None
    ):
        """
        A standalone scene plotting function that initializes and resets a recording. It is assumed this function will be used not while recording steps for a full animation.

        Parameters
        ----------
        agent_states: List[AgentState]
            A list of agents to be visualized in the image.
        agent_attributes: List[AgentState]
            Static attributes of the agent, which don’t change over the course of a simulation. We assume every agent is a rectangle obeying a kinematic bicycle model.
        traffic_light_states: Optional[Dict[int, TrafficLightState]]
            Optional parameter containing the state of the traffic lights to be visualized in the image. This parameter should only be used if the corresponding map contains traffic light static actors.
        conditional_agents: List[int]
            Optional parameter containing a list of agent IDs of conditional agents to be visualized in the image to distinguish themselves.
        ax: Optional[Axes] = None
            A matplotlib Axes object used to plot the image. By default, an Axes object is created if a value of None is passed.
        numbers: bool = False
            Flag to determine if the ID's of all agents should be plotted in the image. By default this flag is set to False.
        direction_vec: bool = True
            Flag to determine if a vector showing the vehicles direction should be plotted in the image. By default this flag is set to True.
        velocity_vec: bool = False
            Flag to determine if the a vector showing the vehicles velocity should be plotted in the animation. By default this flag is set to False.
        agent_face_colors: Optional[List[Tuple[float,float,float]]] = None
            An optional parameter containing a list of either RGB tuples indicating the desired color of the agent with the corresponding index ID. A value of None in this list will use the default color. This value gets overwritten by the conditional agent color.
        agent_edge_colors: Optional[List[Tuple[float,float,float]]] = None
            An optional parameter containing a list of either RGB tuples indicating the desired color of a border around the agent with the corresponding index ID. A value of None in this list will use the default color. This value gets overwritten by the conditional agent color.

        """
        self.initialize_recording(agent_states, agent_attributes,
                                  traffic_light_states=traffic_light_states,
                                  conditional_agents=conditional_agents)

        self._validate_agent_style_data(agent_face_colors,agent_edge_colors)

        self._plot_frame(idx=0, ax=ax, numbers=numbers, direction_vec=direction_vec,
                        velocity_vec=velocity_vec, plot_frame_number=False)

        self.reset_recording()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def animate_scene(
        self,
        output_name: Optional[str] = None,
        start_idx: int = 0, 
        end_idx: int = -1,
        ax: Optional[Axes] = None,
        numbers: bool = False, 
        direction_vec: bool = True, 
        velocity_vec: bool = False,
        plot_frame_number: bool = False, 
        agent_face_colors: Optional[List[Optional[Tuple[float,float,float]]]] = None,
        agent_edge_colors: Optional[List[Optional[Tuple[float,float,float]]]] = None
    ) -> FuncAnimation:
        """
        Produce an animation of the sequentially recorded steps. A matplotlib animation object can be returned and/or a gif saved of the scene.

        Parameters
        ----------
        output_name: Optional[str] = None
            File name of the gif to which the animation will be saved.
        start_idx: int = 0
            The index of the time step from which the animation will begin. By default it is assumed all recorded steps are desired to be animated.
        end_idx: int = 0
            The index of the time step from which the animation will end. By default it is assumed all recorded steps are desired to be animated.
        ax: Optional[Axes] = None
            A matplotlib Axes object used to plot the animation. By default, an Axes object is created if a value of None is passed.
        numbers: bool = False
            Flag to determine if the ID's of all agents should be plotted in the animation. By default this flag is set to False.
        direction_vec: bool = True
            Flag to determine if a vector showing the vehicles direction should be plotted in the animation. By default this flag is set to True.
        velocity_vec: bool = False
            Flag to determine if the a vector showing the vehicles velocity should be plotted in the animation. By default this flag is set to False.
        plot_frame_number: bool = False
            Flag to determine if the frame numbers should be plotted in the animation. By default this flag is set to False.
        agent_face_colors: Optional[List[Tuple[float,float,float]]] = None
            An optional parameter containing a list of either RGB tuples indicating the desired color of the agent with the corresponding index ID. A value of None in this list will use the default color. This value gets overwritten by the conditional agent color.
        agent_edge_colors: Optional[List[Tuple[float,float,float]]] = None
            An optional parameter containing a list of either RGB tuples indicating the desired color of a border around the agent with the corresponding index ID. A value of None in this list will use the default color. This value gets overwritten by the conditional agent color.
        """

        self._validate_agent_style_data(agent_face_colors,agent_edge_colors)

        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        end_idx = len(self.agent_states_history) if end_idx == -1 else end_idx
        fig = self.current_ax.figure
        fig.set_size_inches(self.resolution[0] / self.dpi, self.resolution[1] / self.dpi, True)

        def animate(i):
            self._update_frame_to(i)

        ani = FuncAnimation(
            fig, animate, np.arange(start_idx, end_idx), interval=100)
        if output_name is not None:
            ani.save(f'{output_name}', writer='pillow', dpi=self.dpi)
        return ani

    def _validate_kwargs(self,arg_name):
        if arg_name in kwargs: 
            setattr(self,kwargs["map_image"])
        else: 
            raise Exception("Expected keyword argument 'map_image' but none was given.")


    def _plot_frame(self, idx, ax=None, numbers=False, direction_vec=False,
                   velocity_vec=False, plot_frame_number=False):
        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        self._update_frame_to(idx)

    def _validate_agent_style_data(self,agent_face_colors,agent_edge_colors):
        if self.agent_attributes is not None: 
            if agent_face_colors is not None:
                if len(agent_face_colors) != len(self.agent_attributes):
                    raise Exception("Number of agent face colors does not match number of agents.")
            if agent_edge_colors is not None:
                if len(agent_edge_colors) != len(self.agent_attributes):
                    raise Exception("Number of agent edge colors does not match number of agents.")

        self.agent_face_colors = agent_face_colors
        self.agent_edge_colors = agent_edge_colors

    def _initialize_plot(self, ax=None, numbers=False, direction_vec=True,
                         velocity_vec=False, plot_frame_number=False):
        if ax is None:
            plt.clf()
            ax = plt.gca()
        if not self.open_drive:
            ax.imshow(self.map_image, extent=self.extent)
        else:
            self._draw_xodr_map(ax)
            self.extent = (self.map_center[0] - self.fov / 2, self.map_center[0] + self.fov / 2) +\
                (self.map_center[1] - self.fov / 2, self.map_center[1] + self.fov / 2)
            ax.set_xlim((self.extent[0], self.extent[1]))
            ax.set_ylim((self.extent[2], self.extent[3]))
        self.current_ax = ax

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.frame_label = None

        self.numbers = numbers
        self.direction_vec = direction_vec
        self.velocity_vec = velocity_vec
        self.plot_frame_number = plot_frame_number

        self._update_frame_to(0)

    def _get_color(self,agent_idx,color_list):
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
        for i, (agent, agent_attribute) in enumerate(zip(self.agent_states_history[frame_idx], self.agent_attributes)):
            self._update_agent(i, agent, agent_attribute)

        if self.traffic_lights_history[frame_idx] is not None:
            for light_id, light_state in self.traffic_lights_history[frame_idx].items():
                self._plot_traffic_light(light_id, light_state)

        if self.plot_frame_number:
            if self.frame_label is None:
                self.frame_label = self.current_ax.text(
                    self.extent[0], self.extent[2], str(frame_idx), c='r', fontsize=18)
            else:
                self.frame_label.set_text(str(frame_idx))

        if not self.open_drive:
            self.current_ax.set_xlim(*self.extent[0:2])
            self.current_ax.set_ylim(*self.extent[2:4])

    def _update_agent(self, agent_idx, agent, agent_attribute):
        l, w = agent_attribute.length, agent_attribute.width
        x, y = agent.center.x, agent.center.y
        v = agent.speed
        psi = agent.orientation
        box = np.array([
            [0, 0], [l * 0.5, 0],  # direction vector
            [0, 0], [v * 0.5, 0],  # speed vector at (0.5 m / s ) / m
        ])
        box = np.matmul(rot(psi), box.T).T + np.array([[x, y]])
        if self.direction_vec:
            if agent_idx not in self.dir_lines:
                self.dir_lines[agent_idx] = self.current_ax.plot(
                    box[0:2, 0], box[0:2, 1], lw=2.0, c=self.dir_c)[0]  # plot the direction vector
            else:
                self.dir_lines[agent_idx].set_xdata(box[0:2, 0])
                self.dir_lines[agent_idx].set_ydata(box[0:2, 1])

        if self.velocity_vec:
            if agent_idx not in self.v_lines:
                self.v_lines[agent_idx] = self.current_ax.plot(
                    box[2:4, 0], box[2:4, 1], lw=1.5, c=self.v_c)[0]  # plot the speed
            else:
                self.v_lines[agent_idx].set_xdata(box[2:4, 0])
                self.v_lines[agent_idx].set_ydata(box[2:4, 1])
        if (type(self.numbers) == bool and self.numbers) or \
           (type(self.numbers) == list and agent_idx in self.numbers):
            if agent_idx not in self.box_labels:
                self.box_labels[agent_idx] = self.current_ax.text(
                    x, y, str(agent_idx), c='r', fontsize=18)
                self.box_labels[agent_idx].set_clip_on(True)
            else:
                self.box_labels[agent_idx].set_x(x)
                self.box_labels[agent_idx].set_y(y)

        lw = 1
        fc = self._get_color(agent_idx,self.agent_face_colors)
        if not fc:
            if agent_idx in self.conditional_agents:
                fc = self.cond_c
            else:
                fc = self.agent_c
        ec = self._get_color(agent_idx,self.agent_edge_colors)
        if not ec:
            lw = 0
            ec = fc

        rect = Rectangle((x - l / 2, y - w / 2), l, w, angle=psi *
                         180 / np.pi, rotation_point='center', fc=fc, ec=ec, lw=lw)
        if agent_idx in self.actor_boxes:
            self.actor_boxes[agent_idx].remove()
        self.actor_boxes[agent_idx] = rect
        self.actor_boxes[agent_idx].set_clip_on(True)
        self.current_ax.add_patch(self.actor_boxes[agent_idx])

    def _plot_traffic_light(self, light_id, light_state):
        light = self.traffic_lights[light_id]
        x, y = light.center.x, light.center.y
        psi = light.orientation
        l, w = light.length, light.width

        rect = Rectangle((x - l / 2, y - w / 2), l, w, angle=psi * 180 / np.pi,
                         rotation_point='center',
                         fc=self.traffic_light_colors[light_state], lw=0)
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
        with open(self.open_drive) as f:
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
            if i == len(positions) or (pos[0] == 'lane' and i > 0 and current_lane_id == '0'):

                if current_lane_section == '0':
                    road_id.append(int(current_road_id))
                    index = int(len(ref_x[-1]) / 3.0)
                    h = ref_h[-1][index]
                    road_id_x.append(
                        ref_x[-1][index] + (text_x_offset * math.cos(h) - text_y_offset * math.sin(h)))
                    road_id_y.append(
                        ref_y[-1][index] + (text_x_offset * math.sin(h) + text_y_offset * math.cos(h)))
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

            if pos[0] == 'lane':
                current_road_id = pos[1]
                current_lane_section = pos[2]
                current_lane_id = pos[3]
                if pos[3] == '0':
                    ltype = 'ref'
                    ref_x.append([])
                    ref_y.append([])
                    ref_z.append([])
                    ref_h.append([])

                elif pos[4] == 'no-driving':
                    ltype = 'border'
                    border_x.append([])
                    border_y.append([])
                    border_z.append([])
                    border_h.append([])
                else:
                    ltype = 'lane'
                    lane_x.append([])
                    lane_y.append([])
                    lane_z.append([])
                    lane_h.append([])
            else:
                if ltype == 'ref':
                    ref_x[-1].append(float(pos[0]))
                    ref_y[-1].append(float(pos[1]))
                    ref_z[-1].append(float(pos[2]))
                    ref_h[-1].append(float(pos[3]))

                elif ltype == 'border':
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
            ax.plot(lane_x[i], lane_y[i], linewidth=1.0, color='#222222')

        # plot road ref line segments
        for i in range(len(ref_x)):
            ax.plot(ref_x[i], ref_y[i], linewidth=2.0, color='#BB5555')

        # plot border lanes in gray
        for i in range(len(border_x)):
            ax.plot(border_x[i], border_y[i], linewidth=1.0, color='#AAAAAA')

        if extras:
            # plot red dots indicating lane dections
            for i in range(len(lane_section_dots_x)):
                ax.plot(lane_section_dots_x[i], lane_section_dots_y[i], 'o', ms=4.0, color='#BB5555')

            for i in range(len(road_start_dots_x)):
                # plot a yellow dot at start of each road
                ax.plot(road_start_dots_x[i], road_start_dots_y[i], 'o', ms=5.0, color='#BBBB33')
                # and an arrow indicating road direction
                ax.arrow(road_start_dots_x[i], road_start_dots_y[i], arrow_dx[i],
                         arrow_dy[i], width=0.1, head_width=1.0, color='#BB5555')
            # plot road id numbers
            for i in range(len(road_id)):
                ax.text(road_id_x[i], road_id_y[i], road_id[i], size=text_size,
                        ha='center', va='center', color='#3333BB')

        return None
