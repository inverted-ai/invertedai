import asyncio
import json
import os
import re
import numpy as np
import csv
import math
import logging

from typing import Dict, Optional, List, Tuple
from tqdm.contrib import tmap
from itertools import product
from copy import deepcopy
from pydantic import validate_call

import requests
from requests import Response
from requests.auth import AuthBase
from requests.adapters import HTTPAdapter, Retry

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib import transforms

import invertedai as iai
import invertedai.api
import invertedai.api.config
from invertedai import error
from invertedai.common import AgentState, AgentAttributes, StaticMapActor, TrafficLightStatesDict, Point
from invertedai.future import to_thread
from invertedai.error import InvertedAIError

H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7
TIMEOUT_SECS = 600
MAX_RETRIES = 10
SLACK = 2
AGENT_SCOPE_FOV = 120
AGENT_FOV = 35

logger = logging.getLogger(__name__)

class Session:
    def __init__(self):
        self.session = requests.Session()
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
                "x-client-version": iai.__version__
            }
        )
        self._base_url = self._get_base_url()

    @property
    def base_url(self):
        return self._base_url

    @base_url.setter
    def base_url(self, value):
        self._base_url = value


    def _verify_api_key(self, api_token: str, verifying_url: str):
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
            logger.warning("Commercial access denied and fallback to check for academic access.")
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


    def add_apikey(self, api_token: str = "", key_type: Optional[str] = None, url: Optional[str] = None):
        """
        Bind an API key to the session for authentication.

        Args:
            api_token (str): The API key to be added. Defaults to an empty string.
            key_type (str, optional): The type of API key. Defaults to None. When passed, the base_url will be set according to the key_type.
            url (str, optional): The URL to be used for the request. Defaults to None. When passed, the base_rul will be set to the passed value and the key_type will be ignored.

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
            base_url = iai.commercial_url # Default to commercial when initializing.
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
                                       initialize_fov=AGENT_SCOPE_FOV, get_birdview=False, birdview_path=None):
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
            return [], [], [], ""
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

        return valid_agent_state, valid_agent_attrs, valid_agent_rs, response.api_model_version

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

    model_version = ""
    for result in results:
        new_agent_state += result[0]
        new_attributes += result[1]
        new_recurrent_states += result[2]
        model_version = result[3]

    return invertedai.api.InitializeResponse(
        recurrent_states=new_recurrent_states,
        agent_states=new_agent_state,
        agent_attributes=new_attributes,
        api_model_version=model_version
    )

def area_re_initialization(location, agent_attributes, states_history, traffic_lights_states=None, random_seed=None,
                           map_center=(0, 0), width=100, height=100, initialize_fov=AGENT_SCOPE_FOV, get_birdview=False,
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

    response.recurrent_states = new_recurrent_states
    response.agent_states = new_agent_state
    response.agent_attributes = new_attributes
    return response

def _get_agent_density_per_region(centers,location,agent_density,scaling_factor,display_progress_bar):
    #Get fraction of image that is a drivable surface (assume all non-black pixels are drivable)
    center_road_area_dict = {}
    max_drivable_area_ratio = 0
    
    iterable_regions = None
    if display_progress_bar:
        iterable_regions = tmap(
            Point.fromlist, 
            centers, 
            total=len(centers),
            desc=f"Calculating drivable surface areas"
        )
    else:
        iterable_regions = [Point.fromlist(list(center)) for center in centers]

    for area_center in iterable_regions:
        #Naively check every square within requested area
        #TODO: Use heuristics or other methods to (e.g. map polygon, high FOV image, quadtree) to reduce computation time
        center_tuple = (area_center.x, area_center.y)
        birdview = iai.location_info(
            location=location,
            rendering_fov=100,
            rendering_center=center_tuple
        ).birdview_image.decode()

        ## Get number of black pixels
        birdview_arr_shape = birdview.shape
        total_num_pixels = birdview_arr_shape[0]*birdview_arr_shape[1]
        # Convert to grayscale using Luminosity Method: gray = 0.114*B + 0.587*G + 0.299*R
        # Image should be in BGR color pixel format
        birdview_grayscale = np.matmul(birdview.reshape(total_num_pixels,3),np.array([[0.114],[0.587],[0.299]]))
        number_of_black_pix = np.sum(birdview_grayscale == 0)

        drivable_area_ratio = (total_num_pixels-number_of_black_pix)/total_num_pixels 
        center_road_area_dict[center_tuple] = drivable_area_ratio

        if drivable_area_ratio > max_drivable_area_ratio:
            max_drivable_area_ratio = drivable_area_ratio

    agent_density_list = []
    for area_center, drivable_ratio in center_road_area_dict.items():
        agent_density_list.append(_calculate_agent_density_max_scaled(agent_density,scaling_factor,drivable_ratio,max_drivable_area_ratio))

    return agent_density_list

def _calculate_agent_density_max_scaled(agent_density,scaling_factor,drivable_ratio,max_drivable_area_ratio):

    return round(agent_density*(1-scaling_factor*(max_drivable_area_ratio-drivable_ratio)/max_drivable_area_ratio)) if drivable_ratio > 0.0 else 0

@validate_call
def area_initialization(
    location: str, 
    agent_density: int, 
    agent_attributes: Optional[List[AgentAttributes]] = None,
    states_history: Optional[List[List[AgentState]]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None, 
    random_seed: Optional[int] = None, 
    map_center: Optional[Tuple[float,float]] = (0.0,0.0),
    width: Optional[float] = 100.0, 
    height: Optional[float] = 100.0, 
    stride: Optional[float] = 50.0, 
    scaling_factor: Optional[float] = 1.0, 
    save_birdviews_to: Optional[str] = None,
    display_progress_bar: Optional[bool] = True
):
    """
    A utility function to initialize an area larger than 100x100m. This function breaks up an 
    area into a grid of 100x100m regions and runs initialize on them all. For any particular 
    region of interest, existing agents in overlapping, neighbouring regions are passed as 
    conditional agents to :func:`initialize`. Regions will be rejected and :func:`initialize` 
    will not be called if it is not possible for agents to exist there (e.g. there are no 
    drivable surfaces present) or if the agent density criterion is already satisfied. As well,
    predefined agents may be passed and will be considered as conditional within the 
    appropriate region.

    Arguments
    ----------
    location:
        Location name in IAI format.
    agent_density:
        Maximum agents per 100x100m region to be scaled based on heuristic.
    agent_attributes:
        Static attributes for all pre-defined agents ONLY. Use the agent_density argument to 
        specify the number of agents to be sampled. 
    states_history:
        History of pre-defined agent states - the outer list is over time and the inner over agents,
        in chronological order, i.e., index 0 is the oldest state and index -1 is the current state.
        The order of agents should be the same as in `agent_attributes`.
        For best results, provide at least 10 historical states for each agent.
    traffic_light_state_history:
        History of traffic light states - the list is over time, in chronological order, i.e.
        the last element is the current state. If there are traffic lights in the map, 
        not specifying traffic light state is equivalent to using iai generated light states.
    random_seed:
        Controls the stochastic aspects of initialization for reproducibility.
    map_center:
        The x,y coordinate of the center of the area to be initialized.
    width:
        Distance along the x-axis from the area center to edge of the rectangular area (total 
        width of the region is 2X the value of this parameter).
    height:
        Distance along the y-axis from the area center to edge of the rectangular area (total 
        height of the region is 2X the value of this parameter).
    stride:
        Distance between the centers of the 100x100m regions.
    scaling_factor:
        A factor between [0,1] weighting the heuristic for number of agents to spawn in a region. 
        For example, a value of 0 ignores the heuristic and results in requesting the same number 
        of agents for all regions.
    save_birdviews_to:
        If this variable is not None, the birdview images will be saved to this specified path.
    display_progress_bar:
        If True, a bar is displayed showing the progress of all relevant processes.
    
    See Also
    --------
    :func:`initialize`
    """
    get_birdview = save_birdviews_to is not None

    agent_states_sampled = []
    agent_attributes_sampled = []
    agent_rs_sampled = []

    def inside_fov(center: Point, agent_scope_fov: float, point: Point) -> bool:
        return ((center.x - (agent_scope_fov / 2) < point.x < center.x + (agent_scope_fov / 2)) and
                (center.y - (agent_scope_fov / 2) < point.y < center.y + (agent_scope_fov / 2)))

    centers = get_centers(
        map_center, 
        height, 
        width, 
        stride
    )

    agent_density_list = _get_agent_density_per_region(centers,location,agent_density,scaling_factor,display_progress_bar)

    predefined_agent_recurrent_state_dict = {}
    if states_history is None: states_history = [[]]
    if agent_attributes is None: agent_attributes = []
    for agent in states_history[-1]:
        predefined_agent_recurrent_state_dict[(agent.center.x,agent.center.y)] = None

    iterable_regions = None
    if display_progress_bar:
        iterable_regions = tmap(
            Point.fromlist, 
            centers, 
            total=len(centers),
            desc=f"Initializing {location.split(':')[1]}"
        )
    else:
        iterable_regions = [Point.fromlist(list(center)) for center in centers]

    for i, region_center in enumerate(iterable_regions):

        # Separate all sampled agents into agents within the region of interest vs remaining outside of the region
        conditional_agents_sampled = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=AGENT_SCOPE_FOV, point=x[0].center), 
            zip(agent_states_sampled,agent_attributes_sampled,agent_rs_sampled)
        ))

        remaining_agents = list(filter(
            lambda x: not inside_fov(center=region_center, agent_scope_fov=AGENT_SCOPE_FOV, point=x[0].center), 
            zip(agent_states_sampled,agent_attributes_sampled,agent_rs_sampled)
        ))

        states_history_sampled = [x[0] for x in conditional_agents_sampled]
        agent_attributes_sampled_conditional = [x[1] for x in conditional_agents_sampled]
        agent_recurrent_states_sampled = [x[2] for x in conditional_agents_sampled]
        remaining_agents_states = [x[0] for x in remaining_agents]
        remaining_agents_attrs = [x[1] for x in remaining_agents]
        remaining_agents_rs = [x[2] for x in remaining_agents]

        # Separate all predefined agents into agents within the region of interest vs not
        # Contactenate predefined and samped agents within the region of interest, these are now considered conditional to initialize()
        conditional_agents_predefined = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=AGENT_SCOPE_FOV, point=x[0].center), 
            zip(states_history[-1],agent_attributes)
        ))

        states_history_predefined = [x[0] for x in conditional_agents_predefined]
        states_history_region_conditional = states_history_predefined + states_history_sampled
        states_history_region_conditional = [states_history_region_conditional] if len(states_history_region_conditional) > 0 else None

        agent_attributes_predefined_conditional = [x[1] for x in conditional_agents_predefined]
        agent_attributes_region_conditional = agent_attributes_predefined_conditional + agent_attributes_sampled_conditional
        num_conditional_agents = len(agent_attributes_region_conditional)
        agent_attributes_region_conditional = agent_attributes_region_conditional if num_conditional_agents > 0 else None

        agent_density_region = agent_density_list[i]
        num_agents_to_spawn = agent_density_region - num_conditional_agents

        # Check if this initialization call can be skipped
        if len(agent_attributes_predefined_conditional) <= 0:
            if agent_density_region <= 0:
                #Skip if no agents are requested (e.g. in regions with no drivable surfaces)
                continue
            elif num_agents_to_spawn <= 0:
                #Skip if the calculated number of agents has been satisfied or surpassed
                continue

        try:
            all_agents_attributes_in_region = deepcopy(agent_attributes_region_conditional) if agent_attributes_region_conditional is not None else []
            for _ in range(num_agents_to_spawn):
                # Pad agent attributes list with default values
                all_agents_attributes_in_region.append(AgentAttributes.fromlist(["car"]))

            # Initialize simulation with an API call
            response = iai.initialize(
                location=location,
                states_history=states_history_region_conditional,
                agent_attributes=all_agents_attributes_in_region,
                agent_count=agent_density_region,
                get_infractions=False,
                traffic_light_state_history=traffic_light_state_history,
                location_of_interest=(region_center.x, region_center.y),
                random_seed=random_seed,
                get_birdview=get_birdview,
            )

            if traffic_light_state_history is None:
                # If no traffic light states are given, take the first non-None traffic light states output as the consistent traffic light states across all areas
                traffic_light_state_history = [response.traffic_lights_states]

        except InvertedAIError as e:
            iai.logger.warning(e)

        # Get the recurrent state for any predefined agents in this region
        for s, rs in zip(response.agent_states[:len(states_history_predefined)],response.recurrent_states[:len(states_history_predefined)]):
            predefined_agent_recurrent_state_dict[(s.center.x,s.center.y)] = rs

        # Remove all predefined agents before filtering at the edges
        response_agent_states_sampled = response.agent_states[len(states_history_predefined):]
        response_agent_attributes_sampled = response.agent_attributes[len(states_history_predefined):]
        response_recurrent_states_sampled = response.recurrent_states[len(states_history_predefined):]

        # Filter out agents that are not inside the ROI to avoid collision with other agents not passed as conditional
        # SLACK is for removing the agents that are very close to the boundary and
        # they may collide agents not filtered as conditional
        valid_agents = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=AGENT_SCOPE_FOV - SLACK, point=x[0].center),
            zip(response_agent_states_sampled, response_agent_attributes_sampled, response_recurrent_states_sampled)
        ))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]

        agent_states_sampled = remaining_agents_states + valid_agent_state
        agent_attributes_sampled = remaining_agents_attrs + valid_agent_attrs
        agent_rs_sampled = remaining_agents_rs + valid_agent_rs

        if get_birdview:
            file_path = f"{birdview_path}-{(region_center.x, region_center.y)}.jpg"
            response.birdview.decode_and_save(file_path)

    predefined_agent_recurrent_states = []
    for s in states_history[-1]:
        predefined_agent_recurrent_states.append(predefined_agent_recurrent_state_dict[(s.center.x,s.center.y)])

    response.agent_states = states_history[-1] + agent_states_sampled
    response.recurrent_states = predefined_agent_recurrent_states + agent_rs_sampled
    response.agent_attributes = agent_attributes + agent_attributes_sampled

    return response



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


class ScenePlotter:
    def __init__(self, map_image=None, fov=None, xy_offset=None, static_actors=None,
                 open_drive=None, resolution=(640, 480), dpi=100):
        self.conditional_agents = None
        self.agent_attributes = None
        self.traffic_lights_history = None
        self.agent_states_history = None
        self.open_drive = open_drive
        self.dpi = dpi
        self.resolution = resolution
        self.fov = fov
        self.map_image = map_image
        if not open_drive:
            self.extent = (- self.fov / 2 + xy_offset[0], self.fov / 2 + xy_offset[0]) + \
                (- self.fov / 2 + xy_offset[1], self.fov / 2 + xy_offset[1])
        else:
            self.map_center = xy_offset

        self.traffic_lights = {static_actor.actor_id: static_actor
                               for static_actor in static_actors
                               if static_actor.agent_type == 'traffic-light'}

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

        self.reset_recording()

        self.numbers = False

    def initialize_recording(self, agent_states, agent_attributes, traffic_light_states=None, conditional_agents=None):
        self.agent_states_history = [agent_states]
        self.traffic_lights_history = [traffic_light_states]
        self.agent_attributes = agent_attributes
        if conditional_agents is not None:
            self.conditional_agents = conditional_agents
        else:
            self.conditional_agents = []

    def reset_recording(self):
        self.agent_states_history = []
        self.traffic_lights_history = []
        self.agent_attributes = None
        self.conditional_agents = []

    def record_step(self, agent_states, traffic_light_states=None):
        self.agent_states_history.append(agent_states)
        self.traffic_lights_history.append(traffic_light_states)

    def plot_scene(self, agent_states, agent_attributes, traffic_light_states=None, conditional_agents=None,
                   ax=None, numbers=False, direction_vec=True, velocity_vec=False):
        self.initialize_recording(agent_states, agent_attributes,
                                  traffic_light_states=traffic_light_states,
                                  conditional_agents=conditional_agents)

        self.plot_frame(idx=0, ax=ax, numbers=numbers, direction_vec=direction_vec,
                        velocity_vec=velocity_vec, plot_frame_number=False)

        self.reset_recording()

    def plot_frame(self, idx, ax=None, numbers=False, direction_vec=False,
                   velocity_vec=False, plot_frame_number=False):
        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        self._update_frame_to(idx)

    def animate_scene(self, output_name=None, start_idx=0, end_idx=-1, ax=None,
                      numbers=False, direction_vec=True, velocity_vec=False,
                      plot_frame_number=False):
        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        end_idx = len(self.agent_states_history) if end_idx == -1 else end_idx
        fig = self.current_ax.figure
        fig.set_size_inches(self.resolution[0] / self.dpi, self.resolution[1] / self.dpi, True)

        def animate(i):
            self._update_frame_to(i)

        ani = animation.FuncAnimation(
            fig, animate, np.arange(start_idx, end_idx), interval=100)
        if output_name is not None:
            ani.save(f'{output_name}', writer='pillow', dpi=self.dpi)
        return ani

    def _initialize_plot(self, ax=None, numbers=False, direction_vec=True,
                         velocity_vec=False, plot_frame_number=False):
        if ax is None:
            plt.clf()
            ax = plt.gca()
        if not self.open_drive:
            ax.imshow(self.map_image, extent=self.extent)
        else:
            self._draw_xord_map(ax)
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

        if agent_idx in self.conditional_agents:
            c = self.cond_c
        else:
            c = self.agent_c

        rect = Rectangle((x - l / 2, y - w / 2), l, w, angle=psi *
                         180 / np.pi, rotation_point='center', fc=c, lw=0)
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

    def _draw_xord_map(self, ax, extras=False):
        """
        This function plots the parsed xodr map
        the `odrplot` of `esmini` is used for plotting and parsing xord
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