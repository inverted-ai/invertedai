import time
from pydantic import BaseModel, validate_call
from typing import List, Optional, Dict, Tuple
from itertools import product
from tqdm.contrib import tenumerate
import numpy as np

import invertedai as iai
from invertedai.large.common import Region
from invertedai.api.initialize import InitializeResponse
from invertedai.common import TrafficLightStatesDict, Point
from invertedai.utils import get_default_agent_attributes

SLACK = 2
AGENT_SCOPE_FOV_BUFFER = 20

@validate_call
def define_regions_grid(
    map_center: Optional[Tuple[float,float]] = (0.0,0.0),
    width: Optional[float] = 100.0, 
    height: Optional[float] = 100.0, 
    stride: Optional[float] = 50.0
) -> List[Region]:
    def check_valid_center(center):
        return (map_center[0] - width) < center[0] < (map_center[0] + width) and \
            (map_center[1] - height) < center[1] < (map_center[1] + height)

    def get_neighbors(center):
        return [
            (center[0] + (i * stride), center[1] + (j * stride))
            for i, j in list(product(*[(-1, 1),]* 2))
        ]

    queue, centers = [map_center], []

    while queue:
        center = queue.pop(0)
        neighbors = filter(check_valid_center, get_neighbors(center))
        queue.extend([
            neighbor
            for neighbor in neighbors
            if neighbor not in queue and neighbor not in centers
        ])
        if center not in centers and check_valid_center(center):
            centers.append(center)
    
    regions = [None]*len(centers)
    for i, center in enumerate(centers):
        regions[i] = Region.fromlist([Point.fromlist(list(center))])

    return regions

@validate_call
def get_regions_density_by_road_area(
    location: str,
    regions: List[Region],
    max_agent_density: Optional[int] = 10,
    scaling_factor: Optional[float] = 1.0,
    display_progress_bar: Optional[bool] = True
) -> List[Region]:
    #Get fraction of image that is a drivable surface (assume all non-black pixels are drivable)
    center_road_area_dict = {}
    max_drivable_area_ratio = 0
    
    iterable_regions = None
    if display_progress_bar:
        iterable_regions = tenumerate(
            regions, 
            total=len(regions),
            desc=f"Calculating drivable surface areas"
        )
    else:
        iterable_regions = regions

    for i, region in iterable_regions:
        #Naively check every square within requested area
        center_tuple = (region.center.x, region.center.y)
        birdview = iai.location_info(
            location=location,
            rendering_fov=region.get_region_fov(),
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

    new_regions = [None]*len(regions)
    for i, (region_center, drivable_ratio) in enumerate(center_road_area_dict.items()):
        num_agents = _calculate_agent_density_max_scaled(max_agent_density,scaling_factor,drivable_ratio,max_drivable_area_ratio)
        new_regions[i] = Region.fromlist([Point.fromlist(list(region_center))],agent_attributes=get_default_agent_attributes({"car":num_agents}))

    return new_regions

def _calculate_agent_density_max_scaled(agent_density,scaling_factor,drivable_ratio,max_drivable_area_ratio):
    return round(agent_density*(1-scaling_factor*(max_drivable_area_ratio-drivable_ratio)/max_drivable_area_ratio)) if drivable_ratio > 0.0 else 0

def _get_all_existing_agents_from_regions(regions):
    agent_states = []
    agent_attributes = []
    recurrent_states = []
    for region in regions:
        region_agent_states = region.agent_states
        agent_states.extend(region_agent_states)
        agent_attributes.extend(region.agent_attributes[:len(region_agent_states)])
        recurrent_states.extend(region.recurrent_states)
    
    return agent_states, agent_attributes, recurrent_states

@validate_call
def region_initialize(
    location: str,
    regions: List[Region],
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_birdview: Optional[bool] = False,
    get_infractions: Optional[bool] = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None,
    display_progress_bar: Optional[bool] = True   
) -> InitializeResponse:
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

    regions:
        List of regions that contains information about the location, shape, and number of agents within the region.
    
    traffic_light_state_history:
        History of traffic light states - the list is over time, in chronological order, i.e.
        the last element is the current state. If there are traffic lights in the map, 
        not specifying traffic light state is equivalent to using iai generated light states.

    get_birdview:
        If True, a birdview image will be returned representing the current world. Note this will significantly
        impact on the latency.

    get_infractions:
        If True, infraction metrics will be returned for each agent.
    
    random_seed:
        Controls the stochastic aspects of initialization for reproducibility.

    api_model_version:
        Optionally specify the version of the model. If None is passed which is by default, the best model will be used.

    display_progress_bar:
        If True, a bar is displayed showing the progress of all relevant processes.
    
    See Also
    --------
    :func:`initialize`
    """

    agent_states_sampled = []
    agent_attributes_sampled = []
    agent_rs_sampled = []

    def inside_fov(center: Point, agent_scope_fov: float, point: Point) -> bool:
        return ((center.x - (agent_scope_fov / 2) < point.x < center.x + (agent_scope_fov / 2)) and
                (center.y - (agent_scope_fov / 2) < point.y < center.y + (agent_scope_fov / 2)))
    
    iterable_regions = None
    if display_progress_bar:
        iterable_regions = tenumerate(
            regions, 
            total=len(regions),
            desc=f"Calculating drivable surface areas"
        )
    else:
        iterable_regions = regions

    for i, region in iterable_regions:
        region_center = region.center
        region_fov = region.get_region_fov()

        existing_agent_states, existing_agent_attributes, _ = _get_all_existing_agents_from_regions(regions)

        conditional_agents = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=region_fov+AGENT_SCOPE_FOV_BUFFER, point=x[0].center), 
            zip(existing_agent_states,existing_agent_attributes)
        ))

        conditional_agent_states = [x[0] for x in conditional_agents]
        conditional_agent_attributes = [x[1] for x in conditional_agents]

        region_predefined_agent_states = [] if region.agent_states is None else region.agent_states
        all_agent_states = conditional_agent_states + region_predefined_agent_states
        all_agent_attributes = conditional_agent_attributes + region.agent_attributes

        try:
            # Initialize simulation with an API call
            response = iai.initialize(
                location=location,
                states_history=[all_agent_states],
                agent_attributes=all_agent_attributes,
                get_infractions=get_infractions,
                traffic_light_state_history=traffic_light_state_history,
                location_of_interest=(region_center.x, region_center.y),
                random_seed=random_seed,
                get_birdview=get_birdview,
            )

            if traffic_light_state_history is None and response.traffic_lights_states is not None:
                # If no traffic light states are given, take the first non-None traffic light states output as the consistent traffic light states across all areas
                traffic_light_state_history = [response.traffic_lights_states]

        except InvertedAIError as e:
            iai.logger.warning(e)

        # Remove all predefined agents before filtering at the edges
        response_agent_states_sampled = response.agent_states[len(all_agent_states):]
        response_agent_attributes_sampled = response.agent_attributes[len(all_agent_states):]
        response_recurrent_states_sampled = response.recurrent_states[len(all_agent_states):]

        # Filter out agents that are not inside the ROI to avoid collision with other agents not passed as conditional
        # SLACK is for removing the agents that are very close to the boundary and
        # they may collide agents not filtered as conditional
        valid_agents = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=region_fov - SLACK, point=x[0].center),
            zip(response_agent_states_sampled, response_agent_attributes_sampled, response_recurrent_states_sampled)
        ))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]

        predefined_agents_slice = slice(len(conditional_agent_states),len(conditional_agent_states)+len(region_predefined_agent_states))
        regions[i].agent_states = response.agent_states[predefined_agents_slice] + valid_agent_state
        regions[i].agent_attributes = response.agent_attributes[predefined_agents_slice] + valid_agent_attrs
        regions[i].recurrent_states = response.recurrent_states[predefined_agents_slice] + valid_agent_rs

        # if get_birdview:
        #     file_path = f"{save_birdviews_to}_{(region_center.x, region_center.y)}.jpg"
        #     response.birdview.decode_and_save(file_path)

    all_agent_states, all_agent_attributes, all_recurrent_states = _get_all_existing_agents_from_regions(regions)

    response.agent_states = all_agent_states
    response.recurrent_states = all_agent_attributes
    response.agent_attributes = all_recurrent_states

    return response