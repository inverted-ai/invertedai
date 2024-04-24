import time
from pydantic import BaseModel, validate_call
from typing import List, Optional, Tuple
from itertools import product
from tqdm.contrib import tenumerate
import numpy as np

import invertedai as iai
from invertedai.large.common import Region
from invertedai.api.initialize import InitializeResponse
from invertedai.common import TrafficLightStatesDict, Point
from invertedai.utils import get_default_agent_attributes
from invertedai.error import InvertedAIError

AGENT_SCOPE_FOV_BUFFER = 20

@validate_call
def define_regions_grid(
    width: float,
    height: float,
    map_center: Optional[Tuple[float,float]] = (0.0,0.0), 
    stride: Optional[float] = 50.0
) -> List[Region]:
    """
    A utility function to help initialize an area larger than 100x100m. This function breaks up an 
    area into a grid of 100x100m regions, each of which can be initialized directly.

    Arguments
    ----------
    width:
        The horizontal size of the area which will be broken into smaller regions.

    height:
        The vertical size of the area which will be broken into smaller regions.

    map_center:
        The coordinates of the center of the rectangular area to be broken into smaller regions.

    stride:
        How far apart the centers of the 100x100m regions should be. Some overlap is recommended for best results.
    
    """
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
        regions[i] = Region.init_square_region(center=Point.fromlist(list(center)))

    return regions

@validate_call
def get_number_of_agents_per_region_by_drivable_area(
    location: str,
    regions: List[Region],
    max_agents_per_region: Optional[int] = 10,
    display_progress_bar: Optional[bool] = True
) -> List[Region]:
    """
    This function takes in a list of regions, infers the total drivable surface area per region by calculating 
    the number of black pixels (non-drivable surfaces), then inserts a number of default AgentAttributes objects
    into the region to be initialized porportional to its drivable surface area relative to the other regions.

    Arguments
    ----------
    location:
        Location name in IAI format.

    regions:
        A list of empty Regions (i.e. no pre-existing agent information) for which the number of agents to 
        initialize is calculated. 

    max_agents_per_region:
        The maximum number of agents that can be initialized in any region. The region with the largest drivable
        surface area will have this many agents initialized while other regions will have equal or fewer number 
        of agents.

    display_progress_bar:
        A flag to control whether a command line progress bar is displayed for convenience.
    
    """

    center_road_area_dict = {}
    max_drivable_area_ratio = 0
    
    if display_progress_bar:
        iterable_regions = tenumerate(
            regions, 
            total=len(regions),
            desc=f"Calculating drivable surface areas"
        )
    else:
        iterable_regions = regions

    for i, region in iterable_regions:
        center_tuple = (region.center.x, region.center.y)
        birdview = iai.location_info(
            location=location,
            rendering_fov=int(region.size),
            rendering_center=center_tuple
        ).birdview_image.decode()

        birdview_arr_shape = birdview.shape
        total_num_pixels = birdview_arr_shape[0]*birdview_arr_shape[1]
        number_of_black_pix = np.sum(birdview.sum(axis=-1) == 0)

        drivable_area_ratio = (total_num_pixels-number_of_black_pix)/total_num_pixels 
        center_road_area_dict[center_tuple] = drivable_area_ratio

        if drivable_area_ratio > max_drivable_area_ratio:
            max_drivable_area_ratio = drivable_area_ratio

    for i, (region_center, drivable_ratio) in enumerate(center_road_area_dict.items()):
        num_agents = _calculate_agent_density_max_scaled(
            agent_density=max_agents_per_region,
            scaling_factor=1.0,
            drivable_ratio=drivable_ratio,
            max_drivable_area_ratio=max_drivable_area_ratio
        )
        regions[i].agent_attributes.extend(get_default_agent_attributes({"car":num_agents}))

    return regions

def _calculate_agent_density_max_scaled(agent_density,scaling_factor,drivable_ratio,max_drivable_area_ratio):
    return max(round(agent_density*(1-scaling_factor*(max_drivable_area_ratio-drivable_ratio)/max_drivable_area_ratio)) if drivable_ratio > 0.0 else 0, 0)

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
def large_initialize(
    location: str,
    regions: List[Region],
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
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
    predefined agents may be passed via the regions and will be considered as conditional. 

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
    light_recurrent_states = None

    def inside_fov(center: Point, agent_scope_fov: float, point: Point) -> bool:
        return ((center.x - (agent_scope_fov / 2) < point.x < center.x + (agent_scope_fov / 2)) and
                (center.y - (agent_scope_fov / 2) < point.y < center.y + (agent_scope_fov / 2)))
    
    if display_progress_bar:
        iterable_regions = tenumerate(
            regions, 
            total=len(regions),
            desc=f"Initializing regions"
        )
    else:
        iterable_regions = regions

    for i, region in iterable_regions:
        region_center = region.center
        region_size = region.size

        existing_agent_states, existing_agent_attributes, _ = _get_all_existing_agents_from_regions(regions)

        # Acquire agents that exist in other regions that must be passed as conditional to avoid collisions
        out_of_region_conditional_agents = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=region_size+AGENT_SCOPE_FOV_BUFFER, point=x[0].center), 
            zip(existing_agent_states,existing_agent_attributes)
        ))

        out_of_region_conditional_agent_states = [x[0] for x in out_of_region_conditional_agents]
        out_of_region_conditional_agent_attributes = [x[1] for x in out_of_region_conditional_agents]

        region_predefined_agent_states = [] if region.agent_states is None else region.agent_states
        all_agent_states = out_of_region_conditional_agent_states + region_predefined_agent_states
        all_agent_attributes = out_of_region_conditional_agent_attributes + region.agent_attributes

        try:
            if len(all_agent_attributes) > 0:
                response = iai.initialize(
                    location=location,
                    states_history=None if len(all_agent_states) == 0 else [all_agent_states],
                    agent_attributes=all_agent_attributes,
                    get_infractions=get_infractions,
                    traffic_light_state_history=traffic_light_state_history,
                    location_of_interest=(region_center.x, region_center.y),
                    random_seed=random_seed
                )

                if traffic_light_state_history is None and response.traffic_lights_states is not None:
                    traffic_light_state_history = [response.traffic_lights_states]
                    light_recurrent_states = response.light_recurrent_states
            else:
                #There are no agents to initialize within this region
                continue

        except InvertedAIError as e:
            iai.logger.warning(e)
            continue

        # Filter out conditional agents from other regions
        num_out_of_region_conditional_agents = len(out_of_region_conditional_agent_states)
        regions[i].clear_agents()
        for state, attrs, r_state in zip(
            response.agent_states[num_out_of_region_conditional_agents:],
            response.agent_attributes[num_out_of_region_conditional_agents:],
            response.recurrent_states[num_out_of_region_conditional_agents:]
        ):
            regions[i].insert_all_agent_details(state,attrs,r_state)

    all_agent_states, all_agent_attributes, all_recurrent_states = _get_all_existing_agents_from_regions(regions)

    response.agent_states = all_agent_states
    response.agent_attributes = all_agent_attributes
    response.recurrent_states = all_recurrent_states 
    response.light_recurrent_states = light_recurrent_states
    
    return response