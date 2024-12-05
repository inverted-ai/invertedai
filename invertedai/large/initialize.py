import time
import numpy as np

from random import choices, seed, randint
from math import sqrt
from copy import deepcopy
from pydantic import BaseModel, validate_call
from typing import Union, List, Optional, Tuple, Dict
from itertools import product
from tqdm.contrib import tenumerate

import invertedai as iai
from invertedai.large.common import Region, REGION_MAX_SIZE
from invertedai.api.initialize import InitializeResponse
from invertedai.utils import get_default_agent_properties
from invertedai.error import InvertedAIError
from invertedai.common import (
    AgentProperties, 
    AgentState, 
    AgentType,
    Point,
    TrafficLightStatesDict   
)

AGENT_SCOPE_FOV_BUFFER = 60
ATTEMPT_PER_NUM_REGIONS = 15


@validate_call
def get_regions_default(
    location: str,
    total_num_agents: Optional[int] = None,
    agent_count_dict: Optional[Dict[AgentType,int]] = None,
    area_shape: Optional[Tuple[float,float]] = None,
    map_center: Optional[Tuple[float,float]] = (0.0,0.0),
    random_seed: Optional[int] = None, 
    display_progress_bar: Optional[bool] = True
) -> List[Region]:
    """
    A utility function to create a set of Regions to be passed into :func:`large_initialize` in
    a single convenient entry point. 

    Arguments
    ----------
    location:
        Location name in IAI format.

    total_num_agents:
        Deprecated. The total number of agents to initialize across all regions.

    agent_count_dict:
        The number of agents to place within the regions per specified agent type.

    area_shape:
        Contains the [width, height] to either side of the center of the rectangular area to be broken into 
        smaller regions (i.e. half of full width and height of the region). If this argument is not provided, 
        a bounding box around the location polygon from :func:`location_info` will be used.

    map_center:
        The coordinates of the center of the rectangular area to be broken into smaller regions. If
        this argument is not provided, default coordinates of (0,0) are used.

    random_seed:
        Controls the stochastic aspects of assigning agents to regions for reproducibility. If this
        argument is not provided, a random seed is chosen.

    display_progress_bar:
        A flag to control whether a command line progress bar is displayed for convenience.
    
    """
    if agent_count_dict is None:
        if total_num_agents is None:
            raise InvertedAIError(message=f"Error: Must specify a number of agents within the regions.")
        else:
            agent_count_dict = {AgentType.car: total_num_agents}

    if area_shape is None:
        location_response = iai.location_info(
            location = location,
            rendering_center = map_center
        )
        polygon_x, polygon_y = [], []
        for pt in location_response.bounding_polygon:
            polygon_x.append(pt.x)
            polygon_y.append(pt.y)
        width = max(polygon_x) - min(polygon_x)
        height = max(polygon_y) - min(polygon_y)

        area_shape = (width,height)

    regions = iai.get_regions_in_grid(
        width = area_shape[0], 
        height = area_shape[1],
        map_center = map_center
    )

    new_regions = iai.get_number_of_agents_per_region_by_drivable_area(
        location = location,
        regions = regions,
        agent_count_dict = agent_count_dict,
        random_seed = random_seed,
        display_progress_bar = display_progress_bar
    )

    return new_regions


@validate_call
def get_regions_in_grid(
    width: float,
    height: float,
    map_center: Optional[Tuple[float,float]] = (0.0,0.0), 
    stride: Optional[float] = 50.0
) -> List[Region]:
    """
    A utility function to help initialize an area larger than 100x100m. This function breaks up an 
    area into a grid of 100x100m regions, each of which can be initialized directly. Each of these
    regions will only have its shaped specified and no agent information. 

    Arguments
    ----------
    width:
        Half of the horizontal size of the area which will be broken into smaller regions.

    height:
        Half of the vertical size of the area which will be broken into smaller regions.

    map_center:
        The coordinates of the center of the rectangular area to be broken into smaller regions. If
        this argument is not provided, default coordinates of (0,0) are used.

    stride:
        How far apart the centers of the 100x100m regions should be. Some overlap is recommended for 
        best results and if no argument is provided, a value of 50 is used.
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
    
    regions = [None for _ in range(len(centers))]
    for i, center in enumerate(centers):
        regions[i] = Region.create_square_region(center=Point.fromlist(list(center)))

    return regions


@validate_call
def get_number_of_agents_per_region_by_drivable_area(
    location: str,
    regions: List[Region],
    total_num_agents: Optional[int] = None,
    agent_count_dict: Optional[Dict[AgentType,int]] = None,
    random_seed: Optional[int] = None,
    display_progress_bar: Optional[bool] = True
) -> List[Region]:
    """
    This function takes in a list of regions, calculates the driveable area for each of them using 
    :func:`location_info`, then creates a new Region object with copied location and shape data and 
    inserts a number of car agents to be **sampled** into it porportional to its drivable surface 
    area relative to the other regions. Regions with no or a relatively small amount of drivable 
    surfaces will be removed. If a region is at its capacity (e.g. due to pre-existing agents), no 
    more agents will be added to it. 

    Arguments
    ----------
    location:
        Location name in IAI format.

    regions:
        A list of empty Regions (i.e. no pre-existing agent information) for which the number of 
        agents to initialize is calculated. 

    total_num_agents:
        Deprecated. The total number of agents to initialize across all regions.

    agent_count_dict:
        The number of agents to place within the regions per specified agent type.

    random_seed:
        Controls the stochastic aspects of assigning agents to regions for reproducibility. If this
        argument is not provided, a random seed is chosen.

    display_progress_bar:
        A flag to control whether a command line progress bar is displayed for convenience.
    """

    if agent_count_dict is None:
        if total_num_agents is None:
            raise InvertedAIError(message=f"Error: Must specify a number of agents within the regions.")
        else:
            agent_count_dict = {AgentType.car: total_num_agents}

    agent_list_types = []
    for agent_type, num_agents in agent_count_dict.items():
        agent_list_types = agent_list_types + [agent_type]*num_agents

    new_regions = [Region.copy(region) for region in regions]
    region_road_area = []
    total_drivable_area_ratio = 0

    if random_seed is not None:
        seed(random_seed)
    
    if display_progress_bar:
        iterable_regions = tenumerate(
            new_regions, 
            total=len(new_regions),
            desc=f"Calculating drivable surface areas"
        )
    else:
        iterable_regions = enumerate(new_regions)

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
        total_drivable_area_ratio += drivable_area_ratio
        region_road_area.append(drivable_area_ratio)

    # Select region in which to assign agents using drivable area as weight
    all_region_weights = [0]*len(new_regions)
    for i, drivable_ratio in enumerate(region_road_area):
        all_region_weights[i] = drivable_ratio/total_drivable_area_ratio
    random_indexes = choices(list(range(len(new_regions))), weights=all_region_weights, k=len(agent_list_types))

    number_sampled_agents = {}
    for ind in random_indexes:
        if ind not in number_sampled_agents:
            number_sampled_agents[ind] = 1
        else:
            number_sampled_agents[ind] += 1

    for agent_id, ind in enumerate(random_indexes):
        if len(new_regions[ind].agent_properties) < number_sampled_agents[ind]:
            new_regions[ind].agent_properties = new_regions[ind].agent_properties + get_default_agent_properties({agent_list_types[agent_id]:1})

    filtered_regions = []
    for region in new_regions:
        if len(region.agent_properties) > 0:
            filtered_regions.append(region)

    return filtered_regions


@validate_call
def _insert_agents_into_nearest_regions(
    regions: List[Region],
    agent_properties: List[AgentProperties],
    agent_states: List[AgentState],
    return_region_index: Optional[bool] = False,
    random_seed: Optional[int] = None
) -> Union[List[Region],Tuple[List[Region],List[Tuple[int,int]]]]:
    """
    Helper function to place pre-existing agents into a group of regions. If agents exist 
    within the bounds of multiple regions, it is placed within the region to which whose 
    center it is closest. Agents will be placed "into" the region that is closest even if 
    it is not within the bounds of the region. The length of the agent_states list must be
    equal or less than the length of agent_properties. To remain compliant with :func:`initialize`, 
    agents with defined agent states are placed at the beginning of the list. Optionally 
    using the return_region_index parameter will return a list indicating in which region 
    the agent is placed to preserve agent indexing. A random seed parameter is included for 
    repeatability.

    Arguments
    ----------
    regions:
        A list of Regions with bounds and centre defined for which agents are associated. 

    agent_states:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    agent_properties:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    return_region_index:
        Whether to map the region in which agents of the same index have been placed. Returns 
        a list of the same size as the agent_properties parameter.
    """
    num_agent_states = len(agent_states)
    num_regions = len(regions)
    assert num_regions > 0, "Invalid parameter: number of regions must be greater than zero."
    assert len(agent_properties) >= num_agent_states, "Invalid parameters: number of agent properties must be larger than number agent states."

    if return_region_index: 
        region_map = []
    else:
        region_map = None

    if len(agent_states) > 0: 
        region_agent_states_lengths = [len(region.agent_states) for region in regions]

        for i, (prop, state) in enumerate(zip(agent_properties[:num_agent_states],agent_states)):
            region_distances = []
            for region in regions:
                region_distances.append(sqrt((state.center.x-region.center.x)**2 + (state.center.y-region.center.y)**2))

            closest_region_index = region_distances.index(min(region_distances))
            insert_index = region_agent_states_lengths[closest_region_index]
            region_agent_states_lengths[closest_region_index] += 1
            regions[closest_region_index].agent_properties.insert(insert_index,prop)
            regions[closest_region_index].agent_states.insert(insert_index,state)

            if return_region_index: region_map.append(tuple([closest_region_index,insert_index]))

    if random_seed is not None: seed(random_seed)
    for prop in agent_properties[num_agent_states:]:
        random_region_index = randint(0,num_regions-1)
        regions[random_region_index].agent_properties.append(prop)

        if return_region_index: region_map.append(tuple([random_region_index,len(regions[random_region_index].agent_properties)-1]))

    return regions, region_map


def _consolidate_all_responses(
    all_responses: List[InitializeResponse],
    region_map: Optional[List[Tuple[int,int]]] = None,
    return_exact_agents: bool = False,
    get_infractions: bool = False
):
    if len(all_responses) > 0:
        # Get non-region-specific values such as api_model_version and traffic_light_states from an existing response
        response = deepcopy(all_responses[0]) 

        agent_states = []
        agent_properties = []
        recurrent_states = []
        infractions = []

        region_agent_keep_map = {i: [True]*len(res.agent_properties) for i, res in enumerate(all_responses)}

        if region_map is not None:
            for (region_id, agent_id) in region_map:
                try:
                    agent_states.append(all_responses[region_id].agent_states[agent_id])
                    agent_properties.append(all_responses[region_id].agent_properties[agent_id])
                    recurrent_states.append(all_responses[region_id].recurrent_states[agent_id])

                    if get_infractions:
                        infractions.append(all_responses[region_id].infractions[agent_id])

                    region_agent_keep_map[region_id][agent_id] = False
                except IndexError as e: 
                    exception_message = f"Warning: Unable to fetch specified agent ID {agent_id} in region {region_id}."
                    if not return_exact_agents: 
                        iai.logger.debug(exception_message)
                    else:
                        raise InvertedAIError(message=exception_message)
        
        for ind, response in enumerate(all_responses):
            response_agent_states = response.agent_states
            agent_states = agent_states + [state for i, state in enumerate(response_agent_states) if region_agent_keep_map[ind][i]]
            agent_properties = agent_properties + [prop for i, prop in enumerate(response.agent_properties[:len(response_agent_states)]) if region_agent_keep_map[ind][i]]
            recurrent_states = recurrent_states + [recurr for i, recurr in enumerate(response.recurrent_states) if region_agent_keep_map[ind][i]]
            if get_infractions:
                infractions = infractions + [infr for i, infr in enumerate(response.infractions) if region_agent_keep_map[ind][i]]
        
        response.infractions = infractions
        response.agent_states = agent_states
        response.agent_properties = agent_properties
        response.recurrent_states = recurrent_states 
    else:
        raise InvertedAIError(message=f"Unable to initialize any given region. Please check the input parameters.")

    return response


def _get_all_existing_agents_from_regions(
    regions: List[Region],
    exclude_index: Optional[int] = None,
    nearby_region: Optional[Region] = None,
):
    agent_states = []
    agent_properties = []

    for ind, region in enumerate(regions):
        if not ind == exclude_index:
            if nearby_region is not None:
                if sqrt((nearby_region.center.x-region.center.x)**2+(nearby_region.center.y-region.center.y)**2) > (REGION_MAX_SIZE + AGENT_SCOPE_FOV_BUFFER):
                    continue
            region_agent_states = region.agent_states
            agent_states = agent_states + region_agent_states
            agent_properties = agent_properties + [prop for prop in region.agent_properties[:len(region_agent_states)]]

    return agent_states, agent_properties


def _initialize_regions(
    location: str,
    regions: List[Region],
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None,
    display_progress_bar: bool = True,
    return_exact_agents: bool = False
) -> Tuple[List[Region],List[InitializeResponse]]:
    
    agent_states_sampled = []
    agent_properties_sampled = []
    agent_rs_sampled = []

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
        iterable_regions = enumerate(regions)

    num_attempts = 1 + len(regions) // ATTEMPT_PER_NUM_REGIONS
    all_responses = []
    for i, region in iterable_regions:
        region_center = region.center
        region_size = region.size

        existing_agent_states, existing_agent_properties = _get_all_existing_agents_from_regions(
            regions = regions,
            exclude_index = i,
            nearby_region = region
        )

        # Acquire agents that exist in other regions that must be passed as conditional to avoid collisions
        out_of_region_conditional_agents = list(filter(
            lambda x: inside_fov(center=region_center, agent_scope_fov=region_size+AGENT_SCOPE_FOV_BUFFER, point=x[0].center), 
            zip(existing_agent_states,existing_agent_properties)
        ))

        out_of_region_conditional_agent_states = [x[0] for x in out_of_region_conditional_agents]
        out_of_region_conditional_agent_properties = [x[1] for x in out_of_region_conditional_agents]

        region_conditional_agent_states = [] if region.agent_states is None else region.agent_states
        num_region_conditional_agents = len(region_conditional_agent_states)
        region_conditional_agent_properties = [] if region.agent_properties is None else region.agent_properties[:num_region_conditional_agents]
        region_unsampled_agent_properties = [] if region.agent_properties is None else region.agent_properties[num_region_conditional_agents:]
        all_agent_states = out_of_region_conditional_agent_states + region_conditional_agent_states
        all_agent_properties = out_of_region_conditional_agent_properties + region_conditional_agent_properties + region_unsampled_agent_properties

        num_out_of_region_conditional_agents = len(out_of_region_conditional_agent_states)

        regions[i].clear_agents()
        response = None
        if len(all_agent_properties) > 0:
            for attempt in range(num_attempts):
                try:
                    response = iai.initialize(
                        location=location,
                        states_history=None if len(all_agent_states) == 0 else [all_agent_states],
                        agent_properties=all_agent_properties,
                        get_infractions=get_infractions,
                        traffic_light_state_history=traffic_light_state_history,
                        location_of_interest=(region_center.x, region_center.y),
                        random_seed=random_seed
                    )

                except InvertedAIError as e:
                    # If error has occurred, display the warning and retry
                    iai.logger.debug(f"Region initialize attempt {attempt} error: {e}")
                    continue

                # Initialization of this region was successful, break the loop and proceed to the next region
                break
            
            else:
                exception_string = f"Unable to initialize region {i} at {region.center} with size {region.size} after {num_attempts} attempts."
                if return_exact_agents: 
                    raise InvertedAIError(message=exception_string)
                else:
                    iai.logger.debug(exception_string)
                    if num_region_conditional_agents > 0:
                    # Get the recurrent states for all predefined agents within the region
                        response = iai.initialize(
                            location=location,
                            states_history=[all_agent_states],
                            agent_properties=all_agent_properties[:num_out_of_region_conditional_agents+num_region_conditional_agents],
                            get_infractions=get_infractions,
                            traffic_light_state_history=traffic_light_state_history,
                            location_of_interest=(region_center.x, region_center.y),
                            random_seed=random_seed
                        )
            
            if response is not None:
                # Filter out conditional agents from other regions
                infractions = []
                for j, (state, props, r_state) in enumerate(zip(
                    response.agent_states[num_out_of_region_conditional_agents:],
                    response.agent_properties[num_out_of_region_conditional_agents:],
                    response.recurrent_states[num_out_of_region_conditional_agents:]
                )):
                    if not return_exact_agents:
                        if not inside_fov(center=region_center, agent_scope_fov=region_size, point=state.center):
                            continue

                    regions[i].insert_all_agent_details(state,props,r_state)
                    if get_infractions:
                        infractions.append(response.infractions[num_out_of_region_conditional_agents:][j])

                response.infractions = infractions
                response.agent_states = regions[i].agent_states
                response.agent_properties = regions[i].agent_properties
                response.recurrent_states = regions[i].recurrent_states
                all_responses.append(response)

                if traffic_light_state_history is None and response.traffic_lights_states is not None:
                    traffic_light_state_history = [response.traffic_lights_states]
        else:
            #There are no agents to initialize within this region, proceed to the next region
            continue

    return regions, all_responses


@validate_call
def large_initialize(
    location: str,
    regions: List[Region],
    agent_properties: Optional[List[AgentProperties]] = None,
    agent_states: Optional[List[AgentState]] = None,
    traffic_light_state_history: Optional[List[TrafficLightStatesDict]] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None,
    display_progress_bar: bool = True,
    return_exact_agents: bool = False
) -> InitializeResponse:
    """
    A utility function to initialize an area larger than 100x100m. This function takes in a 
    list of Region objects on each of which :func:`initialize` is run and each initialize 
    response is combined into a single response which is returned. While looping over all
    regions, if there are agents in other regions that are near enough to the region of
    interest, they will be passed as conditional to :func:`initialize`. :func:`initialize` 
    will not be called if no agent_states or agent_properties are specified in the region. 
    A boolean flag can be used to control failure behaviour if :func:`initialize` is unable 
    to produce viable vehicle placements if the initialization should continue or raise an 
    exception.

    As well, predefined agents may be passed to this function in 2 different ways. If the index
    of the predefined agents must be preserved, pass these agents' data into the agent_properties 
    and agent_states parameters. Each agent for which its states is defined MUST have its respective
    agent properties defined as well but an agent is permitted to be defined by its properties only
    and :func:`initialize` will fill in the state information. If the index of the predefined agents
    does not matter, they may be placed directly into the region objects or passed into the parameters
    mentioned previously, but make sure to avoid adding these agents twice.

    Arguments
    ----------
    location:
        Please refer to the documentation of :func:`initialize` for information on this parameter.

    regions:
        List of regions that contains information about the center, size, agent states and 
        agent properties formatted to align with :func:`initialize`. Please refer to the 
        documentation for :func:`initialize` for more details. 

    agent_properties:
        The properties of the agents that will have their indexes preserved within the response 
        output. Please refer to the documentation of :func:`initialize` for information on this 
        parameter.

    agent_states:
        One timestep worth of agent states that will have their indexes preserved within the response 
        output. Please refer to the documentation of :func:`initialize` for more information on this 
        parameter.
    
    traffic_light_state_history:
        Please refer to the documentation of :func:`initialize` for information on this parameter.

    get_infractions:
        Please refer to the documentation of :func:`initialize` for information on this parameter.
    
    random_seed:
        Please refer to the documentation of :func:`initialize` for information on this parameter.

    api_model_version:
        Please refer to the documentation of :func:`initialize` for information on this parameter.

    display_progress_bar:
        If True, a bar is displayed showing the progress of all relevant processes.

    return_exact_agents:
        If set to True, this function will raise an InvertedAIError exception if it cannot fit 
        the requested number of agents in any single region. If set to False, a region that 
        fails to return the number of requested agents will be skipped and only its predefined 
        agents (if any) will be returned with respective RecurrentState's. 
    
    See Also
    --------
    :func:`initialize`
    """

    if (agent_properties is not None and agent_states is not None) or (agent_properties is None and agent_states is not None):
        assert len(agent_properties) >= len(agent_states), "Invalid parameters: number of agent properties must be larger than number agent states."

    regions, region_map = _insert_agents_into_nearest_regions(
        regions = regions,
        agent_properties = [] if agent_properties is None else agent_properties,
        agent_states = [] if agent_states is None else agent_states,
        return_region_index = True,
        random_seed = random_seed
    )

    regions, all_responses = _initialize_regions(
        location = location,
        regions = regions,
        traffic_light_state_history = traffic_light_state_history,
        get_infractions = get_infractions,
        random_seed = random_seed,
        api_model_version = api_model_version,
        display_progress_bar = display_progress_bar,
        return_exact_agents = return_exact_agents
    )

    response = _consolidate_all_responses(
        all_responses = all_responses,
        region_map = region_map,
        return_exact_agents = return_exact_agents,
        get_infractions = get_infractions
    )
    
    return response