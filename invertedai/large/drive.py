import asyncio
import warnings
from typing import Tuple, Optional, List, Union
from pydantic import BaseModel, validate_call
from math import ceil

import invertedai as iai
from invertedai.large.common import Region
from invertedai.common import Point, AgentState, AgentAttributes, AgentProperties, RecurrentState, TrafficLightStatesDict, LightRecurrentState, LightRecurrentStates
from invertedai.api.drive import DriveResponse, serialize_drive_request_parameters
from invertedai.utils import convert_attributes_to_properties
from invertedai.error import InvertedAIError, InvalidRequestError
from invertedai.logs.debug_logger import DebugLogger
from ._quadtree import QuadTreeAgentInfo, QuadTree, _flatten_and_sort, QUADTREE_SIZE_BUFFER

DRIVE_MAXIMUM_NUM_AGENTS = 100

async def async_drive_all(async_input_params):
    all_responses = await asyncio.gather(*[iai.async_drive(**input_params) for input_params in async_input_params])
    return all_responses

@validate_call
def large_drive(
    location: str,
    agent_states: List[AgentState],
    agent_properties: List[Union[AgentAttributes,AgentProperties]],
    recurrent_states: Optional[List[RecurrentState]] = None,
    traffic_lights_states: Optional[TrafficLightStatesDict] = None,
    light_recurrent_states: Optional[List[LightRecurrentState]] = None,
    get_infractions: bool = False,
    random_seed: Optional[int] = None,
    api_model_version: Optional[str] = None,
    single_call_agent_limit: Optional[int] = None,
    async_api_calls: bool = True
) -> DriveResponse:
    """
    A utility function to drive more than the normal capacity of agents in a call to :func:`drive`.
    The agents are inserted into a quadtree structure and :func:`drive` is then called on each
    region represented by a leaf node of the quadtree. Agents near this region are included in the
    :func:`drive` calls to ensure the agents see all their neighbours. The quadtree is constructed 
    during each call to this utility function to maintain statelessness.

    Parameters
    ----------
    location:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    agent_states:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    agent_properties:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    recurrent_states:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    traffic_lights_states:
       Please refer to the documentation of :func:`drive` for information on this parameter.

    light_recurrent_states:
       Please refer to the documentation of :func:`drive` for information on this parameter.

    get_infractions:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    random_seed:
        Please refer to the documentation of :func:`drive` for information on this parameter.

    api_model_version:
        Please refer to the documentation of :func:`drive` for information on this parameter.
    
    single_call_agent_limit:
        The number of agents allowed in a region before it must subdivide. Currently this value 
        represents the capacity of a quadtree leaf node that will subdivide if the number of vehicles 
        in the region, plus relevant neighbouring regions, passes this threshold. In any case, this 
        will be limited to the maximum currently supported by :func:`drive`.

    async_api_calls:
        A flag to control whether to use asynchronous DRIVE calls.

    See Also
    --------
    :func:`drive`
    """

    # Validate input arguments
    if single_call_agent_limit is None:
        single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS
    if single_call_agent_limit > DRIVE_MAXIMUM_NUM_AGENTS:
        single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS
        iai.logger.warning(f"Single Call Agent Limit cannot be more than {DRIVE_MAXIMUM_NUM_AGENTS}, limiting this value to {DRIVE_MAXIMUM_NUM_AGENTS} and proceeding.")
    num_agents = len(agent_states)
    if not (num_agents == len(agent_properties)):
        if recurrent_states is not None and not (num_agents == len(recurrent_states)):
            raise InvalidRequestError(message="Input lists are not of equal size.")
    if not num_agents > 0:
        raise InvalidRequestError(message="Valid call must contain at least 1 agent.")

    # Convert any AgentAttributes to AgentProperties for backwards compatibility 
    agent_properties_new = []
    is_using_attributes = False
    for properties in agent_properties:
        properties_new = properties
        if isinstance(properties,AgentAttributes):
            properties_new = convert_attributes_to_properties(properties)
            is_using_attributes = True
        agent_properties_new.append(properties_new)
    agent_properties = agent_properties_new

    if is_using_attributes:
        warnings.warn('agent_attributes is deprecated. Please use agent_properties.',category=DeprecationWarning)

    is_debug_logging = DebugLogger.check_instance_exists(DebugLogger)
    if is_debug_logging:
        debug_logger = DebugLogger()
        debug_large_drive_parameters = serialize_drive_request_parameters(
            location = location,
            agent_states = agent_states,
            agent_attributes = None,
            agent_properties = agent_properties,
            recurrent_states = recurrent_states,
            traffic_lights_states = traffic_lights_states,
            light_recurrent_states = light_recurrent_states,
            get_birdview = False,
            rendering_center = None,
            rendering_fov = None,
            get_infractions = get_infractions,
            random_seed = random_seed,
            api_model_version = api_model_version
        )
        debug_logger.append_request(
            model = "large_drive",
            data_dict = debug_large_drive_parameters
        )

    # Generate quadtree
    agent_x = [agent.center.x for agent in agent_states]
    agent_y = [agent.center.y for agent in agent_states]
    max_x, min_x, max_y, min_y = max(agent_x), min(agent_x), max(agent_y), min(agent_y)
    region_size = ceil(max(max_x - min_x, max_y - min_y)) + QUADTREE_SIZE_BUFFER
    region_center = (round((max_x+min_x)/2),round((max_y+min_y)/2))

    quadtree = QuadTree(
        capacity=single_call_agent_limit,
        region=Region.create_square_region(
            center=Point.fromlist(list(region_center)),
            size=region_size
        ),
    )
    for i, (agent, attrs) in enumerate(zip(agent_states,agent_properties)):
        if recurrent_states is None:
            recurr_state = None
        else:
            recurr_state = recurrent_states[i]

        agent_info = QuadTreeAgentInfo.fromlist([agent, attrs, recurr_state, i])
        is_inserted = quadtree.insert(agent_info)

        if not is_inserted:
            raise InvertedAIError(message=f"Unable to insert agent into region.")

    
    # Call DRIVE API on all leaf nodes
    all_leaf_nodes = quadtree.get_leaf_nodes()
    async_input_params = []
    all_responses = []
    non_empty_nodes = []
    agent_id_order = []
    
    if len(all_leaf_nodes) > 1:
        for i, leaf_node in enumerate(all_leaf_nodes):
            region, region_buffer = leaf_node.region, leaf_node.region_buffer
            region_agents_ids = [particle.agent_id for particle in leaf_node.particles]

            if len(region.agent_states) > 0:
                non_empty_nodes.append(leaf_node)
                agent_id_order.extend(region_agents_ids)
                input_params = {
                    "location":location,
                    "agent_states":region.agent_states+region_buffer.agent_states,
                    "recurrent_states":None if recurrent_states is None else region.recurrent_states+region_buffer.recurrent_states,
                    "agent_properties":region.agent_properties+region_buffer.agent_properties,
                    "light_recurrent_states":light_recurrent_states,
                    "traffic_lights_states":traffic_lights_states,
                    "get_birdview":False,
                    "rendering_center":None,
                    "rendering_fov":None,
                    "get_infractions":get_infractions,
                    "random_seed":random_seed,
                    "api_model_version":api_model_version
                }
                if not async_api_calls:
                    all_responses.append(iai.drive(**input_params))
                else:
                    async_input_params.append(input_params)

        if async_api_calls:
            all_responses = asyncio.run(async_drive_all(async_input_params))

        response = DriveResponse(
            agent_states = _flatten_and_sort([region_response.agent_states[:leaf_node.get_number_of_agents_in_node()] for region_response, leaf_node in zip(all_responses,non_empty_nodes)],agent_id_order),
            recurrent_states = _flatten_and_sort([region_response.recurrent_states[:leaf_node.get_number_of_agents_in_node()] for region_response, leaf_node in zip(all_responses,non_empty_nodes)],agent_id_order),
            is_inside_supported_area = _flatten_and_sort([region_response.is_inside_supported_area[:leaf_node.get_number_of_agents_in_node()] for region_response, leaf_node in zip(all_responses,non_empty_nodes)],agent_id_order),
            infractions = [] if not get_infractions else _flatten_and_sort([region_response.infractions[:leaf_node.get_number_of_agents_in_node()] for region_response, leaf_node in zip(all_responses,non_empty_nodes)],agent_id_order),
            api_model_version = all_responses[0].api_model_version,
            birdview = None,
            traffic_lights_states = all_responses[0].traffic_lights_states,
            light_recurrent_states = all_responses[0].light_recurrent_states
        )

    else:
        # Quadtree capacity has not been surpassed therefore can just call regular drive()
        response = iai.drive(
            location = location,
            agent_states = agent_states,
            agent_properties = agent_properties,
            recurrent_states = recurrent_states,
            traffic_lights_states = traffic_lights_states,
            light_recurrent_states = light_recurrent_states,
            get_birdview = False,
            rendering_center = None,
            rendering_fov = None,
            get_infractions = get_infractions,
            random_seed = random_seed,
            api_model_version = api_model_version
        )

    if is_debug_logging:
        debug_logger.append_response(
            model = "large_drive",
            data_dict = response.serialize_drive_response_parameters()
        )

    return response