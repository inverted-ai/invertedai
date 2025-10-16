    #pragma once

    #include <vector>
    #include <map>
    #include <optional>
    #include <random>
    #include <stdexcept>
    #include "error.h"           
    #include "invertedai/api.h"             
    #include "invertedai/data_utils.h"      

    namespace invertedai {
/**
 * @brief Assign agents to regions proportionally to drivable area.
 *
 * Uses birdview images from the IAI API to estimate drivable area per region,
 * then distributes agents stochastically according to these weights.
 *
 * Steps:
 *  - Call `location_info` for each region to fetch its birdview.
 *  - Measure drivable ratio (non-black pixels).
 *  - Normalize weights across regions.
 *  - Sample regions for each agent using weighted random distribution.
 *
 * @param location Map/location identifier string.
 * @param regions Candidate simulation regions.
 * @param total_num_agents Optional total agent count (if dict not provided).
 * @param agent_count_dict Optional mapping of agent types to counts.
 * @param session Active IAI session used for birdview queries.
 * @param random_seed Optional seed for reproducibility.
 * @return std::vector<Region> Regions that have agents assigned.
 *
 * @throws InvertedAIError if neither total_num_agents nor agent_count_dict is given,
 *         or if no drivable area is detected.
 * @throws std::runtime_error if birdview decoding fails.
 */
std::vector<Region> get_number_of_agents_per_region_by_drivable_area(
    const std::string& location,
    const std::vector<Region>& regions,
    std::optional<int> total_num_agents,
    std::optional<std::map<AgentType,int>> agent_count_dict,
    Session& session,
    std::optional<int> random_seed
);

/**
 * @brief Build default simulation regions with agents distributed.
 *
 * Convenience wrapper for creating regions in a grid and assigning agents
 * proportionally to drivable area. If `agent_count_dict` is not provided,
 * defaults to spawning cars only.
 *
 * @param location Map/location identifier string.
 * @param total_num_agents Optional total number of agents (deprecated).
 * @param agent_count_dict Optional mapping of agent types to counts.
 * @param session Active IAI session used for map queries.
 * @param area_shape Optional grid area (width, height in meters).
 *                   Defaults to 100m × 100m (50, 50 half-size).
 * @param map_center Center of the region grid (world coordinates).
 * @param random_seed Optional seed for reproducibility.
 * @return std::vector<Region> Regions with agent properties assigned.
 *
 * @throws InvertedAIError if neither total_num_agents nor agent_count_dict is given.
 */
std::vector<Region> get_regions_default(
    const std::string& location,
    std::optional<int> total_num_agents,
    std::optional<std::map<AgentType,int>> agent_count_dict,
    Session& session,
    std::optional<std::pair<float,float>> area_shape = std::nullopt,
    std::pair<float,float> map_center = {0.0f, 0.0f},
    std::optional<int> random_seed = std::nullopt
);

/**
 * @brief Assign agents to regions proportionally to drivable road area.
 *
 * For each region, computes the ratio of drivable area (from birdview images
 * obtained via the LocationInfo API). Agents are then distributed across
 * regions with probability proportional to this ratio.
 *
 * - If `agent_count_dict` is provided, it determines how many agents of each
 *   type (car, pedestrian, etc.) will be assigned.
 * - If `total_num_agents` is provided instead, that many cars will be assigned
 *   by default.
 * - Only regions with at least one assigned agent are returned in the final list.
 *
 * @param location The location string for the map (IAI format).
 * @param regions Candidate regions to evaluate (typically from get_regions_in_grid()).
 * @param total_num_agents Total number of agents (if agent_count_dict is not given).
 * @param agent_count_dict Optional dictionary of {AgentType → count}.
 * @param session Active API session for making LocationInfo calls.
 * @param random_seed Optional random seed for reproducible sampling.
 *
 * @return std::vector<Region> Subset of regions, each with assigned AgentProperties.
 *
 * @throws InvertedAIError if neither `total_num_agents` nor `agent_count_dict` is provided,
 *         or if no drivable area is detected in any region.
 */
std::vector<Region> get_number_of_agents_per_region_by_drivable_area(
    const std::string& location,
    const std::vector<Region>& regions,
    std::optional<int> total_num_agents,
    std::optional<std::map<AgentType,int>> agent_count_dict,
    Session& session,
    std::optional<int> random_seed
);

/**
 * @brief Generate a uniform grid of square regions.
 *
 * Creates square regions that cover a rectangular area centered on `map_center`.
 * Each region has fixed stride (spacing) between centers.
 *
 * @param width Total width of the grid (meters).
 * @param height Total height of the grid (meters).
 * @param map_center Center of the grid in world coordinates.
 * @param stride Spacing between region centers (default = 100m).
 * @return std::vector<Region> List of generated square regions.
 */
std::vector<invertedai::Region> get_regions_in_grid(
    float width,
    float height,
    std::pair<float,float> map_center = {0.0f,0.0f},
    float stride = 100.0f
);

/**
 * @brief Create default AgentProperties for a given agent type.
 *
 * Initializes minimal agent properties with only the agent_type field set.
 * Other fields (length, width, speed, etc.) are left unspecified and can
 * be filled later (e.g., during initialization).
 *
 * @param type AgentType enum (car or pedestrian).
 * @return AgentProperties with default agent_type set.
 */
AgentProperties make_default_properties(AgentType type);

} // namespace invertedai
