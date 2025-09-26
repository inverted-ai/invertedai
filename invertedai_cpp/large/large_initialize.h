#include <vector>
#include <string>
#include <optional>
#include <map>
#include "common.h"
#include "invertedai/api.h"



namespace invertedai {
using TrafficLightStatesDict = std::vector<TrafficLightState>;

/**
 * @brief Output of large_initialize_with_regions function.
 *
 * Contains the consolidated InitializeResponse and the updated list of regions
 * with their initialized agents, useful for visualization and further analysis.
 */
struct LargeInitializeOutput {
    invertedai::InitializeResponse response;
    std::vector<invertedai::Region> regions; // updated with agent_states / props / recurrent_states // for testing the regions
};

/**
 * @brief Configuration options for large-scale agent initialization.
 *
 * This struct collects all inputs needed to call `large_initialize()` or
 * `large_initialize_with_regions()`. It provides control over the simulation
 * location, candidate regions, initial agent placement, and initialization
 * behavior.
 */
struct LargeInitializeConfig {
/**
     * @brief Location string (IAI format, e.g., "carla:Town10HD").
     */
    std::string location;

    /**
     * @brief Candidate regions for initialization.
     *
     * Typically generated via helper functions like `get_regions_default()` or
     * `get_regions_in_grid()`. Each region defines a 100m × 100m FOV with
     * center coordinates.
     */
    std::vector<invertedai::Region> regions;

    /**
     * @brief Active API session used for all requests.
     */
    Session& session;

    /**
     * @brief Optional agent properties to initialize.
     *
     * If provided, must be at least as long as `agent_states` (if states are
     * also provided). Properties without corresponding states are treated as
     * unsampled agents to be spawned by the API.
     */
    std::optional<std::vector<AgentProperties>> agent_properties = std::nullopt;

    /**
     * @brief Optional initial agent states.
     *
     * If provided, each entry must have a matching `AgentProperties` entry.
     * Defines explicit positions, headings, and speeds for seeded agents.
     */
    std::optional<std::vector<AgentState>> agent_states = std::nullopt;

    /**
     * @brief Optional traffic light state history.
     *
     * If provided, overrides default behavior and sets initial light states.
     */
    std::optional<std::map<std::string, std::string>> traffic_light_state_history = std::nullopt;

    /**
     * @brief Whether to request infraction indicators for initialized agents.
     */
    bool get_infractions = false;

    /**
     * @brief Optional random seed for reproducibility.
     */
    std::optional<int> random_seed = std::nullopt;

    /**
     * @brief Optional API model version to target.
     *
     * Leave unset to use the default backend version.
     */
    std::optional<std::string> api_model_version = std::nullopt;

    /**
     * @brief Whether to return exactly the initialized agents (true) or
     * restrict results to those inside the region FOV (false).
     */
    bool return_exact_agents = false;

    /**
     * @brief Construct with a required API session reference.
     */
    explicit LargeInitializeConfig(Session& sess) : session(sess) {}
};

/**
 * @brief Extended initialize that also returns final region layouts.
 *
 * Similar to `large_initialize`, but also returns the updated list of
 * regions with their initialized agents, useful for visualization.
 *
 * @param cfg LargeInitializeConfig configuration (regions, agents, options).
 * @return LargeInitializeOutput {consolidated response, final regions}.
 *
 * @throws std::invalid_argument if inputs are inconsistent.
 */
LargeInitializeOutput large_initialize_with_regions(invertedai::LargeInitializeConfig& cfg); // for testing the regions

/**
 * @brief Assign agents into the nearest simulation regions.
 *
 * Agents with predefined states are inserted into the region closest
 * to their (x, y) position. Remaining properties without states are
 * randomly assigned to regions.
 *
 * @param regions List of simulation regions.
 * @param agent_properties Properties of all agents.
 * @param agent_states States of agents with explicit positions.
 * @param return_region_index If true, also return (region_idx, agent_idx) map.
 * @param random_seed Optional random seed for reproducible placement.
 * @return std::pair<std::vector<Region>, RegionMap>
 *         Updated regions and optional mapping of agent indices.
 *
 * @throws std::invalid_argument if regions are empty or properties < states.
 */
std::pair<std::vector<Region>, std::vector<std::pair<int, int>>> insert_agents_into_nearest_regions(
    std::vector<Region> regions,
    const std::vector<AgentProperties>& agent_properties,
    const std::vector<AgentState>& agent_states,
    bool return_region_index = false,
    std::optional<int> random_seed = std::nullopt
);

std::vector<std::map<std::string,std::string>>
convert_traffic_light_history(const std::vector<TrafficLightStatesDict>& dicts);

/**
 * @brief Run initialization on a set of regions with agents.
 *
 * Iterates over each region, collects nearby context, prepares agent
 * states/properties, and calls the IAI `initialize` API. Results are
 * filtered to keep only valid agents in the region’s FOV.
 *
 * @param location Location string (map name or domain).
 * @param regions Regions to initialize (modified in-place).
 * @param session Active IAI API session.
 * @param traffic_light_state_history Optional traffic light states.
 * @param get_infractions Whether to collect infraction indicators.
 * @param random_seed Optional random seed for reproducibility.
 * @param api_model_version Optional API model version to request.
 * @param return_exact_agents If false, discards agents outside FOV.
 * @return std::pair<std::vector<Region>, std::vector<InitializeResponse>>
 *         Updated regions and raw initialization responses.
 *
 * @throws InvertedAIError if region initialization fails and exact agents requested.
 */
std::pair<std::vector<invertedai::Region>, std::vector<invertedai::InitializeResponse>> initialize_regions(
    const std::string& location,
    std::vector<invertedai::Region> regions,
    Session& session,
    std::optional<std::map<std::string, std::string>>& traffic_light_state_history,
    bool get_infractions = false,
    std::optional<int> random_seed = std::nullopt,
    std::optional<std::string> api_model_version = std::nullopt,
    bool return_exact_agents = false
);

/**
 * @brief Merge multiple initialization responses into one.
 *
 * Combines agent states, properties, recurrent states, and infractions
 * from multiple regions. Optionally filters agents based on a provided
 * region-agent mapping.
 *
 * @param all_responses Responses from multiple regions.
 * @param region_map Optional explicit region-agent index mapping.
 * @param return_exact_agents Whether to enforce strict agent matching.
 * @param get_infractions Whether to collect infraction indicators.
 * @return InitializeResponse Merged initialization response.
 *
 * @throws InvertedAIError if responses are empty or region_map is invalid.
 */
InitializeResponse consolidate_all_responses(
    const std::vector<InitializeResponse>& all_responses,
    const std::optional<std::vector<std::pair<int,int>>>& region_map = std::nullopt,
    bool return_exact_agents = false,
    bool get_infractions = false
);

/**
 * @brief High-level entry point for multi-region initialization.
 *
 * Inserts agents into regions, runs initialization on each region,
 * then consolidates results into a single InitializeResponse.
 *
 * @param cfg LargeInitializeConfig configuration (regions, agents, options).
 * @return InitializeResponse Final merged response.
 *
 * @throws std::invalid_argument if inputs are inconsistent.
 */
invertedai::InitializeResponse large_initialize(invertedai::LargeInitializeConfig& cfg);
} // namespace invertedai
#ifndef LARGE_INITIALIZE_H
#define LARGE_INITIALIZE_H

// declarations...

#endif // LARGE_INITIALIZE_H
