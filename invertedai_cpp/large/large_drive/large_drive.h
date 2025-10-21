#pragma once
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"
#include "quadtree.h"
#include "invertedai/api.h"
#include "invertedai/error.h"

#include <vector>
#include <optional>
#include <map>
#include <cmath>
#include <stdexcept>

namespace invertedai {

/**
 * @brief Maximum number of agents allowed per single DRIVE API call.
 *
 * This represents the upper bound for the number of agents that can be simulated
 * in a single region before subdivision occurs. The value is capped to prevent
 * overly large requests to the API.
 */
constexpr int DRIVE_MAXIMUM_NUM_AGENTS = 100;
/**
 * @brief Structure representing a single leaf task in the quadtree.
 *
 * Each leaf corresponds to a simulation region containing a subset of agents.
 * A task consists of the leaf index and its associated DRIVE request.
 */
struct LeafTask {
    size_t idx;        ///< Index of the corresponding leaf region.
    DriveRequest req;  ///< DRIVE API request containing agent states and properties.
};
/**
 * @brief Configuration for a large-scale DRIVE simulation.
 *
 * This configuration object contains all the necessary parameters and options
 * to perform a multi-region (quadtree-based) DRIVE simulation through
 * `large_drive()` or `large_drive_with_regions()`.
 *
 * The parameters correspond directly to the Python SDK’s `iai.drive()` API,
 * extended for large-scale batching and asynchronous execution.
 */
struct LargeDriveConfig {
    /**
     * @brief Logger for optional API request/response tracking.
     */
    LogWriter logger;

    /**
     * @brief Location name in IAI format (e.g., "carla:Town03").
     */
    std::string location;

    /**
     * @brief Current agent states for all simulated entities.
     *
     * Each state must include:
     * - `x`, `y`: coordinates in meters,
     * - `orientation`: heading in radians (0 = +x, π/2 = +y),
     * - `speed`: linear velocity in m/s.
     */
    std::vector<AgentState> agent_states;

    /**
     * @brief Static properties for all agents.
     *
     * Each property must define:
     * - `length`, `width`, `rear_axis_offset`: physical dimensions in meters,
     * - `agent_type`: either `"car"` or `"pedestrian"`,
     * - optional `waypoint` and `max_speed` fields.
     */
    std::vector<AgentProperties> agent_properties;

    /**
     * @brief Active API key used for authentication during asynchronous DRIVE calls.
     */
    std::string api_key;

    /**
     * @brief Optional recurrent states from a previous DRIVE or INITIALIZE step.
     *
     * Each vector corresponds to the recurrent memory for one agent.
     * Must be the same length as `agent_states` when provided.
     */
    std::optional<std::vector<std::vector<double>>> recurrent_states = std::nullopt;

    /**
     * @brief Optional map of current traffic light states.
     *
     * If set, these values override any states derived from
     * `light_recurrent_states`. Must be provided consistently
     * across simulation steps when used.
     */
    std::optional<std::map<std::string, std::string>> traffic_lights_states = std::nullopt;

    /**
     * @brief Optional recurrent traffic light states from the previous timestep.
     *
     * Pass this to maintain consistent light sequences across DRIVE calls.
     * If both this and `traffic_lights_states` are set, the explicit traffic
     * light states take priority.
     */
    std::optional<std::vector<LightRecurrentState>> light_recurrent_states = std::nullopt;

    /**
     * @brief Whether to request infraction indicators from the API.
     *
     * When true, DRIVE will return potential collision or violation flags.
     */
    bool get_infractions = false;

    /**
     * @brief Optional random seed for reproducible stochastic agent behavior.
     */
    std::optional<int> random_seed = std::nullopt;

    /**
     * @brief Optional model version to request from the API.
     *
     * If unset, the backend will automatically select the best available model.
     */
    std::optional<std::string> api_model_version = std::nullopt;

    /**
     * @brief Maximum number of agents per DRIVE call before region subdivision.
     *
     * Defines the maximum capacity of a quadtree leaf node. If a leaf exceeds
     * this limit, it subdivides to maintain per-region constraints. Automatically
     * capped at `DRIVE_MAXIMUM_NUM_AGENTS`.
     */
    int single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS;

    /**
     * @brief Whether to perform DRIVE calls asynchronously.
     *
     * If true, each quadtree leaf node performs its DRIVE call asynchronously.
     * If false, calls are made synchronously.
     */
    bool async_api_calls = true;

    /**
     * @brief Reference to an active IAI session for API communication.
     */
    Session& session;

    /**
     * @brief Construct a configuration with a required Session reference.
     */
    explicit LargeDriveConfig(Session& sess) : session(sess) {}
};

/**
 * @brief Output structure combining DRIVE results with quadtree leaf regions.
 *
 * Used by `large_drive_with_regions()` to return both the merged DRIVE response
 * and the set of quadtree leaf regions used during simulation.
 * 
 * For visualization, the 'regions' can be used to dray quadtree layouts.
 */
struct LargeDriveWithRegions {
    invertedai::DriveResponse response;  ///merged DRIVE response across all regions.
    std::vector<invertedai::Region> regions; ///quadtree leaf regions used for partitioning.
};

/**
 * @brief Perform a large-scale DRIVE simulation with automatic region subdivision.
 *
 * Executes the DRIVE API over multiple quadtree regions when the total number
 * of agents exceeds the per-call limit. Each region’s agents are simulated
 * independently, and the results are merged into a unified `DriveResponse`.
 *
 * @param cfg LargeDriveConfig configuration containing all simulation parameters.
 * @return DriveResponse Combined DRIVE response containing agent states,
 * recurrent states, infractions, and traffic light data.
 *
 * @throws InvertedAIError If agent vectors are mismatched or API calls fail.
 *
 * @note When `async_api_calls` is true, all DRIVE requests are executed in
 * parallel using separate sessions. When false, requests are executed
 * sequentially.
 */
invertedai::DriveResponse large_drive(invertedai::LargeDriveConfig& cfg);

/**
 * @brief Run a large DRIVE simulation and also return the region layout.
 *
 * Same as `large_drive()`, but also returns the quadtree’s final leaf regions
 * used for agent subdivision. Useful for visualization and debugging.
 *
 * @param cfg LargeDriveConfig configuration for the simulation.
 * @return LargeDriveWithRegions Structure containing the DRIVE response and
 * the quadtree leaf regions used during subdivision.
 *
 * @throws InvertedAIError If input data are invalid or DRIVE calls fail.
 */
invertedai::LargeDriveWithRegions large_drive_with_regions(invertedai::LargeDriveConfig& cfg);


} // namespace invertedai