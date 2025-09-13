#include <vector>
#include <string>
#include <optional>
#include <map>
#include "data_utils.h"
#include "common.h"



namespace invertedai {
using TrafficLightStatesDict = std::vector<TrafficLightState>;
struct LargeInitializeConfig {
    std::string location;
    std::vector<Region> regions;

    std::optional<std::vector<AgentProperties>> agent_properties = std::nullopt;
    std::optional<std::vector<AgentState>> agent_states = std::nullopt;
    std::optional<std::vector<TrafficLightStatesDict>> traffic_light_state_history = std::nullopt;

    bool get_infractions = false;
    std::optional<int> random_seed = std::nullopt;
    std::optional<std::string> api_model_version = std::nullopt;
    bool display_progress_bar = true;
    bool return_exact_agents = false;
};

std::pair<std::vector<Region>, std::vector<std::pair<int, int>>> insert_agents_into_nearest_regions(
    std::vector<Region> regions,
    const std::vector<AgentProperties>& agent_properties,
    const std::vector<AgentState>& agent_states,
    bool return_region_index = false,
    std::optional<int> random_seed = std::nullopt
);
//invertedai::InitializeResponse large_initialize(const LargeInitializeConfig& cfg);
} // namespace invertedai