#include <vector>
#include <string>
#include <optional>
#include <map>
#include "common.h"
#include "invertedai/api.h"



namespace invertedai {
using TrafficLightStatesDict = std::vector<TrafficLightState>;
struct LargeInitializeConfig {
    std::string location;
    std::vector<invertedai::Region> regions;

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

std::vector<std::map<std::string,std::string>>
convert_traffic_light_history(const std::vector<TrafficLightStatesDict>& dicts);

std::pair<std::vector<invertedai::Region>, std::vector<invertedai::InitializeResponse>> initialize_regions(
    const std::string& location,
    std::vector<invertedai::Region> regions,
    const std::optional<std::vector<TrafficLightStatesDict>>& traffic_light_state_history = std::nullopt,
    bool get_infractions = false,
    std::optional<int> random_seed = std::nullopt,
    std::optional<std::string> api_model_version = std::nullopt,
    bool display_progress_bar = true,
    bool return_exact_agents = false
);

InitializeResponse consolidate_all_responses(
    const std::vector<InitializeResponse>& all_responses,
    const std::optional<std::vector<std::pair<int,int>>>& region_map = std::nullopt,
    bool return_exact_agents = false,
    bool get_infractions = false
);

invertedai::InitializeResponse large_initialize(const invertedai::LargeInitializeConfig& cfg);
} // namespace invertedai
#ifndef LARGE_INITIALIZE_H
#define LARGE_INITIALIZE_H

// declarations...

#endif // LARGE_INITIALIZE_H
