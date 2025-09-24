#include <vector>
#include <string>
#include <optional>
#include <map>
#include "common.h"
#include "invertedai/api.h"



namespace invertedai {
using TrafficLightStatesDict = std::vector<TrafficLightState>;
struct LargeInitializeOutput {
    invertedai::InitializeResponse response;
    std::vector<invertedai::Region> regions; // updated with agent_states / props / recurrent_states // for testing the regions
};


struct LargeInitializeConfig {
    std::string location;
    std::vector<invertedai::Region> regions;
    Session& session;

    std::optional<std::vector<AgentProperties>> agent_properties = std::nullopt;
    std::optional<std::vector<AgentState>> agent_states = std::nullopt;
    std::optional<std::map<std::string, std::string>> traffic_light_state_history = std::nullopt;

    bool get_infractions = false;
    std::optional<int> random_seed = std::nullopt;
    std::optional<std::string> api_model_version = std::nullopt;
    bool return_exact_agents = false;

    LargeInitializeConfig(Session& sess): 
        session(sess) {}
};

LargeInitializeOutput large_initialize_with_regions(invertedai::LargeInitializeConfig& cfg); // for testing the regions

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
    Session& session,
    std::optional<std::map<std::string, std::string>>& traffic_light_state_history,
    bool get_infractions = false,
    std::optional<int> random_seed = std::nullopt,
    std::optional<std::string> api_model_version = std::nullopt,
    bool return_exact_agents = false
);

InitializeResponse consolidate_all_responses(
    const std::vector<InitializeResponse>& all_responses,
    const std::optional<std::vector<std::pair<int,int>>>& region_map = std::nullopt,
    bool return_exact_agents = false,
    bool get_infractions = false
);

invertedai::InitializeResponse large_initialize(invertedai::LargeInitializeConfig& cfg);
} // namespace invertedai
#ifndef LARGE_INITIALIZE_H
#define LARGE_INITIALIZE_H

// declarations...

#endif // LARGE_INITIALIZE_H
