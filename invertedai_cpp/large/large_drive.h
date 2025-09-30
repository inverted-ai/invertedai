#pragma once
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"
#include "quadtree.h"
#include "common.h"
#include "invertedai/api.h"
#include "invertedai/error.h"

#include <vector>
#include <optional>
#include <map>
#include <cmath>
#include <stdexcept>

namespace invertedai {

constexpr int DRIVE_MAXIMUM_NUM_AGENTS = 100;



struct LargeDriveConfig {
    LogWriter logger;
    std::string location;
    std::vector<AgentState> agent_states;
    std::vector<AgentProperties> agent_properties;
    std::optional<std::vector<std::vector<double>>> recurrent_states = std::nullopt;
    std::optional<std::map<std::string, std::string>> traffic_lights_states = std::nullopt;
    std::optional<std::vector<LightRecurrentState>> light_recurrent_states = std::nullopt;

    bool get_infractions = false;
    std::optional<int> random_seed = std::nullopt;
    std::optional<std::string> api_model_version = std::nullopt;
    int single_call_agent_limit = DRIVE_MAXIMUM_NUM_AGENTS;

    Session& session;

    LargeDriveConfig(Session& sess) : session(sess) {}
};

// Ported function
DriveResponse large_drive(LargeDriveConfig& cfg);

} // namespace invertedai