#include "logger.h"
#include "externals/json.hpp"

#include <string>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

namespace invertedai {
    ScenarioLog::ScenarioLog(
        std::string location_,
        std::vector<std::vector<AgentState>> agent_states_,
        std::vector<AgentProperties> agent_properties_,
        std::optional<std::vector<std::map<std::string, std::string>>> traffic_states_,
        std::optional<std::pair<double,double>> rendering_center_,
        std::optional<int> rendering_fov_,

        std::optional<int> lights_seed_,
        std::optional<int> init_seed_,
        std::optional<int> drive_seed_,

        std::optional<std::string> init_version_,
        std::optional<std::string> drive_version_,

        std::optional<std::vector<LightRecurrentState>> light_states_,
        std::optional<std::vector<RecurrentState>> recurrent_states_,
        std::optional<std::map<std::string, std::vector<Point2d>>> waypoints_,
        std::vector<std::vector<int>> present_indexes_
    ) {
        location = location_;
        agent_states = agent_states_;
        agent_properties = agent_properties_;
        traffic_lights_states = traffic_states_;

        rendering_center = rendering_center_;
        rendering_fov = rendering_fov_;
    
        lights_random_seed = lights_seed_;
        initialize_random_seed = init_seed_;
        drive_random_seed = drive_seed_;
    
        initialize_model_version = init_version_;
        drive_model_version = drive_version_;
    
        light_recurrent_states = light_states_;
        recurrent_states = recurrent_states_;
    
        waypoints = waypoints_;
        present_indexes = present_indexes_;

        validate_states_and_present_indexes_init();
    }
    void ScenarioLog::validate_states_and_present_indexes_init() {
        if (present_indexes.empty())
            return;
    
        if (present_indexes.size() != agent_states.size()) {
            throw std::runtime_error("Given different number of time steps for agent states and present indexes.");
        }
    
        for (size_t t = 0; t < agent_states.size(); ++t) {
            validate_states_and_present_indexes_time_step(
                agent_states[t],
                present_indexes[t]
            );
        }
    }
    void ScenarioLog::validate_states_and_present_indexes_time_step(
        const std::vector<AgentState>& states,
        const std::vector<int>& present
    ) {
        if (states.size() != present.size()) {
            throw std::runtime_error("Given number of agent states does not match number of present agents.");
        }
        for (int idx : present) {
            if (idx < 0)
                throw std::runtime_error("Invalid agent ID's in given list of present indexes.");
        }
    }
    void ScenarioLog::add_time_step_data(
        std::vector<AgentState> current_states,
        std::vector<int> current_present
    ) {
        validate_states_and_present_indexes_time_step(current_states, current_present);
        agent_states.push_back(current_states);
        present_indexes.push_back(current_present);
    }

    template<typename ValueType>
    std::vector<std::pair<std::string, ValueType>>
    sort_dict(
        const std::map<std::string, ValueType>& dict,
        const std::map<std::string, int>& id_map
    ) {
        std::vector<std::pair<std::string, ValueType>> out;
        out.reserve(dict.size());
        for (const auto& kv : dict) {
            out.emplace_back(std::string(kv.first), kv.second);
        }
        std::sort(out.begin(), out.end(),
                  [&](const auto& a, const auto& b) {
                      return id_map.at(a.first) < id_map.at(b.first);
                  });
        return out;
    }

    ScenarioLogReader::ScenarioLogReader(const std::string &file_path) { 
        std::string json_body = invertedai::read_file(file_path.c_str());

        json j = json::parse(json_body);

        std::string location = j["location"]["identifier"];

        simulation_length = j["scenario_length"];

        std::vector<std::map<std::string, AgentState>> all_agent_states_unsorted;
        std::map<std::string, AgentProperties> all_agent_properties_unsorted;

        std::vector<std::vector<int>> present_indexes_unsorted;

        std::map<std::string, int> agent_id_list;
        int agent_id_sequence_num = 0;

        for(int t = 0; t < simulation_length; t++) {
            std::map<std::string, AgentState> agent_states_ts;
            std::vector<int> present_indexes_ts;

            for (auto& kv : j["predetermined_agents"].items()) {
                std::string agent_id = kv.key();
                const json& agent = kv.value();
                if (agent_id_list.count(agent_id) == 0) {
                    const json& attr = agent["static_attributes"];
                    AgentProperties props;
                    props.length = attr["length"].get<double>();
                    props.width = attr["width"].get<double>();
                    props.rear_axis_offset = attr["rear_axis_offset"].get<double>();
                    props.agent_type = agent["entity_type"].get<std::string>();
                    all_agent_properties_unsorted[agent_id] = props;
                    agent_id_list[agent_id] = agent_id_sequence_num++;
                }
                std::string ts_key = std::to_string(t);
                if (agent["states"].contains(ts_key)) {
                    present_indexes_ts.push_back(agent_id_list[agent_id]);
                    const json& st = agent["states"][ts_key];
                    AgentState state;
                    state.x = st["center"]["x"].get<double>();
                    state.y = st["center"]["y"].get<double>();
                    state.orientation = st["orientation"].get<double>();
                    state.speed = st["speed"].get<double>();
                    agent_states_ts[agent_id] = state;
                }
            }
            all_agent_states_unsorted.push_back(agent_states_ts);
            present_indexes_unsorted.push_back(present_indexes_ts);
        }

        auto properties_sorted = sort_dict(all_agent_properties_unsorted, agent_id_list);
        std::vector<AgentProperties> sorted_agent_properties;
        for (auto& kv : properties_sorted) {
            sorted_agent_properties.push_back(kv.second);
        }
        
        std::vector<std::vector<AgentState>> agent_states_over_time;
        for (auto& states_map : all_agent_states_unsorted) {
            auto sorted_vec = sort_dict(states_map, agent_id_list);
            std::vector<AgentState> states_only;
            states_only.reserve(sorted_vec.size());

            for (auto& kv : sorted_vec) {
                states_only.push_back(kv.second);
            }
            agent_states_over_time.push_back(states_only);
        }
        
        std::optional<std::vector<std::map<std::string, std::string>>> traffic_light_states_over_time;
        std::vector<std::vector<int>> present_indexes_sorted;
        for (auto& vec : present_indexes_unsorted) {
            std::sort(vec.begin(), vec.end());
            present_indexes_sorted.push_back(vec);
        }

        if (j.contains("predetermined_controls")) {
            std::vector<std::map<std::string,std::string>> tl_history(simulation_length);
            tl_history.resize(simulation_length);

            for (int t = 0; t < simulation_length; t++) {
                std::string ts_key = std::to_string(t);
        
                for (auto& kv : j["predetermined_controls"].items()) {
                    const std::string actor_id = kv.key();
                    const json& actor = kv.value();
        
                    if (actor["entity_type"] == "traffic_light" &&
                        actor["states"].contains(ts_key)) {
                        tl_history[t][actor_id] =
                            actor["states"][ts_key]["control_state"].get<std::string>();
                    }
                }
            }
            traffic_light_states_over_time = tl_history;
        }
        std::map<std::string, std::vector<Point2d>> agent_waypoints;
        if (j.contains("individual_suggestions")) { 
            std::map<std::string,std::vector<Point2d>> wp;
            for (auto& kv : j["individual_suggestions"].items()) {
                std::string ag = kv.key();

                for (auto& st : kv.value()["states"]) {
                    const json& c = st["center"];
                    wp[ag].push_back(Point2d{c["x"], c["y"]});
                }
            }
            if (!wp.empty())
            agent_waypoints = wp;
        }

        std::optional<std::vector<LightRecurrentState>> light_rs = std::nullopt;
        if (j.contains("light_recurrent_states") &&
            j["light_recurrent_states"].is_array() &&
            j["light_recurrent_states"].size() > 0)
        {
            std::vector<LightRecurrentState> lights;        
            for (const auto& arr : j["light_recurrent_states"]) {
                if (!arr.is_array() || arr.size() != 2 ||
                    !arr[0].is_number() || !arr[1].is_number()) {
                    throw std::runtime_error("Invalid entry in light_recurrent_states");
                }
                LightRecurrentState lrs;
                lrs.state = arr[0].get<int>();
                lrs.time_remaining = arr[1].get<int>();
                lights.push_back(lrs);
            }
            light_rs = lights;
        }

        std::optional<std::vector<RecurrentState>> rnn_states = std::nullopt;
        auto lights_seed =
            j.contains("lights_random_seed")
            ? std::optional<int>(j["lights_random_seed"])
            : std::nullopt;
        auto init_seed =
            j.contains("initialize_random_seed")
            ? std::optional<int>(j["initialize_random_seed"])
            : std::nullopt;           
        auto drive_seed =
            j.contains("drive_random_seed")
            ? std::optional<int>(j["drive_random_seed"])
            : std::nullopt;    
        auto init_version =
            j.contains("initialize_model_version")
            ? std::optional<std::string>(j["initialize_model_version"])
            : std::nullopt;  
        auto drive_version =
            j.contains("drive_model_version")
            ? std::optional<std::string>(j["drive_model_version"])
            : std::nullopt;
        scenario_log_ = ScenarioLog(
            location,
            agent_states_over_time,
            sorted_agent_properties,
            traffic_light_states_over_time.has_value() ? traffic_light_states_over_time : std::nullopt,
            std::optional<std::pair<double,double>>({
                j["birdview_options"]["rendering_center"][0],
                j["birdview_options"]["rendering_center"][1]
            }),
            j["birdview_options"]["renderingFOV"].get<int>(),
            lights_seed,
            init_seed,
            drive_seed,
            init_version,
            drive_version,
            light_rs,
            rnn_states,
            agent_waypoints,
            present_indexes_unsorted
        );
        reset_log();
    }
    
    const std::string& ScenarioLogReader::get_location() const {
        return this->scenario_log_.location;
    }
    
    std::optional<int> ScenarioLogReader::get_fov() {
        return this->scenario_log_.rendering_fov;
    }
    std::optional<std::pair<double, double>> ScenarioLogReader::get_rendering_center() {
        return this->scenario_log_.rendering_center;
    }
    ScenarioLog ScenarioLogReader::get_scenario_log() {
        return this->scenario_log_;
    }

    bool ScenarioLogReader::return_state_at_timestep(int t) {
        if (t < 0 || t >= simulation_length) {
            return false;
        }
        current_timestep = t;
        return true;
    }
    
    bool ScenarioLogReader::return_last_state() {
        if (simulation_length == 0) {
            return false;
        }
        current_timestep = simulation_length - 1;
        return true;
    }

    bool ScenarioLogReader::initialize() {
        bool init_response = return_state_at_timestep(0);
        current_timestep = 1;
        return init_response;
    }

    bool ScenarioLogReader::drive() {
        if (current_timestep + 1 >= simulation_length) {
            return false;
        }
        ++current_timestep;
        return true;
    }

    void ScenarioLogReader::reset_log() {
        current_timestep = 0;
    }

    std::vector<AgentState>
    ScenarioLogReader::current_agent_states() const {
        return scenario_log_.agent_states[current_timestep];
    }


    std::vector<AgentProperties>
    ScenarioLogReader::current_agent_properties() const {
        std::vector<AgentProperties> result;
        for (int idx : scenario_log_.present_indexes[current_timestep]) {
            result.push_back(scenario_log_.agent_properties[idx]);
        }
        return result;
    }

    std::optional<std::map<std::string, std::string>>
    ScenarioLogReader::current_traffic_lights() const {
        if (!scenario_log_.traffic_lights_states) {
            return std::nullopt;
        }
        return (*scenario_log_.traffic_lights_states)[current_timestep];
    }

    int ScenarioLogReader::get_scenario_length() {
        return this->scenario_log_.agent_states.size();
    }

    std::vector<AgentProperties> ScenarioLogReader::get_agent_properties() {
        return this->scenario_log_.agent_properties;
    }

    std::vector<std::vector<AgentState>> ScenarioLogReader::get_agent_states_over_time() {
        return scenario_log_.agent_states;
    }

    std::optional<std::vector<std::map<std::string, std::string>>>
    ScenarioLogReader::get_traffic_lights_states_over_time() {
        return this->scenario_log_.traffic_lights_states;
    }

    std::optional<std::vector<LightRecurrentState>> ScenarioLogReader::current_light_recurrent_state() const {
        return this->scenario_log_.light_recurrent_states;
    }
    
    std::optional<std::vector<RecurrentState>> ScenarioLogReader::current_recurrent_states() const {
        return this->scenario_log_.recurrent_states;
    }
}