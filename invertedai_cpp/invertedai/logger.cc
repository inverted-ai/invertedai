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
        std::optional<float> rendering_fov_,

        std::optional<int> lights_seed_,
        std::optional<int> init_seed_,
        std::optional<int> drive_seed_,

        std::optional<std::string> init_version_,
        std::optional<std::string> drive_version_,

        std::optional<std::vector<LightRecurrentState>> light_states_,
        std::optional<std::vector<std::vector<double>>> recurrent_states_,
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
                lrs.state = arr[0].get<float>();
                lrs.time_remaining = arr[1].get<float>();
                lights.push_back(lrs);
            }
            light_rs = lights;
        }

        std::optional<std::vector<std::vector<double>>> rnn_states = std::nullopt;
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
    
    std::optional<std::vector<std::vector<double>>> ScenarioLogReader::current_recurrent_states() const {
        return this->scenario_log_.recurrent_states;
    }

    ScenarioLogWriter::ScenarioLogWriter() {}

    std::pair<int, int> ScenarioLogWriter::count_agent_types() const {
        int num_cars = 0;
        int num_pedestrians = 0;
        
        for (const auto& prop : scenario_log_.agent_properties) {
            if (prop.agent_type.has_value()) {
                if (prop.agent_type.value() == "car") {
                    num_cars++;
                } else if (prop.agent_type.value() == "pedestrian") {
                    num_pedestrians++;
                }
            }
        }
        
        return {num_cars, num_pedestrians};
    }
    

    std::tuple<int, int, int, int> ScenarioLogWriter::count_control_types(
        const LocationInfoResponse& location_info_response
    ) const {
        int num_lights = 0;
        int num_yield = 0;
        int num_stop = 0;
        int num_other = 0;
        
        auto static_actors = location_info_response.static_actors();
        
        for (const auto& actor : static_actors) {
            if (actor.agent_type == "traffic_light") {
                num_lights++;
            } else if (actor.agent_type == "yield_sign") {
                num_yield++;
            } else if (actor.agent_type == "stop_sign") {
                num_stop++;
            } else {
                num_other++;
            }
        }
        
        return {num_lights, num_yield, num_stop, num_other};
    }

    
    json ScenarioLogWriter::build_individual_suggestions_dict(const ScenarioLog& log) const {
        json suggestions = json::object();
        if (!log.waypoints.has_value()) {
            for (size_t i = 0; i < log.agent_properties.size(); i++) {
                const auto& prop = log.agent_properties[i];
                if (prop.waypoint.has_value()) {
                    const auto& wp = prop.waypoint.value();
                    suggestions[std::to_string(i)] = {
                        {"suggestion_strength", 0.8},
                        {"states", json::array({
                            {
                                {"center", {
                                    {"x", wp.x},
                                    {"y", wp.y}
                                }}
                            }
                        })}
                    };
                }
            }
        } else {
            for (const auto& [agent_id, wps] : log.waypoints.value()) {
                json states_array = json::array();
                for (const auto& wp : wps) {
                    states_array.push_back({
                        {"center", {
                            {"x", wp.x},
                            {"y", wp.y}
                        }}
                    });
                }
                
                suggestions[agent_id] = {
                    {"suggestion_strength", 0.8},
                    {"states", states_array}
                };
            }
        }
        
        return suggestions;
    }
    
    json ScenarioLogWriter::build_predetermined_controls_dict(
        const ScenarioLog& log,
        const LocationInfoResponse& location_info_response
    ) const {
        json controls_dict = json::object();
        
        if (!log.traffic_lights_states.has_value()) {
            return controls_dict;
        }
        
        auto static_actors = location_info_response.static_actors();
        for (const auto& actor : static_actors) {
            if (actor.agent_type == "traffic_light") {
                json states_dict = json::object();
                
                for (size_t t = 0; t < log.traffic_lights_states.value().size(); t++) {
                    const auto& tls = log.traffic_lights_states.value()[t];
                    auto state_it = tls.find(std::to_string(actor.actor_id));
                    
                    if (state_it != tls.end()) {
                        states_dict[std::to_string(t)] = {
                            {"center", {
                                {"x", actor.x},
                                {"y", actor.y}
                            }},
                            {"orientation", actor.orientation},
                            {"speed", 0},
                            {"control_state", state_it->second}
                        };
                    }
                }
                
                controls_dict[std::to_string(actor.actor_id)] = {
                    {"entity_type", "traffic_light"},
                    {"static_attributes", {
                        {"length", actor.length.value_or(1.0)},
                        {"width", actor.width.value_or(1.0)},
                        {"rear_axis_offset", 0}
                    }},
                    {"states", states_dict}
                };
            }
        }
        return controls_dict;
    }

    json ScenarioLogWriter::build_predetermined_agents_dict(
        const ScenarioLog& log
    ) const {
        json agents_dict = json::object();
        
        for (size_t i = 0; i < log.agent_properties.size(); i++) {
            const auto& prop = log.agent_properties[i];
            json states_dict = json::object();
            
            for (size_t t = 0; t < log.agent_states.size(); t++) {
                const auto& present_idx = log.present_indexes[t];
                
                // Check if agent i is present at time t
                auto it = std::find(present_idx.begin(), present_idx.end(), static_cast<int>(i));
                if (it != present_idx.end()) {
                    int idx = std::distance(present_idx.begin(), it);
                    const auto& state = log.agent_states[t][idx];
                    
                    states_dict[std::to_string(t)] = {
                        {"center", {
                            {"x", state.x},
                            {"y", state.y}
                        }},
                        {"orientation", state.orientation},
                        {"speed", state.speed}
                    };
                }
            }
            std::string agent_type_str = prop.agent_type.has_value() 
                ? prop.agent_type.value() 
                : std::string("car");
            agents_dict[std::to_string(i)] = json::object();
            agents_dict[std::to_string(i)]["entity_type"] = agent_type_str;
            agents_dict[std::to_string(i)]["static_attributes"] = {
                {"length", prop.length.value_or(0.0)},
                {"width", prop.width.value_or(0.0)},
                {"rear_axis_offset", prop.rear_axis_offset.value_or(0.0)}
            };
            agents_dict[std::to_string(i)]["states"] = states_dict;
        }
        
        return agents_dict;
    }

    void ScenarioLogWriter::export_to_file(
        const std::string& log_path,
        std::optional<ScenarioLog> scenario_log,
        std::optional<LocationInfoResponse> location_info_response
    ) {    
        const ScenarioLog& log = scenario_log.has_value() ? scenario_log.value() : scenario_log_;    
        auto [num_cars, num_pedestrians] = count_agent_types();
        auto [num_lights, num_yield, num_stop, num_other] = count_control_types(location_info_response.value());
    
        json individual_suggestions = build_individual_suggestions_dict(log);
    
        json predetermined_agents = build_predetermined_agents_dict(log);
    
        json predetermined_controls = json::object();
        if (location_info_response.has_value()) {
            std::tie(num_lights, num_yield, num_stop, num_other) =
                count_control_types(location_info_response.value());
            predetermined_controls =
                build_predetermined_controls_dict(log, location_info_response.value());
        } else {
        }

        json light_recurrent_array = json::array();
        if (log.light_recurrent_states.has_value()) {
            for (const auto& lrs : log.light_recurrent_states.value()) {
                // Each entry should be a two-element array: [state, time_remaining]
                light_recurrent_array.push_back({lrs.state, lrs.time_remaining});
            }
        }
    
        json output_dict = {
            {"location", {
                {"identifier", log.location}
            }},
            {"scenario_length", log.agent_states.size()},
            {"num_agents", {
                {"car", num_cars},
                {"pedestrian", num_pedestrians}
            }},
            {"predetermined_agents", predetermined_agents},
            {"num_controls", {
                {"traffic_light", num_lights},
                {"yield_sign", num_yield},
                {"stop_sign", num_stop},
                {"other", num_other}
            }},
            {"predetermined_controls", predetermined_controls},
            {"individual_suggestions", individual_suggestions},
            {"initialize_random_seed", log.initialize_random_seed.value_or(0)},
            {"lights_random_seed", log.lights_random_seed.value_or(0)},
            {"drive_random_seed", log.drive_random_seed.value_or(0)},
            {"drive_model_version", log.drive_model_version.value_or("best")},
            {"initialize_model_version", log.initialize_model_version.value_or("best")},
            {"light_recurrent_states", light_recurrent_array}
        };
    
        if (log.rendering_center.has_value()) {
            output_dict["birdview_options"] = {
                {"rendering_center", {
                    log.rendering_center->first,
                    log.rendering_center->second
                }},
                {"renderingFOV", log.rendering_fov.value()}
            };
            output_dict["rendering_centers"] = {
                log.rendering_center->first,
                log.rendering_center->second
            };
        }
    
        std::ofstream outfile(log_path);
        if (!outfile.is_open()) {
            throw std::runtime_error("Failed to open file: " + log_path);
        }
    
        std::string dump_str;
        try {
            dump_str = output_dict.dump(4);
        } catch (const std::exception& e) {
            throw;
        }
    
        outfile << dump_str;
        outfile.close();
    }
    


    void ScenarioLogWriter::initialize(
        std::optional<std::string> location,
        std::optional<LocationInfoResponse> location_info_response,
        std::optional<InitializeResponse> init_response,
        std::optional<int> lights_random_seed,
        std::optional<int> initialize_random_seed,
        std::optional<int> drive_random_seed,
        std::optional<std::string> drive_model_version,
        std::optional<ScenarioLog> scenario_log
    ) {
        if (scenario_log.has_value()) {
            // Using provided scenario log
            scenario_log_ = scenario_log.value();
            
            if (scenario_log_.present_indexes.empty()) {
                std::vector<int> initial_present(scenario_log_.agent_properties.size());
                std::iota(initial_present.begin(), initial_present.end(), 0);
                scenario_log_.present_indexes.push_back(initial_present);
            }
            
            simulation_length = scenario_log_.agent_states.size();
        } else {
            if (!location.has_value()) {
                throw std::invalid_argument("No scenario log given, must provide a location argument.");
            }
            if (!location_info_response.has_value()) {
                throw std::invalid_argument("No scenario log given, must provide a location_info_response argument.");
            }
            if (!init_response.has_value()) {
                throw std::invalid_argument("No scenario log given, must provide a init_response argument.");
            }
            
            const auto& init_resp = init_response.value();
            const auto& loc_resp = location_info_response.value();
            
            std::vector<std::vector<int>> present_indexes;
            std::vector<int> initial_present(init_resp.agent_properties().size());
            std::iota(initial_present.begin(), initial_present.end(), 0);
            present_indexes.push_back(initial_present);
            scenario_log_ = ScenarioLog(
                location.value(),
                {init_resp.agent_states()},
                init_resp.agent_properties(),
                std::optional<std::vector<std::map<std::string, std::string>>>(
                    std::vector<std::map<std::string,std::string>>{ init_resp.traffic_lights_states().value() }
                ),
                std::optional<std::pair<double,double>>(std::make_pair(loc_resp.rendering_center().x,loc_resp.rendering_center().y)),
                loc_resp.rendering_fov(),
                lights_random_seed,
                initialize_random_seed,
                drive_random_seed,
                init_resp.model_version(),
                drive_model_version,
                init_resp.light_recurrent_states(),
                init_resp.recurrent_states(),
                std::nullopt,
                present_indexes
            );
            
            simulation_length = 1;
        }
    }

    void ScenarioLogWriter::drive(
        const DriveResponse& drive_response,
        std::optional<std::vector<int>> current_present_indexes,
        std::optional<std::vector<AgentProperties>> new_agent_properties,
        std::optional<std::map<int, std::optional<Point2d>>> waypoints
    ) {
        if (new_agent_properties.has_value()) {
            scenario_log_.agent_properties.insert(
                scenario_log_.agent_properties.end(),
                new_agent_properties.value().begin(),
                new_agent_properties.value().end()
            );
        }
        
        std::vector<int> present_idx;
        if (!current_present_indexes.has_value()) {
            present_idx = scenario_log_.present_indexes[simulation_length - 1];
        } else {
            present_idx = current_present_indexes.value();
        }
        
        scenario_log_.add_time_step_data(
            drive_response.agent_states(),
            present_idx
        );
        
        if (drive_response.traffic_lights_states().has_value()) {
            if (!scenario_log_.traffic_lights_states.has_value()) {
                scenario_log_.traffic_lights_states = std::vector<std::map<std::string, std::string>>();
            }
            scenario_log_.traffic_lights_states.value().push_back(
                drive_response.traffic_lights_states().value()
            );
        }
        
        if (waypoints.has_value()) {
            if (!scenario_log_.waypoints_per_frame.has_value()) {
                scenario_log_.waypoints_per_frame = std::vector<std::map<int, Point2d>>();
            }
            
            std::map<int, Point2d> cleaned_waypoints;
            for (const auto& [aid, wp] : waypoints.value()) {
                if (wp.has_value()) {
                    cleaned_waypoints[aid] = wp.value();
                }
            }
            
            scenario_log_.waypoints_per_frame.value().push_back(cleaned_waypoints);
        }
        
        scenario_log_.drive_model_version = drive_response.model_version();
        scenario_log_.light_recurrent_states = drive_response.light_recurrent_states();
        scenario_log_.recurrent_states = drive_response.recurrent_states();
        
        simulation_length++;
    }

    std::vector<int> ScenarioLogWriter::current_present_indexes() const {
        return scenario_log_.present_indexes[simulation_length - 1];
    }

    std::vector<AgentProperties> ScenarioLogWriter::all_agent_properties() const {
        return scenario_log_.agent_properties;
    }
}