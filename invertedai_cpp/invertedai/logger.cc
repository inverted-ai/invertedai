#include "logger.h"
#include "externals/json.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <time.h>

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

    LogReader::LogReader(const std::string &file_path) { 
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
            auto rendering_fov = j["birdview_options"]["renderingFOV"].get<int>();
        auto rendering_center = std::optional<std::pair<double,double>>({
            j["birdview_options"]["rendering_center"][0],
            j["birdview_options"]["rendering_center"][1]
        });
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
    
    const std::string& LogReader::get_location() const {
        return this->scenario_log_.location;
    }
    
    int LogReader::get_fov() {
        return this->scenario_log_.rendering_fov.value_or(200);
    }
    std::optional<std::pair<double, double>> LogReader::get_rendering_center() {
        return this->scenario_log_.rendering_center;
    }
    ScenarioLog LogReader::get_scenario_log() {
        return this->scenario_log_;
    }

    bool LogReader::return_state_at_timestep(int t) {
        if (t < 0 || t >= simulation_length) {
            return false;
        }
        current_timestep = t;
        return true;
    }
    
    bool LogReader::return_last_state() {
        if (simulation_length == 0) {
            return false;
        }
        current_timestep = simulation_length - 1;
        return true;
    }

    bool LogReader::initialize() {
        bool init_response = return_state_at_timestep(0);
        current_timestep = 1;
        return init_response;
    }

    bool LogReader::drive() {
        return next();
    }

    void LogReader::reset_log() {
        current_timestep = 0;
    }

    bool LogReader::next() {
        if (current_timestep + 1 >= simulation_length) {
            return false;
        }
        ++current_timestep;
        return true;
    }

    const std::vector<AgentState>&
    LogReader::current_agent_states() const {
        return scenario_log_.agent_states[current_timestep];
    }

    std::vector<AgentProperties>
    LogReader::current_agent_properties() const {
        std::vector<AgentProperties> result;
        for (int idx : scenario_log_.present_indexes[current_timestep]) {
            result.push_back(scenario_log_.agent_properties[idx]);
        }
        return result;
    }

    std::optional<std::map<std::string, std::string>>
    LogReader::current_traffic_lights() const {
        if (!scenario_log_.traffic_lights_states) {
            return std::nullopt;
        }
        return (*scenario_log_.traffic_lights_states)[current_timestep];
    }

    int LogReader::get_scenario_length() {
        return this->scenario_log_.agent_states.size();
    }

    std::vector<AgentProperties> LogReader::get_agent_properties() {
        return this->scenario_log_.agent_properties;
    }

    std::vector<std::vector<AgentState>> LogReader::get_agent_states_over_time() {
        return scenario_log_.agent_states;
    }

    std::optional<std::vector<std::map<std::string, std::string>>>
    LogReader::get_traffic_lights_states_over_time() {
        return this->scenario_log_.traffic_lights_states;
    }

    std::optional<std::vector<LightRecurrentState>> LogReader::current_light_recurrent_state() const {
        return this->scenario_log_.light_recurrent_states;
    }
    
    std::optional<std::vector<RecurrentState>> LogReader::current_recurrent_states() const {
        return this->scenario_log_.recurrent_states;
    }

    
    std::string LogWriter::get_current_time_UTC_(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        std::time_t now_t = std::time(0);
        std::tm* now_tm = std::gmtime(&now_t);
        char buf[42];
        std::strftime(buf, 42, "%Y-%m-%d_%H:%M:%S", now_tm);
        std::string date_time = buf;
        date_time += ":" + std::to_string(milliseconds % 1000) + "_UTC";

        return date_time;
    };

    void LogWriter::append_request(const std::string &req, const std::string &mode){
        std::string date_time = this->get_current_time_UTC_();

        if (mode == "location_info"){
            this->loc_request_timestamps_.push_back(date_time);
            this->loc_requests_.push_back(req);
        }
        else if (mode == "initialize"){
            this->init_request_timestamps_.push_back(date_time);
            this->init_requests_.push_back(req);
        }
        else if (mode == "drive"){
            this->drive_request_timestamps_.push_back(date_time);
            this->drive_requests_.push_back(req);
        }
    };

    void LogWriter::append_response(const std::string &res, const std::string &mode){
        std::string date_time = this->get_current_time_UTC_();

        if (mode == "location_info"){
            this->loc_response_timestamps_.push_back(date_time);
            this->loc_responses_.push_back(res);
        }
        else if (mode == "initialize"){
            this->init_response_timestamps_.push_back(date_time);
            this->init_responses_.push_back(res);
        }
        else if (mode == "drive"){
            this->drive_response_timestamps_.push_back(date_time);
            this->drive_responses_.push_back(res);
        }
    };

    nlohmann::ordered_json get_agent_state_data(nlohmann::ordered_json res_agents, const int &index) {
        std::vector<double> state = res_agents[index];
                
        nlohmann::ordered_json state_info;
        state_info["center"]["x"] = state[0];
        state_info["center"]["y"] = state[1];
        state_info["orientation"] = state[2];
        state_info["speed"] = state[3];

        return state_info;
    }

    void LogWriter::write_scenario_log(const std::string &dir_path, const std::string &log_path = ""){
        // Produce an IAI formatted log that can be used in various applications
        // Assumptions: The number of vehicles stays consistent throughout the simulation
        
        //Get data to parse from specified source
        std::vector<nlohmann::ordered_json> drive_responses;
        nlohmann::ordered_json last_init_res;
        nlohmann::ordered_json last_init_req;
        nlohmann::ordered_json last_drive_req;
        nlohmann::ordered_json last_loc_res;
        nlohmann::ordered_json last_loc_req;

        if (log_path.empty()){
            last_init_res = json::parse(this->init_responses_.back());
            last_init_req = json::parse(this->init_requests_.back());
            
            for (auto res: this->drive_responses_) {
                drive_responses.push_back(json::parse(res));
            }
            last_drive_req = json::parse(this->drive_requests_.back());

            //TODO: Find a way to produce data that would be in location info without using Session object.
            if (this->loc_responses_.empty()) {
                last_loc_res = NULL;
            }
            else {
                last_loc_res = json::parse(this->loc_responses_.back());
            }
            if (this->loc_requests_.empty()) {
                last_loc_req = NULL;
            }
            else {
                try {
                    last_loc_req = json::parse(this->loc_requests_.back());
                }
                catch (nlohmann::json_abi_v3_11_2::detail::parse_error err) {
                    std::cout << "WARNING: Could not parse request message: " << this->loc_requests_.back() << ". Not processing data from this message." << std::endl;
                }
            }
        }
        else {
            std::ifstream f(log_path);
            nlohmann::ordered_json data = json::parse(f);

            last_init_res = data["initialize_responses"].back();
            last_init_req = data["initialize_requests"].back();
            std::vector<nlohmann::ordered_json> drive_responses;
            for (auto res: data["drive_responses"]) {
                drive_responses.push_back(res);
            }
            last_drive_req = data["drive_requests"].back();

            if (data["location_responses"].empty()) {
                last_loc_res = NULL;
            }
            else {
                last_loc_res = data["location_responses"].back();
            }
            if (data["location_requests"].empty()) {
                last_loc_req = NULL;
            }
            else {
                last_loc_req = data["location_requests"].back();
            }
        }

        //Begin parsing data
        int num_drive_responses = drive_responses.size();
        nlohmann::ordered_json predetermined_agents;
        nlohmann::ordered_json predetermined_controls;

        //Get agent properties, as well as initial simulation states for agents and controls
        nlohmann::ordered_json num_controls;
        num_controls["traffic_light"] = 0;
        num_controls["yield_sign"] = 0;
        num_controls["stop_sign"] = 0;
        num_controls["other"] = 0;

        if (!last_loc_res.is_null()) {
            if (!last_loc_res["static_actors"].empty()) {
                for (auto actor: last_loc_res["static_actors"]) {
                    std::string agent_type = actor["agent_type"];
                    if (num_controls.contains(agent_type)) {
                        num_controls[agent_type] += 1;
                    }
                    else {
                        num_controls["other"] += 1;
                    }
                    nlohmann::ordered_json control_data;
                    control_data["entity_type"] = agent_type;
                    nlohmann::ordered_json static_attributes;
                    static_attributes["length"] = actor["length"];
                    static_attributes["width"] = actor["width"];
                    static_attributes["rear_axis_offset"] = 0.0;
                    control_data["static_attributes"] = static_attributes;

                    nlohmann::ordered_json states;
                    control_data["states"] = states;
                    int id_int = actor["actor_id"];
                    std::string actor_id = std::to_string(id_int);
                    predetermined_controls[actor_id] = control_data;

                    nlohmann::ordered_json controls_info;
                    controls_info["center"]["x"] = actor["x"];
                    controls_info["center"]["y"] = actor["y"];
                    controls_info["orientation"] = actor["orientation"];
                    controls_info["speed"] = 0.0;
                    if (agent_type == "traffic_light") {
                        controls_info["control_state"] = last_init_res["traffic_lights_states"][actor_id];
                    }
                    else {
                        controls_info["control_state"] = "none";
                    }
                    predetermined_controls[actor_id]["states"]["0"] = controls_info;
                }
            }
            
        }

        //Get agent properties, as well as initial simulation states for agents and controls
        int num_vehicles = 0;
        int num_pedestrians = 0;
        for (int i = 0; i < last_init_res["agent_properties"].size(); i++) {
            nlohmann::ordered_json prop = last_init_res["agent_properties"][i];
            std::string entity_type = prop["agent_type"];
            std::string agent_id = std::to_string(i);

            if (entity_type == "car") {
                num_vehicles++;
            }
            if (entity_type == "pedestrian") {
                num_pedestrians++;
            }

            predetermined_agents[agent_id]["entity_type"] = entity_type;
            
            nlohmann::ordered_json static_attributes;
            static_attributes["length"] = prop["length"];
            static_attributes["width"] = prop["width"];
            static_attributes["rear_axis_offset"] = prop["rear_axis_offset"];
            static_attributes["is_parked"] = false;
            predetermined_agents[agent_id]["static_attributes"] = static_attributes;

            nlohmann::ordered_json states;
            predetermined_agents[agent_id]["states"] = states;
            predetermined_agents[agent_id]["states"]["0"] = get_agent_state_data(last_init_res["agent_states"],i);
        }

        //Get all agent states data for all time steps
        for (int i = 0; i < num_drive_responses; i++) {
            nlohmann::ordered_json drive_res = drive_responses[i];
            std::string ts = std::to_string(i+1);
            
            for (int j = 0; j < drive_res["agent_states"].size(); j++) {
                std::string agent_id = std::to_string(j);
                predetermined_agents[agent_id]["states"][ts] = get_agent_state_data(drive_res["agent_states"],j);
            }

            if (!drive_res["traffic_lights_states"].empty() && !last_loc_res.is_null()) {
                for (const auto& tl_state: drive_res["traffic_lights_states"].items()) {
                    nlohmann::ordered_json tl_state_template = predetermined_controls[tl_state.key()]["states"]["0"];
                    tl_state_template["control_state"] = tl_state.value();
                    predetermined_controls[tl_state.key()]["states"][ts] = tl_state_template;
                }
            }
        }

        //Get waypoint information for last time step
        nlohmann::ordered_json individual_suggestions;
        if (!last_drive_req.is_null()) {
            for (int i = 0; i < last_drive_req["agent_properties"].size(); i++) {
                nlohmann::ordered_json prop = last_drive_req["agent_properties"][i];

                if (prop.contains("waypoint")) {
                    nlohmann::ordered_json wp;
                    wp["suggestion_strength"] = 0.8;
                    std::vector<nlohmann::ordered_json> wp_states;
                    nlohmann::ordered_json wp_state_next;
                    
                    wp_state_next["center"]["x"] = prop["waypoint"][0];
                    wp_state_next["center"]["y"] = prop["waypoint"][1];
                    wp_states.push_back(wp_state_next);
                    wp["states"] = wp_states;

                    individual_suggestions[std::to_string(i)] = wp;
                }

            }
        }
        
        nlohmann::ordered_json birdview_options;
        if (!last_loc_req.is_null()) {
            birdview_options["rendering_center"] = last_loc_req["rendering_center"];
            birdview_options["renderingFOV"] = last_loc_req["renderingFOV"];
        }
        nlohmann::ordered_json light_recurrent_states = drive_responses.back()["light_recurrent_states"];

        nlohmann::ordered_json scenario_log;
        scenario_log["location"]["identifier"] = last_init_req["location"];
        scenario_log["scenario_length"] = num_drive_responses;
        scenario_log["num_agents"]["car"] = num_vehicles;
        scenario_log["num_agents"]["pedestrian"] = num_pedestrians;
        scenario_log["predetermined_agents"] = predetermined_agents;        
        scenario_log["num_controls"] = num_controls;
        scenario_log["predetermined_controls"] = predetermined_controls;
        scenario_log["individual_suggestions"] = individual_suggestions;
        if (!last_drive_req.is_null()) {
            scenario_log["drive_random_seed"] = last_drive_req["random_seed"];
            scenario_log["drive_model_version"] = last_drive_req["model_version"];
        }
        else {
            scenario_log["drive_random_seed"] = NULL;
            scenario_log["drive_model_version"] = NULL;
        }
        scenario_log["birdview_options"] = birdview_options;
        scenario_log["light_recurrent_states"] = light_recurrent_states;


        std::string file_path = "iai_scenario_log_" + this->get_current_time_UTC_() + ".json";
        std::string full_path = dir_path + file_path;

        std::cout << "INFO: IAI Scenario Log written to path: " << full_path << std::endl;

        std::ofstream o(full_path);
        o << std::setw(4) << scenario_log << std::endl;
    };
    
    void LogWriter::write_log_to_file(const std::string &dir_path, const bool &is_scenario_log = false){
        if (is_scenario_log){
            this->write_scenario_log(dir_path);
        }
        else {
            json log;

            log["location_requests"] = this->loc_requests_;
            log["location_responses"] = this->loc_responses_;

            log["location_request_timestamps"] = this->loc_request_timestamps_;
            log["location_response_timestamps"] = this->loc_response_timestamps_;

            log["initialize_requests"] = this->init_requests_;
            log["initialize_responses"] = this->init_responses_;

            log["initialize_request_timestamps"] = this->init_request_timestamps_;
            log["initialize_response_timestamps"] = this->init_response_timestamps_;

            log["drive_requests"] = this->drive_requests_;
            log["drive_responses"] = this->drive_responses_;

            log["drive_request_timestamps"] = this->drive_request_timestamps_;
            log["drive_response_timestamps"] = this->drive_response_timestamps_;

            std::string file_name = this->get_current_time_UTC_() + ".json";
            std::string file_path = "iai_log_" + file_name;
            std::string full_path = dir_path + file_path;

            std::cout << "INFO: IAI Log written to path: " << full_path << std::endl;

            std::ofstream o(full_path);
            o << std::setw(4) << log << std::endl;
        }
    };
}