#include "logger.h"
#include "externals/json.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <time.h>
#include "../invertedai/api.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>

using json = nlohmann::json;

namespace invertedai {
    ScenarioLog::ScenarioLog(
        std::vector<std::vector<AgentState>> agent_states_,
        std::vector<AgentProperties> agent_properties_,
        std::optional<std::vector<std::map<std::string, std::string>>> traffic_states_,
        std::string location_,
        std::optional<std::pair<double,double>> rendering_center_,
        std::optional<int> rendering_fov_,

        std::optional<int> lights_seed_,
        std::optional<int> init_seed_,
        std::optional<int> drive_seed_,

        std::optional<std::string> init_version_,
        std::optional<std::string> drive_version_,

        std::optional<LightRecurrentState> light_states_,
        std::optional<std::vector<RecurrentState>> recurrent_states_,
        std::optional<std::map<std::string, std::vector<Point2d>>> waypoints_,
        std::vector<std::vector<int>> present_indexes_
    ) {
        agent_states = agent_states_;
        agent_properties = agent_properties_;
        traffic_lights_states = traffic_states_;
    
        location = location_;
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
    void LogReader::read_log(const std::string &file_path, std::string API_KEY) { 
        std::string json_body = invertedai::read_file(file_path.c_str());

        json j = json::parse(json_body);

        location = j["location"]["identifier"];

        int scenario_length = 0;
        if (j.contains("scenario_length")) { // im not familiar w these json logs, so just as a safety measure ?? clarify later!
            scenario_length = j["scenario_length"];
        }
        if (j.contains("num_agents")) { // assuming only car agents for now
            this->num_agents = j["num_agents"]["car"];
        }
        std::vector<std::map<std::string, AgentState>> all_agent_states_unsorted;
        std::map<std::string, AgentProperties> all_agent_properties_unsorted;

        std::vector<std::vector<int>> present_indexes_unsorted;

        std::map<std::string, int> agent_id_list;
        int agent_id_sequence_num = 0;

        for(int t = 0; t < scenario_length; t++) {
            std::map<std::string, AgentState> agent_states_ts;
            std::vector<int> present_indexes_ts;

            for (auto& kv : j["predetermined_agents"].items()) {
                std::string agent_id = kv.key();
                const json& agent = kv.value();

                // If first time encountering the agent ID
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
                // Read agent state for this time step
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

        for (auto& kv : properties_sorted) {
            sorted_agent_properties.push_back(kv.second);
        }
        
     
        for (auto& states_map : all_agent_states_unsorted) {
        
            auto sorted_vec = sort_dict(states_map, agent_id_list);
        
            std::vector<AgentState> states_only;
            states_only.reserve(sorted_vec.size());
        
            for (auto& kv : sorted_vec) {
                states_only.push_back(kv.second);
            }

            agent_states_over_time.push_back(states_only);
        }
        
        // sort present indexes
        std::vector<std::vector<int>> present_indexes_sorted;
        for (auto& vec : present_indexes_unsorted) {
            std::sort(vec.begin(), vec.end());
            present_indexes_sorted.push_back(vec);
        }

        if (j.contains("predetermined_controls")) {
            std::vector<std::map<std::string,std::string>> tl_history(scenario_length);
            tl_history.resize(scenario_length);

            for (int t = 0; t < scenario_length; t++) {
                std::string ts_key = std::to_string(t);
        
                for (auto& kv : j["predetermined_controls"].items()) {
                    const std::string actor_id = kv.key();
                    const json& actor = kv.value();
        
                    if (actor["entity_type"] == "traffic_light" &&
                        actor["states"].contains(ts_key)) 
                    {
                        tl_history[t][actor_id] =
                            actor["states"][ts_key]["control_state"].get<std::string>();
                    }
                }
            }
            this->traffic_light_states_over_time = tl_history;
        }
        std::map<std::string, std::vector<Point2d>> agent_waypoints;

        if (j.contains("individual_suggestions")) { // ignore waypoints for now...
            std::map<std::string,std::vector<Point2d>> wp;
            for (auto& kv : j["individual_suggestions"].items()) {
                std::string ag = kv.key();
                for (auto& st : kv.value()["states"]) {
                    const json& c = st["center"];
                    wp[ag].push_back(Point2d{c["x"], c["y"]});
                }
            }
            if (!wp.empty())
                waypoints = wp;
        }

        // light rec state
        std::optional<LightRecurrentState> light_rs = std::nullopt;
        if (j.contains("light_recurrent_states") && j["light_recurrent_states"].is_array()) {
            LightRecurrentState lrs;
            lrs.state = j["light_recurrent_states"][0];
            lrs.time_remaining = j["light_recurrent_states"][1];
            light_rs = lrs;
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

        // construct scneario log
        scenario_log_ = ScenarioLog(
            agent_states_over_time,
            sorted_agent_properties,
            traffic_light_states_over_time.has_value() ? this->traffic_light_states_over_time : std::nullopt,
            location,
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
            waypoints,
            present_indexes_unsorted
        );

        scenario_log_original_ = scenario_log_;

        reset_log();

        simulation_length = agent_states_over_time.size();
        initialize_model_version_ = init_version;
        drive_model_version_ = drive_version;
        waypoints = waypoints;

        std::string loc_body = "{}";
        invertedai::LocationInfoRequest loc_info_req(loc_body);
        loc_info_req.set_location(location);
        loc_info_req.set_rendering_center(scenario_log_.rendering_center);
        loc_info_req.set_rendering_fov(scenario_log_.rendering_fov);
        net::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        // configure connection setting
        invertedai::Session session(ioc, ctx);
        session.set_api_key(API_KEY);
        session.connect();
        location_info_response_ = invertedai::location_info(loc_info_req, &session); // ! TODO 

    }

    bool LogReader::return_state_at_timestep(int t) {
        if (t < 0 || t >= simulation_length)
            return false;
        agent_states = scenario_log_.agent_states[t];

        // Properties must match present_indexes[t]
        agent_properties.clear();
        for (int idx : scenario_log_.present_indexes[t])
            agent_properties.push_back(scenario_log_.agent_properties[idx]);

        // Traffic lights
        if (scenario_log_.traffic_lights_states &&
            t < (int)scenario_log_.traffic_lights_states->size())
            traffic_lights_states = (*scenario_log_.traffic_lights_states)[t];
        else
            traffic_lights_states = std::nullopt;

        // Recurrent states: only present at last timestep in Python
        if (t == simulation_length - 1) {
            light_recurrent_states = scenario_log_.light_recurrent_states;
            recurrent_states = std::nullopt;
        } else {
            light_recurrent_states = std::nullopt;
            recurrent_states= std::nullopt;
        }

        return true;
    }

    bool LogReader::initialize() {
        bool init_response = return_state_at_timestep(0);
        current_timestep = 1;
        return init_response;
    }

    bool LogReader::drive() {
        if (current_timestep >= simulation_length) 
            return false;
        bool response = return_state_at_timestep(current_timestep);
        current_timestep += 1;
        return response;
    }

    bool LogReader::return_last_state() {
        return return_state_at_timestep(simulation_length-1);
    }
    void LogReader::reset_log() {
        scenario_log_ = scenario_log_original_;
        agent_states.clear();
        agent_properties.clear();
        traffic_lights_states = std::nullopt;
        light_recurrent_states = std::nullopt;
        recurrent_states = std::nullopt;
        current_timestep = 1;
    }
    std::string LogReader::get_location() {
        return this->location;
    }
    int LogReader::get_total_num_agents() {
        return this->num_agents;
    }
    int LogReader::get_scenario_length() {
        return this->simulation_length;
    }
    std::vector<AgentProperties> LogReader::get_agent_properties() {
        return this->sorted_agent_properties;
    }
    std::vector<std::vector<AgentState>> LogReader::get_agent_states_over_time() {
        return this->agent_states_over_time;
    }
    std::optional<std::vector<std::map<std::string, std::string>>>
    LogReader::get_traffic_lights_states_over_time() {
        return this->traffic_light_states_over_time;
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

    void LogWriter::write_log_to_file(const std::string &dir_path){
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

        std::string file_path = "iai_log_" + this->get_current_time_UTC_() + ".json";
        std::string full_path = dir_path + file_path;

        std::cout << "INFO: IAI Log written to path: " << full_path << std::endl;

        std::ofstream o(full_path);
        o << std::setw(4) << log << std::endl;
    };
}