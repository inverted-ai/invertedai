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
    void LogReader::read_log(const std::string &file_path) { 
        std::string json_body = invertedai::read_file(file_path.c_str());

        json j = json::parse(json_body);

        location = j["location"]["identifier"];

        if (j.contains("scenario_length")) { // im not familiar w these json logs, so just as a safety measure ?? clarify later!
            this->scenario_length = j["scenario_length"];
        }
        if (j.contains("num_agents")) { // assuming only car agents for now
            this->num_agents = j["num_agents"]["car"];
        }
        std::vector<std::map<std::string, AgentState>> all_agent_states_unsorted;
        std::map<std::string, AgentProperties> all_agent_properties_unsorted;

        std::vector<std::vector<int>> present_indexes_unsorted;

        std::map<std::string, int> agent_id_list;
        int agent_id_sequence_num = 0;

        for(int t = 0; t < this->scenario_length; t++) {
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

        // if (j.contains("individual_suggestions")) { // ignore waypoints for now...

        //     for (auto& kv : j["individual_suggestions"].items()) {
        //         std::string agent_id = kv.key();
        //         agent_waypoints[agent_id] = {};

        //         for (auto& pt : kv.value()["states"]) {
        //             const json& c = pt["center"];
        //             agent_waypoints[agent_id].push_back(
        //                 Point::fromList({c["x"], c["y"]})
        //             );
        //         }
        //     }

        //     if (agent_waypoints.empty())
        //         this->agent_waypoints = std::nullopt;
        //     else
        //         this->agent_waypoints = agent_waypoints;
        // }
    }

    std::string LogReader::get_location() {
        return this->location;
    }
    int LogReader::get_total_num_agents() {
        return this->num_agents;
    }
    int LogReader::get_scenario_length() {
        return this->scenario_length;
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