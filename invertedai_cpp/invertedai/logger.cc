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

    void LogWriter::write_scenario_log(const std::string &dir_path){
        // Produce an IAI formatted log that can be used in various applications
        // Assumptions: The number of vehicles stays consistent throughout the simulation
        
        json scenario_log;

        json last_init_res = json::parse(this->init_responses_.back());
        json last_init_req = json::parse(this->init_requests_.back());
        json drive_responses;

        int num_drive_responses = this->drive_responses_.size();
        scenario_log["location"]["identifier"] = last_init_req["location"];
        scenario_log["scenario_length"] = num_drive_responses;

        int num_vehicles = 0;
        int num_pedestrians = 0;

        json predetermined_agents;

        for (auto i = 0; i < last_init_res["agent_properties"].size(); i++) {
            auto prop = last_init_res["agent_properties"][i];
            std::string entity_type = prop["agent_type"];
            std::string agent_id = std::to_string(i);

            if (entity_type == "car") {
                num_vehicles++;
            }
            if (entity_type == "pedestrian") {
                num_pedestrians++;
            }

            predetermined_agents[agent_id]["entity_type"] = entity_type;
            
            json static_attributes;
            static_attributes["length"] = prop["length"];
            static_attributes["width"] = prop["width"];
            static_attributes["rear_axis_offset"] = prop["rear_axis_offset"];
            static_attributes["is_parked"] = prop["is_parked"];
            predetermined_agents[agent_id]["static_attributes"] = static_attributes;

            json states;
            predetermined_agents[agent_id]["states"] = states;

        }

        scenario_log["num_agents"]["car"] = num_vehicles;
        scenario_log["num_agents"]["pedestrian"] = num_pedestrians;

        
        for (auto i = 0; i < num_drive_responses; i++) {
            json drive_res = json::parse(this->drive_responses_[i]);
            std::string ts = std::to_string(i);
            
            for (auto j = 0; j < drive_res["agent_states"].size(); j++) {
                std::vector<double> state = drive_res["agent_states"][j];
                std::string agent_id = std::to_string(j);
                
                json state_info;
                state_info["center"]["x"] = state[0];
                state_info["center"]["y"] = state[1];
                state_info["orientation"] = state[2];
                state_info["speed"] = state[3];
                predetermined_agents[agent_id]["states"][ts] = state_info;

            }
        }

        scenario_log["predetermined_agents"] = predetermined_agents;

        std::string file_path = "iai_scenario_log_" + this->get_current_time_UTC_() + ".json";
        std::string full_path = dir_path + file_path;

        std::cout << "INFO: IAI Scenario Log written to path: " << full_path << std::endl;

        std::ofstream o(full_path);
        o << std::setw(4) << log << std::endl;
    };
}