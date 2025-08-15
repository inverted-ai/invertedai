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