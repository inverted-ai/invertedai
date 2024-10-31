#include "logger.h"
#include "externals/json.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

using json = nlohmann::json;

namespace invertedai {

    UL LogWriter::get_current_time_in_milliseconds_(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        return (UL) milliseconds;
    };

    void LogWriter::append_request(const std::string &req, const std::string &mode){
        UL time_ms = this->get_current_time_in_milliseconds_();

        if (mode == "location_info"){
            this->loc_request_times_.push_back(time_ms);
            this->loc_requests_.push_back(req);
        }
        else if (mode == "initialize"){
            this->init_request_times_.push_back(time_ms);
            this->init_requests_.push_back(req);
        }
        else if (mode == "drive"){
            this->drive_request_times_.push_back(time_ms);
            this->drive_requests_.push_back(req);
        }
    };

    void LogWriter::append_response(const std::string &res, const std::string &mode){
        UL time_ms = this->get_current_time_in_milliseconds_();

        if (mode == "location_info"){
            this->loc_response_times_.push_back(time_ms);
            this->loc_responses_.push_back(res);
        }
        else if (mode == "initialize"){
            this->init_response_times_.push_back(time_ms);
            this->init_responses_.push_back(res);
        }
        else if (mode == "drive"){
            this->drive_response_times_.push_back(time_ms);
            this->drive_responses_.push_back(res);
        }
    };

    void LogWriter::write_log_to_file(const std::string &dir_path){
        json log;

        log["location_requests"] = this->loc_requests_;
        log["location_responses"] = this->loc_responses_;

        log["location_request_times"] = this->loc_request_times_;
        log["location_response_times"] = this->loc_response_times_;

        log["initialize_requests"] = this->init_requests_;
        log["initialize_responses"] = this->init_responses_;

        log["initialize_request_times"] = this->init_request_times_;
        log["initialize_response_times"] = this->init_response_times_;

        log["drive_requests"] = this->drive_requests_;
        log["drive_responses"] = this->drive_responses_;

        log["drive_request_times"] = this->drive_request_times_;
        log["drive_response_times"] = this->drive_response_times_;

        std::string file_path = "iai_log" + std::to_string(this->get_current_time_in_milliseconds_()) + ".json";
        std::string full_path = dir_path + file_path;

        std::ofstream o(full_path);
        o << std::setw(4) << log << std::endl;
    };
}