#include "logger.h"
#include "externals/json.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

namespace invertedai {

    ul LogWriter::get_current_time_in_milliseconds_(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        return (ul) milliseconds;
    };

    void LogWriter::set_loc_request(const LocationInfoRequest &loc_request){
        this->loc_request_time_ = this->get_current_time_in_milliseconds_();
        this->loc_request_.clear();
        this->loc_request_.push_back(loc_request);
    };

    void LogWriter::set_loc_response(const LocationInfoResponse &loc_response){
        this->loc_response_time_ = this->get_current_time_in_milliseconds_();
        this->loc_response_.clear();
        this->loc_response_.push_back(loc_response);
    };

    void LogWriter::set_init_request(const InitializeRequest &init_request){
        this->init_request_time_ = this->get_current_time_in_milliseconds_();
        this->init_request_.clear();
        this->init_request_.push_back(init_request);
    };

    void LogWriter::set_init_response(const InitializeResponse &init_response){
        this->init_response_time_ = this->get_current_time_in_milliseconds_();
        this->init_response_.clear();
        this->init_response_.push_back(init_response);
    };

    void LogWriter::append_drive_request(const DriveRequest &drive_request){
        this->drive_request_times_.push_back(this->get_current_time_in_milliseconds_());
        this->drive_requests_.push_back(drive_request);
    };

    void LogWriter::append_drive_response(const DriveResponse &drive_response){
        this->drive_response_times_.push_back(this->get_current_time_in_milliseconds_());
        this->drive_responses_.push_back(drive_response);
    };

    void LogWriter::write_log_to_file(std::string file_path){
        json log;

        log["location_request"] = this->loc_request_.front().body_str();
        log["location_response"] = this->loc_response_.front().body_str();

        log["location_request_time"] = this->loc_request_time_;
        log["location_response_time"] = this->loc_response_time_;

        log["initialize_request"] = this->init_request_.front().body_str();
        log["initialize_response"] = this->init_response_.front().body_str();

        log["initialize_request_time"] = this->init_request_time_;
        log["initialize_response_time"] = this->init_response_time_;

        std::vector<std::string> req_vec;
        for (auto& req : this->drive_requests_) { 
            req_vec.push_back(req.body_str());
        } 
        log["drive_requests"] = req_vec;

        std::vector<std::string> res_vec;
        for (auto& res : this->drive_responses_) { 
            res_vec.push_back(res.body_str());
        } 
        log["drive_responses"] = res_vec;


        log["drive_request_times"] = this->drive_request_times_;
        log["drive_response_times"] = this->drive_response_times_;

        // log.dump()
        std::ofstream o(file_path);
        o << std::setw(4) << log << std::endl;
    };
}