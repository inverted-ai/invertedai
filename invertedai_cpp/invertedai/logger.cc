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

    // template <typename T> std::vector<std::string> LogWriter::write_iai_data_vector_to_log_(std::vector<T> data){
    //     std::vector<std::string> output_vec;
    //     for (auto& d : data) { 
    //         output_vec.push_back(d.body_str());
    //     } 

    //     return output_vec;
    // };

    // void LogWriter::append_loc_request(const LocationInfoRequest &loc_request){
    //     this->loc_request_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->loc_requests_.push_back(loc_request);
    // };

    // void LogWriter::append_loc_response(const LocationInfoResponse &loc_response){
    //     this->loc_response_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->loc_responses_.push_back(loc_response);
    // };

    // void LogWriter::append_init_request(const InitializeRequest &init_request){
    //     this->init_request_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->init_requests_.push_back(init_request);
    // };

    // void LogWriter::append_init_response(const InitializeResponse &init_response){
    //     this->init_response_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->init_responses_.push_back(init_response);
    // };

    // void LogWriter::append_drive_request(const DriveRequest &drive_request){
    //     this->drive_request_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->drive_requests_.push_back(drive_request);
    // };

    // void LogWriter::append_drive_response(const DriveResponse &drive_response){
    //     this->drive_response_times_.push_back(this->get_current_time_in_milliseconds_());
    //     this->drive_responses_.push_back(drive_response);
    // };

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

        // log["location_requests"] = this->write_iai_data_vector_to_log_<LocationInfoRequest>(this->loc_requests_);
        // log["location_responses"] = this->write_iai_data_vector_to_log_<LocationInfoResponse>(this->loc_responses_);

        log["location_requests"] = this->loc_requests_;
        log["location_responses"] = this->loc_responses_;

        log["location_request_times"] = this->loc_request_times_;
        log["location_response_times"] = this->loc_response_times_;

        // log["initialize_requests"] = this->write_iai_data_vector_to_log_<InitializeRequest>(this->init_requests_);
        // log["initialize_responses"] = this->write_iai_data_vector_to_log_<InitializeResponse>(this->init_responses_);

        log["initialize_requests"] = this->init_requests_;
        log["initialize_responses"] = this->init_responses_;

        log["initialize_request_times"] = this->init_request_times_;
        log["initialize_response_times"] = this->init_response_times_;

        // log["drive_requests"] = this->write_iai_data_vector_to_log_<DriveRequest>(this->drive_requests_);
        // log["drive_responses"] = this->write_iai_data_vector_to_log_<DriveResponse>(this->drive_responses_);

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