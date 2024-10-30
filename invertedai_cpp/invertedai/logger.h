#ifndef INVERTEDAI_LOGGER_H
#define INVERTEDAI_LOGGER_H

#include "drive_request.h"
#include "drive_response.h"
#include "initialize_request.h"
#include "initialize_response.h"
#include "location_info_request.h"
#include "location_info_response.h"

#include <string>
#include <memory>

typedef unsigned long UL;

namespace invertedai {

    class LogWriter {
        private:
            // std::vector<LocationInfoRequest> loc_requests_;
            // std::vector<LocationInfoResponse> loc_responses_;
            std::vector<std::string> loc_requests_;
            std::vector<std::string> loc_responses_;
            std::vector<UL> loc_request_times_;
            std::vector<UL> loc_response_times_;

            // std::vector<InitializeRequest> init_requests_;
            // std::vector<InitializeResponse> init_responses_;
            std::vector<std::string> init_requests_;
            std::vector<std::string> init_responses_;
            std::vector<UL> init_request_times_;
            std::vector<UL> init_response_times_;

            // std::vector<DriveRequest> drive_requests_;
            // std::vector<DriveResponse> drive_responses_;
            std::vector<std::string> drive_requests_;
            std::vector<std::string> drive_responses_;
            std::vector<UL> drive_request_times_;
            std::vector<UL> drive_response_times_;

            UL get_current_time_in_milliseconds_();

            // template <typename T> std::vector<std::string> write_iai_data_vector_to_log_(std::vector<T> data);

        public:

            // void append_loc_request(const LocationInfoRequest &loc_request);

            // void append_loc_response(const LocationInfoResponse &loc_response);

            // void append_init_request(const InitializeRequest &init_request);

            // void append_init_response(const InitializeResponse &init_response);

            // void append_drive_request(const DriveRequest &drive_request);

            // void append_drive_response(const DriveResponse &drive_response);

            void append_request(const std::string &req, const std::string &mode);

            void append_response(const std::string &res, const std::string &mode);

            void write_log_to_file(const std::string &file_path);
    };
}

#endif