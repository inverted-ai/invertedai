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

typedef unsigned long ul;

namespace invertedai {

    class LogWriter {
        private:
            std::vector<LocationInfoRequest> loc_request_;
            std::vector<LocationInfoResponse> loc_response_;
            ul loc_request_time_;
            ul loc_response_time_;

            std::vector<InitializeRequest> init_request_;
            std::vector<InitializeResponse> init_response_;
            ul init_request_time_;
            ul init_response_time_;

            std::vector<DriveRequest> drive_requests_;
            std::vector<DriveResponse> drive_responses_;
            std::vector<ul> drive_request_times_;
            std::vector<ul> drive_response_times_;

            ul get_current_time_in_milliseconds_();

        public:

            void set_loc_request(const LocationInfoRequest &loc_request);

            void set_loc_response(const LocationInfoResponse &loc_response);

            void set_init_request(const InitializeRequest &init_request);

            void set_init_response(const InitializeResponse &init_response);

            void append_drive_request(const DriveRequest &drive_request);

            void append_drive_response(const DriveResponse &drive_response);

            void write_log_to_file(std::string file_path);
    };
}

#endif