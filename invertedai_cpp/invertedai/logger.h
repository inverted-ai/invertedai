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
            std::vector<std::string> loc_requests_;
            std::vector<std::string> loc_responses_;
            std::vector<UL> loc_request_times_;
            std::vector<UL> loc_response_times_;

            std::vector<std::string> init_requests_;
            std::vector<std::string> init_responses_;
            std::vector<UL> init_request_times_;
            std::vector<UL> init_response_times_;

            std::vector<std::string> drive_requests_;
            std::vector<std::string> drive_responses_;
            std::vector<UL> drive_request_times_;
            std::vector<UL> drive_response_times_;

            UL get_current_time_in_milliseconds_();

        public:

            void append_request(const std::string &req, const std::string &mode);

            void append_response(const std::string &res, const std::string &mode);

            void write_log_to_file(const std::string &file_path);
    };
}

#endif