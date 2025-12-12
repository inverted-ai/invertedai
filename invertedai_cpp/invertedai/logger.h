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

namespace invertedai {

    // a struct to hold simulation information
    struct ScenarioLog {

    };

    class LogReader {
        private:
            std::string location;
            int scenario_length;
            int num_agents;
            // std::optional<std::map<std::string, std::string>> traffic_lights_states;
            LightRecurrentState initial_light_recurr_state;
            std::vector<AgentProperties> sorted_agent_properties;
            std::vector<std::vector<AgentState>> agent_states_over_time;
            std::optional<std::vector<std::map<std::string, std::string>>> traffic_light_states_over_time;
        public:
            void read_log(const std::string &file_path);
            std::string get_location();
            int get_total_num_agents();
            int get_scenario_length();
            std::vector<AgentProperties> get_agent_properties();
            std::vector<std::vector<AgentState>> get_agent_states_over_time();
            std::optional<std::vector<std::map<std::string, std::string>>> get_traffic_lights_states_over_time();

            // std::shared_ptr<LocationInfoRequest> get_location_info_request(int index);
            // std::shared_ptr<LocationInfoResponse> get_location_info_response(int index);
            // std::shared_ptr<InitializeRequest> get_initialize_request(int index);
            // std::shared_ptr<InitializeResponse> get_initialize_response(int index);
            // std::shared_ptr<DriveRequest> get_drive_request(int index);
            // std::shared_ptr<DriveResponse> get_drive_response(int index);


    };
    class LogWriter {
        private:
            std::vector<std::string> loc_requests_;
            std::vector<std::string> loc_responses_;
            std::vector<std::string> loc_request_timestamps_;
            std::vector<std::string> loc_response_timestamps_;

            std::vector<std::string> init_requests_;
            std::vector<std::string> init_responses_;
            std::vector<std::string> init_request_timestamps_;
            std::vector<std::string> init_response_timestamps_;

            std::vector<std::string> drive_requests_;
            std::vector<std::string> drive_responses_;
            std::vector<std::string> drive_request_timestamps_;
            std::vector<std::string> drive_response_timestamps_;

            std::string get_current_time_UTC_();

        public:

            void append_request(const std::string &req, const std::string &mode);

            void append_response(const std::string &res, const std::string &mode);

            void write_log_to_file(const std::string &file_path);
    };
}

#endif