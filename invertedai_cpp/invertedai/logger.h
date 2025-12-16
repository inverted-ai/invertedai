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

using json = nlohmann::json; // from <json.hpp>
namespace invertedai {

    // a struct to hold simulation information
    class ScenarioLog {
        public:
            std::vector<std::vector<AgentState>> agent_states;
            std::vector<AgentProperties> agent_properties;
            std::optional<std::vector<std::map<std::string, std::string>>> traffic_lights_states;

            std::string location;
            std::optional<std::pair<double,double>> rendering_center;
            std::optional<int> rendering_fov;

            std::optional<int> lights_random_seed;
            std::optional<int> initialize_random_seed;
            std::optional<int> drive_random_seed;

            std::optional<std::string> initialize_model_version = std::string("best");
            std::optional<std::string> drive_model_version = std::string("best");

            // recurrent
            std::optional<LightRecurrentState>  light_recurrent_states;
            std::optional<std::vector<RecurrentState>> recurrent_states;

            // Persistent waypoints for each agent (Python: waypoints)
            std::optional<std::map<std::string, std::vector<Point2d>>> waypoints;
            // Per-frame waypoints (Python: waypoints_per_frame)
            std::optional<std::vector<std::map<int, Point2d>>> waypoints_per_frame;
            // Active agents per timestep 
            std::vector<std::vector<int>> present_indexes;
            ScenarioLog(
                std::vector<std::vector<AgentState>> agent_states_,
                std::vector<AgentProperties> agent_properties_,
                std::optional<std::vector<std::map<std::string, std::string>>> traffic_states_,
                std::string location_,
                std::optional<std::pair<double,double>> rendering_center_,
                std::optional<int> rendering_fov_,
        
                std::optional<int> lights_seed_,
                std::optional<int> init_seed_,
                std::optional<int> drive_seed_,
        
                std::optional<std::string> init_version_,
                std::optional<std::string> drive_version_,
        
                std::optional<LightRecurrentState> light_states_,
                std::optional<std::vector<RecurrentState>> recurrent_states_,
                std::optional<std::map<std::string, std::vector<Point2d>>> waypoints_,
                std::vector<std::vector<int>> present_indexes_
            );
            void add_time_step_data(
                std::vector<AgentState> current_states,
                std::vector<int> current_present
            );

        private:
            void validate_states_and_present_indexes_init();
            void validate_states_and_present_indexes_time_step(
                const std::vector<AgentState>&,
                const std::vector<int>&
            );
    };

    class LogReader {
        private:
            std::string location;
            ScenarioLog scenario_log_;              // current working copy
            ScenarioLog scenario_log_original_;     // immutable original
            std::optional<LocationInfoResponse> location_info_response_;
            std::optional<std::string> initialize_model_version_;
            std::optional<std::string> drive_model_version_;

            std::vector<AgentState> agent_states;
            std::vector<AgentProperties> agent_properties;
            std::optional<std::map<std::string, std::string>> traffic_lights_states;
            std::optional<LightRecurrentState> light_recurrent_states;
            std::optional<std::vector<RecurrentState>> recurrent_states;
            int current_timestep = 0;

            int simulation_length;
            int num_agents;
            // std::optional<std::map<std::string, std::string>> traffic_lights_states;
            LightRecurrentState initial_light_recurr_state;
            std::vector<AgentProperties> sorted_agent_properties;
            std::vector<std::vector<AgentState>> agent_states_over_time;
            std::optional<std::vector<std::map<std::string, std::string>>> traffic_light_states_over_time;
            std::optional<std::map<std::string, std::vector<Point2d>>> waypoints;
        public:
            void read_log(const std::string &file_path, std::string API_KEY);

            bool initialize(); 
            bool drive();
            void reset_log();
            bool return_last_state();
            bool return_state_at_timestep(int t);

            std::string get_location();
            int get_total_num_agents();
            int get_scenario_length();
            std::vector<AgentProperties> get_agent_properties();
            std::vector<std::vector<AgentState>> get_agent_states_over_time();
            std::optional<std::vector<std::map<std::string, std::string>>> get_traffic_lights_states_over_time();

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