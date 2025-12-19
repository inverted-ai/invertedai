#ifndef INVERTEDAI_LOGGER_H
#define INVERTEDAI_LOGGER_H

#include "drive_request.h"
#include "drive_response.h"
#include "initialize_request.h"
#include "initialize_response.h"
#include "location_info_request.h"
#include "location_info_response.h"
#include <iostream>

namespace invertedai {

    // a class to hold LogReader information
    struct ScenarioLog {
        public:
            ScenarioLog() = default;
            std::string location;

            std::vector<std::vector<AgentState>> agent_states;
            std::vector<AgentProperties> agent_properties;
            std::optional<std::vector<std::map<std::string, std::string>>> traffic_lights_states;

            std::optional<std::pair<double,double>> rendering_center;
            std::optional<int> rendering_fov;

            std::optional<int> lights_random_seed;
            std::optional<int> initialize_random_seed;
            std::optional<int> drive_random_seed;

            std::optional<std::string> initialize_model_version = std::string("best");
            std::optional<std::string> drive_model_version = std::string("best");

            std::optional<std::vector<LightRecurrentState>>  light_recurrent_states;
            std::optional<std::vector<std::vector<double>>> recurrent_states;
            std::optional<std::map<std::string, std::vector<Point2d>>> waypoints;
            std::optional<std::vector<std::map<int, Point2d>>> waypoints_per_frame;
            std::vector<std::vector<int>> present_indexes;
            ScenarioLog(
                std::string location_,
                std::vector<std::vector<AgentState>> agent_states_,
                std::vector<AgentProperties> agent_properties_,
                std::optional<std::vector<std::map<std::string, std::string>>> traffic_states_,
                std::optional<std::pair<double,double>> rendering_center_,
                std::optional<int> rendering_fov_,
        
                std::optional<int> lights_seed_,
                std::optional<int> init_seed_,
                std::optional<int> drive_seed_,
        
                std::optional<std::string> init_version_,
                std::optional<std::string> drive_version_,
        
                std::optional<std::vector<LightRecurrentState>> light_states_,
                std::optional<std::vector<std::vector<double>>> recurrent_states_,
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

    class ScenarioLogReader {
        private:
            ScenarioLog scenario_log_;
            int current_timestep = 0;
            int simulation_length;
        public:
            explicit ScenarioLogReader(const std::string &file_path);
            const std::string& get_location() const;
            std::optional<std::map<std::string, std::string>> current_traffic_lights() const;
            std::vector<AgentState> current_agent_states() const;
            std::vector<AgentProperties> current_agent_properties() const;
            std::optional<std::vector<LightRecurrentState>> current_light_recurrent_state() const;
            std::optional<std::vector<std::vector<double>>> current_recurrent_states() const;
            bool initialize(); 
            bool drive();
            void reset_log();
            bool return_last_state();
            bool return_state_at_timestep(int t);
            std::optional<int> get_fov();
            std::optional<std::pair<double,double>> get_rendering_center();
            int get_scenario_length();
            ScenarioLog get_scenario_log();
            std::vector<AgentProperties> get_agent_properties();
            std::vector<std::vector<AgentState>> get_agent_states_over_time();
            std::optional<std::vector<std::map<std::string, std::string>>> get_traffic_lights_states_over_time();
    };

    class ScenarioLogWriter {
        private:
            ScenarioLog scenario_log_;
            nlohmann::json output_dict;
            int simulation_length;
            //count agent types
            std::pair<int, int> count_agent_types() const;
            // count control types from static actors
            std::tuple<int, int, int, int> count_control_types(const LocationInfoResponse& location_info_response) const;
            nlohmann::json build_individual_suggestions_dict(const ScenarioLog& log) const;
            nlohmann::json build_predetermined_agents_dict(const ScenarioLog& log) const;
            nlohmann::json build_predetermined_controls_dict(
                const ScenarioLog& log,
                const LocationInfoResponse& location_info_response
            ) const;
        public:
            //scenario_log Optional external scenario log to export (if null, uses internal log)
            ScenarioLogWriter();
            void export_to_file(
                const std::string& log_path,
                std::optional<ScenarioLog> scenario_log = std::nullopt,
                std::optional<LocationInfoResponse> location_info_response = std::nullopt
            );
            //Initialize the log writer with initial simulation data
            void initialize(
                std::optional<std::string> location = std::nullopt,
                std::optional<LocationInfoResponse> location_info_response = std::nullopt,
                std::optional<InitializeResponse> init_response = std::nullopt,
                std::optional<int> lights_random_seed = std::nullopt,
                std::optional<int> initialize_random_seed = std::nullopt,
                std::optional<int> drive_random_seed = std::nullopt,
                std::optional<std::string> drive_model_version = std::nullopt,
                std::optional<int> fov = std::nullopt,
                std::optional<ScenarioLog> scenario_log = std::nullopt
            );
            //Add a drive response to the log
            void drive(
                const DriveResponse& drive_response,
                std::optional<std::vector<int>> current_present_indexes = std::nullopt,
                std::optional<std::vector<AgentProperties>> new_agent_properties = std::nullopt,
                std::optional<std::map<int, std::optional<Point2d>>> waypoints = std::nullopt
            );
            //Get the indexes of agents currently present in the simulation
            std::vector<int> current_present_indexes() const;
            // get all agent props
            std::vector<AgentProperties> all_agent_properties() const;
    };
}

#endif