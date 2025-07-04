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
#include <map>
#include <iostream>   // For input/output operations (std::cout, std::cerr, std::cin)
#include <fstream>    // For file stream operations (std::ifstream, std::ofstream)

namespace invertedai {

    // Structure to hold vehicle dimension data.
    // 'rearAxisOffset' is the distance from the rear axle to the rear bumper.
    struct VehicleDimensions {
        double length;
        double width;
        double height;
        double rearAxisOffset;
    };

    // Structure to hold vehicle state at a specific time step.
    // 'orientation' is in radians, 'speed' is in meters per second.
    struct VehicleState {
        double x;
        double y;
        double orientation;
        double speed;
    };

    // Main structure to hold all parsed simulation data.
    // 'vehicleDefs' maps vehicle IDs to their dimensions.
    // 'timeSteps' maps timestamps (in ms) to a map of vehicle IDs and their states at that time.
    // 'sortedTimestamps' keeps track of the order of timestamps for chronological processing.
    struct SimulationData {
        std::string mapName;
        std::string mapFile;
        std::map<std::string, VehicleDimensions> vehicleDefs;
        std::map<int, std::map<std::string, VehicleState>> timeSteps;
        std::vector<int> sortedTimestamps;
    };

    std::string get_current_time_UTC_();

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

        public:

            void append_request(const std::string &req, const std::string &mode);

            void append_response(const std::string &res, const std::string &mode);

            void write_log_to_file(const std::string &file_path);
    };

    class LogParser {
        private:
            std::vector<invertedai::AgentProperties> agent_properties_;
            std::vector<std::vector<invertedai::AgentState>> all_agent_states_;

            std::string map_name_;
            int drive_random_seed_;
            std::string drive_model_version_;

        public:
            LogParser(const std::string &body_str);

            void generate_OpenSCENARIO(const SimulationData& data, const std::string& outputFile)
    };
}

#endif