#include "logger.h"
#include "externals/json.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <time.h>

using json = nlohmann::json;

namespace invertedai {

    std::string get_current_time_UTC_(){
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        std::time_t now_t = std::time(0);
        std::tm* now_tm = std::gmtime(&now_t);
        char buf[42];
        std::strftime(buf, 42, "%Y-%m-%d_%H:%M:%S", now_tm);
        std::string date_time = buf;
        date_time += ":" + std::to_string(milliseconds % 1000) + "_UTC";

        return date_time;
    };

    void LogWriter::append_request(const std::string &req, const std::string &mode){
        std::string date_time = get_current_time_UTC_();

        if (mode == "location_info"){
            this->loc_request_timestamps_.push_back(date_time);
            this->loc_requests_.push_back(req);
        }
        else if (mode == "initialize"){
            this->init_request_timestamps_.push_back(date_time);
            this->init_requests_.push_back(req);
        }
        else if (mode == "drive"){
            this->drive_request_timestamps_.push_back(date_time);
            this->drive_requests_.push_back(req);
        }
    };

    void LogWriter::append_response(const std::string &res, const std::string &mode){
        std::string date_time = get_current_time_UTC_();

        if (mode == "location_info"){
            this->loc_response_timestamps_.push_back(date_time);
            this->loc_responses_.push_back(res);
        }
        else if (mode == "initialize"){
            this->init_response_timestamps_.push_back(date_time);
            this->init_responses_.push_back(res);
        }
        else if (mode == "drive"){
            this->drive_response_timestamps_.push_back(date_time);
            this->drive_responses_.push_back(res);
        }
    };

    void LogWriter::write_log_to_file(const std::string &dir_path){
        json log;

        log["location_requests"] = this->loc_requests_;
        log["location_responses"] = this->loc_responses_;

        log["location_request_timestamps"] = this->loc_request_timestamps_;
        log["location_response_timestamps"] = this->loc_response_timestamps_;

        log["initialize_requests"] = this->init_requests_;
        log["initialize_responses"] = this->init_responses_;

        log["initialize_request_timestamps"] = this->init_request_timestamps_;
        log["initialize_response_timestamps"] = this->init_response_timestamps_;

        log["drive_requests"] = this->drive_requests_;
        log["drive_responses"] = this->drive_responses_;

        log["drive_request_timestamps"] = this->drive_request_timestamps_;
        log["drive_response_timestamps"] = this->drive_response_timestamps_;

        std::string file_path = "iai_log_" + get_current_time_UTC_() + ".json";
        std::string full_path = dir_path + file_path;

        std::cout << "INFO: IAI Log written to path: " << full_path << std::endl;

        std::ofstream o(full_path);
        o << std::setw(4) << log << std::endl;
    };

    LogParser::LogParser(const std::string& file_path) {
        std::ifstream log_file(file_path);
        json json_log_file = json::parse(log_file);

        this->map_name_ = json_log_file["location"]["identifier"];
        his->drive_random_seed_ = json_log_file["drive_random_seed"];
        his->drive_model_version_ = json_log_file["drive_model_version"];
        
        for (int i = 0; i < json_log_file["scenario_length"]; i++) {
            std::vector<invertedai::AgentState> ts_agent_states;
            std::string index = std::to_string(i);
            for (const auto& agent_states : json_log_file["predetermined_agents"].items()) {
                if (i == 0) {
                    invertedai::AgentProperties properties;
                    properties.length = agent_states["static_attributes"]["length"];
                    properties.width = agent_states["static_attributes"]["length"];
                    properties.rear_axis_offset = agent_states["static_attributes"]["rear_axis_offset"];
                    properties.agent_type = agent_states["entity_type"];
                    this->agent_properties_.push_back(properties);
                }
                
                if (agent_states.value()["states"].contains(index)) {
                    json state_ts = agent_states.value()["states"][index];
                    invertedai::AgentState agent_state;
                    agent_state.x = state_ts["center"]["x"];
                    agent_state.y = state_ts["center"]["y"];
                    agent_state.orientation = state_ts["orientation"];
                    agent_state.speed = state_ts["speed"];
                    ts_agent_states.push_back(agent_state);
                } 
            }
            this->all_agent_states_.push_back(ts_agent_states);
        }
    };

    /**
    * @brief Generates an OpenSCENARIO 1.3 XML file from the parsed simulation data.
    *
    * This function constructs the XML by writing directly to an output file stream.
    * It defines vehicles, their initial positions, and then creates events for each
    * time step to update vehicle positions and speeds.
    *
    * @param outputFile The path where the OpenSCENARIO XML file will be saved.
    */
    void LogParser::generate_OpenSCENARIO(const std::string& outputFile) {
        std::ofstream outfile(outputFile);
        // Check if the output file could be opened successfully.
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open output file " << outputFile << std::endl;
            return;
        }

        // Set floating-point precision for XML output to 3 decimal places.
        outfile << std::fixed << std::setprecision(3);

        // Write the XML header and basic OpenSCENARIO structure.
        outfile << R"(<?xml version="1.0" encoding="UTF-8"?>
        <OpenSCENARIO xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="OpenSCENARIO-1.3.xsd">
            <FileHeader revMajor="1" revMinor="3" date=")" << get_current_time_UTC_() << R"(" description="Traffic simulation data converted to OpenSCENARIO" author="TrafficConverter"/>

            <ParameterDeclarations/>

            <CatalogLocations/>

            <RoadNetwork>
                <LogicFile filepath=")" << this->map_name_ << R"("/>
                <SceneGraphFile filepath=""/>
            </RoadNetwork>

            <Entities>
        )";

        // Define each vehicle as a ScenarioObject.
        for (int i = 0; i < this->agent_properties_.size(); i++) {
            const std::string& vehicleId = std::to_string(i);
           
            VehicleDimensions dims_mut;
            dims_mut.length = this->agent_properties_.at(i).length;
            dims_mut.width = this->agent_properties_.at(i).width;
            dims_mut.height = this->agent_properties_.at(i).width;
            dims_mut.rearAxisOffset = this->agent_properties_.at(i).rear_axis_offset;
            const VehicleDimensions& dims = dims_mut;

            // Calculate the center of the bounding box relative to the vehicle's origin (rear axle).
            // OpenSCENARIO's BoundingBox Center is relative to the origin of the entity.
            // If the entity origin is the rear axle:
            // x_center = distance from rear axle to center of vehicle length
            // y_center = 0 (assuming center of width)
            // z_center = half of vehicle height (assuming ground is z=0)
            double center_x = dims.length / 2.0 - dims.rearAxisOffset;
            double center_y = 0.0;
            double center_z = dims.height / 2.0;

            outfile << "        <ScenarioObject name=\"" << vehicleId << "\">\n";
            outfile << "            <Vehicle name=\"" << vehicleId << "\" vehicleCategory=\"car\" model=\"default\">\n";
            outfile << "                <BoundingBox>\n";
            outfile << "                    <Center x=\"" << center_x << "\" y=\"" << center_y << "\" z=\"" << center_z << "\"/>\n";
            outfile << "                    <Dimensions width=\"" << dims.width << "\" length=\"" << dims.length << "\" height=\"" << dims.height << "\"/>\n";
            outfile << "                </BoundingBox>\n";
            outfile << "                <Axles>\n";
            // Front Axle: positionX is distance from vehicle origin (rear axle) to front axle.
            // Rear Axle: positionX is distance from vehicle origin (rear axle) to rear axle (which is 0.0).
            // Default wheelDiameter and trackWidth are used as typical values.
            outfile << "                    <FrontAxle maxSteering=\"" << M_PI / 4.0 << "\" wheelDiameter=\"0.6\" trackWidth=\"" << dims.width * 0.9 << "\" positionX=\"" << dims.length - dims.rearAxisOffset << "\" positionZ=\"0.3\"/>\n";
            outfile << "                    <RearAxle maxSteering=\"0\" wheelDiameter=\"0.6\" trackWidth=\"" << dims.width * 0.9 << "\" positionX=\"0.0\" positionZ=\"0.3\"/>\n";
            outfile << "                </Axles>\n";
            outfile << "                <Properties>\n";
            outfile << "                    <Property name=\"mass\" value=\"1500\"/>\n"; // Default mass for the vehicle.
            outfile << "                </Properties>\n";
            outfile << "            </Vehicle>\n";
            outfile << "        </ScenarioObject>\n";
        }

        outfile << "    </Entities>\n\n";
        outfile << "    <Storyboard>\n";
        outfile << "        <Init>\n";
        outfile << "            <Actions>\n";

        // Set initial positions for all vehicles at the first timestamp (time 0).
        for (int i = 0; i < this->all_agent_states_.at(0).size(); i++) {
            const std::string& vehicleId = std::to_string(i);
            
            VehicleState initialState_mut;
            initialState_mut.x = this->all_agent_states_.at(0).at(i).x;
            initialState_mut.y = this->all_agent_states_.at(0).at(i).y;
            initialState_mut.orientation = this->all_agent_states_.at(0).at(i).orientation;
            initialState_mut.speed = this->all_agent_states_.at(0).at(i).speed;
            const VehicleState& initialState = initialState_mut;

            outfile << "                <Private entityRef=\"" << vehicleId << "\">\n";
            outfile << "                    <PrivateAction>\n";
            outfile << "                        <TeleportAction>\n";
            outfile << "                            <Position>\n";
            outfile << "                                <WorldPosition x=\"" << initialState.x << "\" y=\"" << initialState.y << "\" z=\"0.0\" heading=\"" << initialState.orientation << "\"/>\n";
            outfile << "                            </Position>\n";
            outfile << "                        </TeleportAction>\n";
            outfile << "                    </PrivateAction>\n";
            outfile << "                </Private>\n";
        }
        outfile << "            </Actions>\n";
        outfile << "        </Init>\n\n";

        outfile << "        <Story name=\"TrafficSimulationStory\">\n";
        outfile << "            <Act name=\"TrafficSimulationAct\">\n";

        // Create a ManeuverGroup and Event for each subsequent time step (excluding the initial state).
        // The initial state is handled by the <Init> block.
        for (int i = 1; i < this->all_agent_states_.size(); i++) {
            int timestamp_ms = i * 100;
            double time_s = static_cast<double>(timestamp_ms) / 1000.0; // Convert milliseconds to seconds.

            outfile << "                <ManeuverGroup name=\"TimeStep_" << timestamp_ms << "_Group\" maximumExecutionCount=\"1\">\n";
            outfile << "                    <Actors selectTriggeringEntities=\"false\">\n";
            outfile << "                        <EntityRef entityRef=\"all\"/>\n"; // This group applies to all entities.
            outfile << "                    </Actors>\n";
            outfile << "                    <Maneuver name=\"TimeStep_" << timestamp_ms << "_Maneuver\">\n";
            outfile << "                        <Event name=\"TimeStep_" << timestamp_ms << "_Event\" priority=\"overwrite\" maximumExecutionCount=\"1\">\n";
            outfile << "                            <Action name=\"TimeStep_" << timestamp_ms << "_Actions\">\n";

            // Add individual PrivateActions for each vehicle at the current timestamp.
            for (int j = 0; j < this->all_agent_states_.at(i).size(); j++) {
                const std::string& vehicleId = std::to_string(j);
                
                VehicleState state_mut;
                state_mut.x = this->all_agent_states_.at(i).at(j).x;
                state_mut.y = this->all_agent_states_.at(i).at(j).y;
                state_mut.orientation = this->all_agent_states_.at(i).at(j).orientation;
                state_mut.speed = this->all_agent_states_.at(i).at(j).speed;
                const VehicleState& state = state_mut;

                outfile << "                                <Private entityRef=\"" << vehicleId << "\">\n";
                outfile << "                                    <PrivateAction>\n";
                outfile << "                                        <TeleportAction>\n"; // Update position and heading.
                outfile << "                                            <Position>\n";
                outfile << "                                                <WorldPosition x=\"" << state.x << "\" y=\"" << state.y << "\" z=\"0.0\" heading=\"" << state.orientation << "\"/>\n";
                outfile << "                                            </Position>\n";
                outfile << "                                        </TeleportAction>\n";
                outfile << "                                    </PrivateAction>\n";
                outfile << "                                    <PrivateAction>\n";
                outfile << "                                        <LongitudinalAction>\n";
                outfile << "                                            <SpeedAction>\n";
                outfile << "                                                <AbsoluteTargetSpeed value=\"" << state.speed << "\"/>\n";
                // TransitionDynamics with dynamicsShape="step" means the speed changes instantaneously.
                outfile << "                                                <TransitionDynamics dynamicsShape=\"step\" value=\"0.0\" dynamicsDimension=\"time\"/>\n";
                outfile << "                                            </SpeedAction>\n";
                outfile << "                                        </LongitudinalAction>\n";
                outfile << "                                    </PrivateAction>\n";
                outfile << "                                </Private>\n";
            }
            outfile << "                            </Action>\n";
            outfile << "                            <StartTrigger>\n";
            outfile << "                                <ConditionGroup>\n";
            outfile << "                                    <Condition name=\"TimeCondition_" << timestamp_ms << "\" delay=\"0\" conditionEdge=\"rising\">\n";
            outfile << "                                        <ByValueCondition>\n";
            // Trigger the event when simulation time is greater than the current time step.
            outfile << "                                            <SimulationTimeCondition value=\"" << time_s << "\" rule=\"greaterThan\"/>\n";
            outfile << "                                        </ByValueCondition>\n";
            outfile << "                                    </Condition>\n";
            outfile << "                                </ConditionGroup>\n";
            outfile << "                            </StartTrigger>\n";
            outfile << "                        </Event>\n";
            outfile << "                    </Maneuver>\n";
            outfile << "                </ManeuverGroup>\n";
        }

        outfile << "            </Act>\n";
        outfile << "        </Story>\n";
        outfile << "        <StopTrigger/>\n"; // An optional stop trigger for the scenario.
        outfile << "    </Storyboard>\n";
        outfile << "</OpenSCENARIO>\n";

        outfile.close(); // Close the output file.
    }
}