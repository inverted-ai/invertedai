#include "large_drive.h"
#include "invertedai/session.h"
#include "invertedai/drive_response.h"
#include "invertedai/drive_request.h"
#include <iostream>
#include "invertedai/logger.h"
using namespace invertedai;

// bazel build //large:test_large_drive
// ./bazel-bin/large/test_large_drive
int main() {
    try {
        // Create session with API key (from env var)
        // --- Session connection
        boost::asio::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        invertedai::Session session(ioc, ctx);
        session.set_api_key("");
        session.connect();

        // Hardcoded 3 agents, spread apart to trigger quadtree splitting
        std::vector<AgentState> states = {
            AgentState{-50, -50, 0.0, 5.0},   // x, y, orientation, speed
            AgentState{ 50,  50, 1.0, 5.0},
            AgentState{150, 150, 2.0, 5.0}
        };

        std::vector<AgentProperties> props = {
            AgentProperties(json{
                {"agent_type", "car"},
                {"length", 4.5},
                {"width", 2.0},
                {"rear_axis_offset", 1.5},
                {"max_speed", 15.0},
                {"waypoint", {100.0, 100.0}}
            }),
            AgentProperties(json{
                {"agent_type", "car"},
                {"length", 4.7},
                {"width", 2.1},
                {"rear_axis_offset", 1.6},
                {"max_speed", 18.0},
                {"waypoint", {200.0, 200.0}}
            }),
            AgentProperties(json{
                {"agent_type", "car"},
                {"length", 5.0},
                {"width", 2.2},
                {"rear_axis_offset", 1.7},
                {"max_speed", 20.0},
                {"waypoint", {300.0, 300.0}}
            })
        };
        
        // Each car needs valid physical attributes
        // std::vector<AgentAttributes> attrs = {
        //     AgentAttributes{.length=4.5, .width=2.0, .rear_axis_offset=1.5},
        //     AgentAttributes{.length=4.7, .width=2.1, .rear_axis_offset=1.6},
        //     AgentAttributes{.length=5.0, .width=2.2, .rear_axis_offset=1.7}
        // };

        LargeDriveConfig cfg(session);
        // attach logger
        cfg.logger = invertedai::LogWriter();
        cfg.location = "can:ubc_roundabout";   
        cfg.agent_states = states;
        cfg.agent_properties = props;
        cfg.single_call_agent_limit = 1;      // force quadtree to split
        // cfg.agent_attributes = attrs;  // TODO create compatibility shim

        std::cerr << "[INFO] Calling large_drive...\n";
        DriveResponse resp = large_drive(cfg);

        std::cout << "Received " << resp.agent_states().size()
                  << " driven agents." << std::endl;

        for (size_t i = 0; i < resp.agent_states().size(); i++) {
            const auto& s = resp.agent_states()[i];
            std::cout << "Agent " << i << " pos=("
                      << s.x << "," << s.y
                      << ") speed=" << s.speed
                      << " orient=" << s.orientation
                      << std::endl;
        }
        std::cout << "[INFO] large_drive succeeded, writing log" << std::endl;
        cfg.logger.write_log_to_file("/usr/src/myapp/tmp/");

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << std::endl;
        //cfg.logger.write_log_to_file("/usr/src/myapp/tmp/");
        return 1;
    }

    return 0;
}
