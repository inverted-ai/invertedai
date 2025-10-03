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
        // --- Session connection
        boost::asio::io_context ioc;
        ssl::context ctx(ssl::context::tlsv12_client);
        invertedai::Session session(ioc, ctx);
        session.set_api_key("wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni");
        session.connect();

        // --- Step 1: Build InitializeRequest ---
        InitializeRequest init_req("{}");
        init_req.set_location("can:ubc_roundabout");
        init_req.set_num_agents_to_spawn(5);
        init_req.set_get_infractions(true);
        init_req.set_get_birdview(false);

        // --- Call initialize API ---
        std::cerr << "[INFO] Calling initialize...\n";
        InitializeResponse init_resp = initialize(init_req, &session);

        std::cerr << "[INFO] Initialize returned "
                  << init_resp.agent_states().size()
                  << " agents.\n";

        // Debug: print initial agent states
        for (size_t i = 0; i < init_resp.agent_states().size(); i++) {
            const auto& s = init_resp.agent_states()[i];
            std::cout << "Init agent " << i << " pos=("
                      << s.x << "," << s.y
                      << ") speed=" << s.speed
                      << " orient=" << s.orientation
                      << std::endl;
        }

        // --- Step 2: Feed initialize response into LargeDrive ---
        std::cout << "Starting large_drive loop...\n";

        LargeDriveConfig cfg(session);
        cfg.location = "can:ubc_roundabout";
        cfg.agent_states = init_resp.agent_states();
        cfg.agent_properties = init_resp.agent_properties();
        cfg.recurrent_states = init_resp.recurrent_states();
        cfg.light_recurrent_states = init_resp.light_recurrent_states();
        cfg.get_infractions = true;
        cfg.single_call_agent_limit = 1; // or smaller if you want to force splits
        
        for (int step = 0; step < 100; ++step) {
            std::cout << "=== LargeDrive step " << step << " ===\n";
        
            try {
                DriveResponse drive_res = large_drive(cfg);
        
                // Debug print
                if (!drive_res.agent_states().empty()) {
                    const auto &s = drive_res.agent_states()[0];
                    std::cout << "Agent0 pos=(" << s.x << "," << s.y
                              << ") speed=" << s.speed
                              << " orient=" << s.orientation << std::endl;
                }
        
                // Feed outputs back into cfg for the next step
                cfg.agent_states         = drive_res.agent_states();
                cfg.recurrent_states     = drive_res.recurrent_states();
                cfg.light_recurrent_states = drive_res.light_recurrent_states();
                // agent_properties usually stays constant unless you mutate them
            }
            catch (const std::exception &e) {
                std::cerr << "[FATAL] LargeDrive failed at step " << step
                          << ": " << e.what() << std::endl;
                break;
            }
        }
        
        std::cout << "LargeDrive loop finished.\n";
}