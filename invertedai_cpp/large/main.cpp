#include <iostream>
#include <vector>
#include "large_initialize.h"
#include "common.h"
#include "initialize_response.h"
#include <cstdlib>
#include <stdlib.h>
#include "large_initialize.cc"
#include "large_initialize.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/api.h"
#include "large_helpers.h"
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "larger_helpers.cc"


int main() {

    // --- Arguments (replace with CLI parser if needed)
    std::string location = "iai:10th_and_dunbar";
    int fov = 100;
    int num_agents = 40;
    int width = 100;
    int height = 100;
    std::optional<std::string> model_version_drive = std::nullopt; // like Python’s None
    bool save_sim = true;

    // --- Random seeds
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10000);
    int initialize_seed = dis(gen);
    int drive_seed = dis(gen);

    std::optional<unsigned int> random_seed = std::nullopt; // Define random_seed
    std::mt19937 rng(random_seed.value_or(std::random_device{}()));
    // session configuration
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key("wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni");           // later we need to create am argument for this session
    session.connect();

    // --- Get default regions
    std::cout << "Begin initialization.\n";
    std::vector<invertedai::Region> regions = get_regions_default(
        location,
        num_agents,               // total_num_agents
        std::nullopt,             // agent_count_dict
        std::pair<float,float>{(float)width, (float)height}, // area_shape
        std::pair<float,float>{0.0f, 0.0f}, // map_center
        initialize_seed,          // random_seed
        true
                                           
        // display_progress_bar
    );

    // --- Large initialize
    invertedai::LargeInitializeConfig cfg;
    cfg.location = location;
    cfg.regions = regions;
    cfg.random_seed = initialize_seed;
    cfg.get_infractions = true;

    invertedai::InitializeResponse response = large_initialize(cfg);

    // --- Results
    int total_num_agents = response.agent_states().size();
    std::cout << "Number of agents in simulation: " << total_num_agents << "\n";

    return 0;

}
