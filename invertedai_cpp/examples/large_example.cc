#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <optional>
#include <opencv2/opencv.hpp>

#include "invertedai/api.h"
#include "invertedai/session.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"
#include "large/large_drive/large_drive.h"
#include "large/large_initialize/large_init_helpers.h"
#include "large/visualizer/visualizer.h"
#include "large/visualizer/visualizer_helpers.h"

using namespace invertedai;
static std::unordered_map<std::pair<double,double>, cv::Mat, PairHash> cache_region_tiles_for_drive(
    Session& session,
    const std::string& location,
    const std::vector<Region>& drive_tiles,
    double scale
);
// Simple agent generator for testing
std::pair<std::vector<AgentState>, std::vector<AgentProperties>>
initialize_agents_for_region(
    invertedai::Session& session,
    const std::string& location,
    const Region& region,
    int num_agents,
    int random_seed = 1,
    bool get_birdview = false,
    bool get_infractions = false
);
/*                                                                                 
            HOW TO RUN EXECUTABLE:

            Join docker:
            docker compose build
            docker compose run --rm dev 
            
            bazel build //examples:large_example 
            
            To view the visualizers, run with the --debug flag:
            ./bazel-bin/examples/large_example --debug 

            To turn off the visualizers, run without the --debug flag:
            ./bazel-bin/examples/large_example

            To run modifying all arguments, get_infractions_enabled, and visualizers on:
            ./bazel-bin/examples/large_example --location carla:Town10HD --num_agents 50 --sim_length 100 --width 500 --height 500 --get_infractions --debug

            To get help:
            ./bazel-bin/examples/large_example --help

*/
int main(int argc, char** argv) {
    bool DEBUG_VISUALS = false;

    // default
    std::string location = "carla:Town03";
    int total_num_agents = 10;
    int sim_length = 100;
    int width = 100;
    int height = 100;
    bool get_infractions=false;

    const std::string API_KEY = getenv("IAI_API_KEY"); // in the docker - 'export IAI_API_KEY="your key here"'

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            DEBUG_VISUALS = true;
        } else if (arg == "--location" && i + 1 < argc) {
            location = argv[++i];
        } else if (arg == "--num_agents" && i + 1 < argc) {
            total_num_agents = std::stoi(argv[++i]);
        } else if (arg == "--sim_length" && i + 1 < argc) {
            sim_length = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--get_infractions") {
            get_infractions = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --location <str>        Map location (default: carla:Town03)\n"
                      << "  --num_agents <int>      Number of agents (default: 10)\n"
                      << "  --sim_length <int>      Simulation length (default: 100)\n"
                      << "  --width <int>           Map width in meters (default: 100)\n"
                      << "  --height <int>          Map height in meters (default: 100)\n"
                      << "  --get_infractions       Enable simulation to capture to infractions data (default: false)\n"
                      << "  --debug                 Enable debug visualization mode (default: false)\n";
            return 0;
        }
    }

    bool FLIP_X_FOR_THIS_DOMAIN = false; 
    if (location.rfind("carla:", 0) == 0) {
        FLIP_X_FOR_THIS_DOMAIN = true;
    }

    std::cout << "[INFO] Debug visualization mode: "
    << (DEBUG_VISUALS ? "ON" : "OFF") << "\n";

    // Random seed 
    std::random_device rd;
    std::mt19937 gen(rd());
    int seed = std::uniform_int_distribution<>(1, 1000000)(gen); // or fixed for repeatability 

    // session connection
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(API_KEY);
    session.connect();

    //get map info (for map_center)
    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    LocationInfoResponse li_res = location_info(li_req, &session);

    std::pair<float,float> map_center{
        static_cast<float>(li_res.map_origin().x),
        static_cast<float>(li_res.map_origin().y)
    };

    // generate default regions
    std::map<AgentType,int> agent_count_dict = {
        {AgentType::car, total_num_agents}
    };

    std::cout << "Generating default regions...\n";
    std::vector<Region> regions = get_regions_default(
        location,
        total_num_agents,                              
        agent_count_dict,                              
        session,
        std::pair<float,float>{width/2.f, height/2.f}, 
        map_center,                                    
        seed                               

    );
    std::cout << "Generated " << regions.size() << " regions.\n";

    // set up arguments for large initialize
    LargeInitializeConfig cfg(session);
    cfg.location = location;
    cfg.regions = regions;
    cfg.random_seed = seed;
    cfg.get_infractions = get_infractions;
    cfg.traffic_light_state_history = std::nullopt;
    cfg.return_exact_agents = true;
    cfg.api_model_version = std::nullopt;
    // Optional agent generator per region
    auto [init_states, init_props] =
    initialize_agents_for_region(session, location, regions[0], 3, seed);
    cfg.agent_states = std::nullopt;//init_states; //change to init_states to use the generator function
    cfg.agent_properties = std::nullopt;//init_props; 

    std::cout << "Calling large_initialize with " << regions.size() << " regions...\n";
    std::vector<Region> outputed_regions;
    invertedai::InitializeResponse response = DEBUG_VISUALS
        ? invertedai::large_initialize(cfg, &outputed_regions)
        : invertedai::large_initialize(cfg);
    std::vector<AgentState> agent_states     = response.agent_states();
    std::vector<AgentProperties> agent_props = response.agent_properties();
    std::vector<std::vector<double>> recurrent    = response.recurrent_states();
    std::optional<std::map<std::string, std::string>> traffic_lights_states = response.traffic_lights_states();
    std::optional<std::vector<LightRecurrentState>> light_recurrent_states = response.light_recurrent_states();

    // generate all the tiles required for driving
    std::map<AgentType,int> agent_count_dict_drive = {
        {AgentType::car, 9999} // a lot of agents to initialize every tile 
    };
    std::vector<Region> drive_tiles = get_regions_default(
        location,
        9999,                               //  a lot of agents to initialize every tile 
        agent_count_dict_drive,                              
        session,
        std::pair<float,float>{width/2.f, height/2.f}, 
        map_center,                        // map center from location_info
        seed                               // random seed

    );
    // visualize initialize results 
    if(DEBUG_VISUALS) {
        visualize_large_initialize(cfg.location, cfg.session, outputed_regions, drive_tiles, li_res, FLIP_X_FOR_THIS_DOMAIN);
    }
    
    // time to drive
    std::cout << "Starting simulation for " << sim_length << " steps...\n";
    LargeDriveConfig drive_cfg(session);
    drive_cfg.location = location;
    drive_cfg.api_key = API_KEY;
    drive_cfg.agent_states = agent_states;
    drive_cfg.agent_properties = agent_props;
    drive_cfg.recurrent_states = recurrent;
    drive_cfg.traffic_lights_states = traffic_lights_states;
    drive_cfg.light_recurrent_states = light_recurrent_states;
    drive_cfg.random_seed = seed;
    drive_cfg.get_infractions = true;
    drive_cfg.single_call_agent_limit = 100;
    drive_cfg.async_api_calls = true;
    
    cv::Rect2d bounds = compute_bounds_rect(drive_tiles);
    const double scale = get_render_scale(li_res, drive_tiles.front());
    const int canvas_w = static_cast<int>(std::ceil(bounds.width * scale));
    const int canvas_h = static_cast<int>(std::ceil(bounds.height * scale));
    std::unordered_map<std::pair<double,double>, cv::Mat, PairHash> drive_cached_tiles = cache_region_tiles_for_drive(
        drive_cfg.session, drive_cfg.location, drive_tiles, scale
    );
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
    cv::VideoWriter writer("large_drive_quadtree_sim.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        10, // fps
        cv::Size(canvas_w, canvas_h));
    for (int step = 0; step < sim_length; ++step) {   
        std::vector<Region> leaf_regions;
        invertedai::DriveResponse drive_response = DEBUG_VISUALS ? 
            invertedai::large_drive(drive_cfg, &leaf_regions) : invertedai::large_drive(drive_cfg);
        auto states = drive_response.agent_states();
        auto recur  = drive_response.recurrent_states();
        auto lights_recur = drive_response.light_recurrent_states();
        auto traff  = drive_response.traffic_lights_states();
        drive_cfg.agent_states           = states;
        drive_cfg.recurrent_states       = recur;
        drive_cfg.light_recurrent_states = lights_recur;
        drive_cfg.traffic_lights_states  = std::nullopt;
        drive_cfg.api_model_version      = drive_response.model_version();
        drive_cfg.random_seed            = std::nullopt;

        if(DEBUG_VISUALS) {
        //visualize each drive step
            visualize_large_drive(drive_cfg,
                leaf_regions,
                outputed_regions,
                li_res,
                traff,
                drive_tiles,
                drive_cached_tiles,
                writer,
                FLIP_X_FOR_THIS_DOMAIN,
                step);
                // track some statistics
                int total_agents = drive_response.agent_states().size();
                int num_leaves = leaf_regions.size();
                double avg_agents_per_leaf = double(total_agents) / num_leaves;
                std::cout << "[Step " << step << "] " << num_leaves
                        << " leaves, avg " << avg_agents_per_leaf << " agents/leaf\n";
        }
    }
    writer.release();

    return 0;
}


// struct to later paste the driving tiles based on world coordinates
static std::unordered_map<std::pair<double,double>, cv::Mat, PairHash> cache_region_tiles_for_drive(
    Session& session,
    const std::string& location,
    const std::vector<Region>& drive_tiles,
    double scale
) {
    std::cerr << "Caching " << drive_tiles.size() << " tiles for drive steps...\n";
    std::unordered_map<std::pair<double,double>, cv::Mat, PairHash> drive_cached_tiles;
    for (size_t i = 0; i < drive_tiles.size(); ++i) {
        const Region& r = drive_tiles[i];
        LocationInfoRequest req("{}");
        req.set_location(location);
        req.set_rendering_center(std::make_pair(r.center.x, r.center.y));
        req.set_rendering_fov(static_cast<int>(r.size));
        req.set_include_map_source(false);

        LocationInfoResponse res = location_info(req, &session);
        cv::Mat tile = cv::imdecode(res.birdview_image(), cv::IMREAD_COLOR);
        if (tile.empty()) {
            std::cerr << "[WARN] drive Tile " << i << " is empty.\n";
            continue;
        }

        const int tile_px = static_cast<int>(std::llround(r.size * scale));
        if (tile.cols != tile_px || tile.rows != tile_px) {
            cv::resize(tile, tile, cv::Size(tile_px, tile_px), 0, 0, cv::INTER_LINEAR);
        }

        drive_cached_tiles[{r.center.x, r.center.y}] = tile;

    }
    return drive_cached_tiles;
}

// Simple agent generator for testing
std::pair<std::vector<AgentState>, std::vector<AgentProperties>>
initialize_agents_for_region(
    invertedai::Session& session,
    const std::string& location,
    const Region& region,
    int num_agents,
    int random_seed,
    bool get_birdview,
    bool get_infractions)
{
    using namespace invertedai;

    // Build the API request
    InitializeRequest req("{}");
    req.set_location(location);
    req.set_num_agents_to_spawn(num_agents);
    req.set_location_of_interest(std::make_pair(region.center.x, region.center.y));
    req.set_get_birdview(get_birdview);
    req.set_get_infractions(get_infractions);
    req.set_random_seed(random_seed);

    // Make the API call
    InitializeResponse resp = initialize(req, &session);

    // Extract agent data
    std::vector<AgentState> states = resp.agent_states();
    std::vector<AgentProperties> props = resp.agent_properties();

    std::cout << "[INFO] Initialized " << states.size()
              << " agents in region centered at ("
              << region.center.x << ", " << region.center.y << ")\n";

    return {states, props};
}
