#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <optional>
#include <opencv2/opencv.hpp>  // OpenCV for image handling

#include "invertedai/api.h"
#include "invertedai/session.h"
#include "invertedai/location_info_request.h"
#include "invertedai/location_info_response.h"
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "large_initialize.h"
#include "large_helpers.h"
#include "common.h"

using namespace invertedai;

int main() {
    // --- Arguments
    std::string location = "iai:10th_and_dunbar";
    int num_agents = 40;
    int width = 100;
    int height = 100;

    // --- Random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    int initialize_seed = std::uniform_int_distribution<>(1, 10000)(gen);

    // --- Session setup
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key("");  // 🔑 replace with your key
    session.connect();

    // --- Regions
    std::cout << "Generating default regions...\n";
    auto regions = get_regions_default(
        location,
        num_agents,
        session,
        std::pair<float,float>{width/2.0f, height/2.0f},  // area_shape
        {0.0f, 0.0f},                                     // map_center
        std::nullopt,                                     // random_seed
        true                                              // show progress
    );

    // --- Large initialize
    invertedai::LargeInitializeConfig cfg;
    cfg.location = location;
    cfg.regions = regions;
    cfg.random_seed = initialize_seed;
    cfg.get_infractions = true;

    InitializeResponse response = invertedai::large_initialize(cfg);
    std::cout << "Number of agents: " << response.agent_states().size() << "\n";

    // --- Save initialized images
    std::cout << "Saving initialized region images...\n";
    int idx = 0;
    for (auto& region : regions) {
        // Build request
        invertedai::LocationInfoRequest loc_info_req("{}");
        loc_info_req.set_location(location);
        loc_info_req.set_rendering_center(std::make_optional(std::make_pair(region.center.x, region.center.y)));
        loc_info_req.set_rendering_fov((int)region.size);
        loc_info_req.set_include_map_source(false);

        // Fetch image
        LocationInfoResponse loc_info_res = invertedai::location_info(loc_info_req, &session);
        std::vector<unsigned char> img_bytes = loc_info_res.birdview_image();

        // Decode and save
        cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
        if (!birdview.empty()) {
            std::string filename = "region_" + std::to_string(idx++) + ".png";
            cv::imwrite(filename, birdview);
            std::cout << "Saved " << filename << "\n";
        } else {
            std::cerr << "Failed to decode image for region " << idx << "\n";
        }
    }

    std::cout << "All done!\n";
    return 0;
}
