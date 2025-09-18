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
inline cv::Point world_to_pixel(double x, double y,
    double cx, double cy,
    int img_w, int img_h,
    double fov_meters) {
double scale = img_w / fov_meters;  // pixels per meter
int u = static_cast<int>((x - (cx - fov_meters/2)) * scale);
int v = static_cast<int>((cy + fov_meters/2 - y) * scale);
return cv::Point(u, v);
}
int main() {
    // --- Arguments
    std::string location = "iai:ubc_roundabout";
    int num_agents = 25;
    int width = 1000;
    int height = 1000;

    // --- Random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    int initialize_seed = std::uniform_int_distribution<>(1, 10000)(gen);

    // --- Session setup
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key("wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni");  // api key
    session.connect();
    LocationInfoRequest req("{}");
    req.set_location(location);
    req.set_include_map_source(true);
    std::string request = req.body_str();
    nlohmann::json body_json = nlohmann::json::parse(request);
    std::cerr << "DEBUG req location_info: " << request << "\n";
    for (auto& [key, val] : body_json.items()) {
        if (val.is_null()) {
            std::cerr << "FIELD IS NULL: " << key << "\n";
        }
    }
    

    LocationInfoResponse res = location_info(req, &session);


    // if LocationInfoResponse has a .body_str() or similar:
    std::string raw = res.body_str();
    std::cerr << "DEBUG raw location_info: " << raw << "\n";
    // --- Regions
    std::cout << "Generating default regions...\n";
    auto regions = get_regions_default(
        location,
        num_agents,
        session,
        std::pair<float,float>{width/2.0f, height/2.0f},  // area_shape
        std::make_pair(res.map_origin().x, res.map_origin().y),                                  // map_center
        std::nullopt,                                     // random_seed
        true                                              // show progress
    );

    // --- Large initialize
    invertedai::LargeInitializeConfig cfg;
    cfg.location = location;
    cfg.regions = regions;
    cfg.random_seed = initialize_seed;
    cfg.get_infractions = true;
    cfg.traffic_light_state_history = std::nullopt;
    cfg.display_progress_bar = true;
    cfg.return_exact_agents = false;
    cfg.api_model_version = std::nullopt;

    cfg.agent_properties = std::nullopt;
    cfg.agent_states = std::nullopt;   // let API sample positions
    std::cout << "Calling large_initialize with " << regions.size() << " regions...\n";

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

        for (const auto& agent : response.agent_states()) {
            cv::Point pt = world_to_pixel(
                agent.x, agent.y,
                region.center.x, region.center.y,
                birdview.cols, birdview.rows,
                region.size
            );
            cv::circle(birdview, pt, 5, cv::Scalar(0,0,255), -1);
        }

cv::imwrite("region_with_agents.png", birdview);
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
