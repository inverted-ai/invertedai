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
// ./bazel-bin/large/large_main
// bazel build //large:large_main 

int main() {
    // --- Arguments
    std::string location = "can:boundary_rd_and_kingsway_canada";
    int num_agents = 50;
    int width = 800;
    int height = 800;

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
    // std::cerr << "DEBUG req location_info: " << request << "\n";
    // for (auto& [key, val] : body_json.items()) {
    //     if (val.is_null()) {
    //         std::cerr << "FIELD IS NULL: " << key << "\n";
    //     }
    // }
    

    LocationInfoResponse res = location_info(req, &session);


    // if LocationInfoResponse has a .body_str() or similar:
    std::string raw = res.body_str();
    std::cerr << "DEBUG raw location_info: " << raw << "\n";
    // --- Regions
    std::cout << "Generating default regions...\n";
    // --- Regions (use agent_count_dict instead of total_num_agents)
    std::cout << "Generating default regions...\n";

    
    std::map<AgentType,int> agent_count_dict = {
        {AgentType::car, num_agents}
    };

    std::pair<float, float> map_center{ 
        static_cast<float>(res.map_origin().x), 
        static_cast<float>(res.map_origin().y) 
    };
    

    auto regions = get_regions_default(
        location,
        num_agents,          // total_num_agents -- (deprecated)
        agent_count_dict,      // NEW: agent_count_dict
        session,
        std::pair<float,float>{width/2.0f, height/2.0f}, // area_shape
        map_center,            // map_center from location_info
        initialize_seed,       // random_seed
        true                   // show progress
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

    // --- Compute global bounding box (in meters)
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    for (auto& region : regions) {
        min_x = std::min(min_x, region.center.x - region.size/2);
        max_x = std::max(max_x, region.center.x + region.size/2);
        min_y = std::min(min_y, region.center.y - region.size/2);
        max_y = std::max(max_y, region.center.y + region.size/2);
    }

    // --- Pick scale: pixels per meter
    int region_px = 512;  // typical size of birdview images
    //double scale = region_px / regions[0].size;  // px per meter
    double scale = -1.0;
    int tile_px = -1, tile_py = -1;

    // Probe first region to learn tile resolution
    
        const auto& r0 = regions.front();
        LocationInfoRequest probe("{}");
        probe.set_location(location);
        probe.set_rendering_center(std::make_pair(r0.center.x, r0.center.y));
        probe.set_rendering_fov(static_cast<int>(r0.size));
        probe.set_include_map_source(false);

        auto res0 = location_info(probe, &session);
        cv::Mat b0 = cv::imdecode(res0.birdview_image(), cv::IMREAD_COLOR);
        if (b0.empty()) throw std::runtime_error("Failed to decode probe birdview.");

        tile_px = b0.cols;
        tile_py = b0.rows;
        scale   = static_cast<double>(tile_px) / r0.size;  // px per meter

        // Now that scale is known, allocate canvas

    int canvas_w = static_cast<int>((max_x - min_x) * scale);
    int canvas_h = static_cast<int>((max_y - min_y) * scale);
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
    auto clamp = [](int v, int lo, int hi){ return std::max(lo, std::min(v, hi)); };
int idx = 0;

for (const auto& region : regions) {
    invertedai::LocationInfoRequest loc_info_req("{}");
    loc_info_req.set_location(location);
    loc_info_req.set_rendering_center(std::make_optional(std::make_pair(region.center.x, region.center.y)));
    loc_info_req.set_rendering_fov((int)region.size);
    loc_info_req.set_include_map_source(false);

    LocationInfoResponse loc_info_res = invertedai::location_info(loc_info_req, &session);
    std::vector<unsigned char> img_bytes = loc_info_res.birdview_image();
    cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
    if (birdview.empty()) {
        std::cerr << "[WARN] Region " << idx << ": empty birdview, skipping.\n";
        ++idx; continue;
    }

    // Resize tile to match global scale exactly
    int tile_px = (int)std::round(region.size * scale);
    if (birdview.cols != tile_px || birdview.rows != tile_px) {
        cv::resize(birdview, birdview, cv::Size(tile_px, tile_px), 0, 0, cv::INTER_LINEAR);
    }

    // Compute top-left offset (world -> canvas px)
    int offset_x = (int)std::floor((region.center.x - region.size * 0.5 - min_x) * scale);
    int offset_y = (int)std::floor((max_y - (region.center.y + region.size * 0.5)) * scale); // flip y

    // Clamp/crop ROI safely
    int x0 = clamp(offset_x, 0, stitched.cols);
    int y0 = clamp(offset_y, 0, stitched.rows);
    int x1 = clamp(offset_x + birdview.cols, 0, stitched.cols);
    int y1 = clamp(offset_y + birdview.rows, 0, stitched.rows);

    if (x1 <= x0 || y1 <= y0) {
        std::cerr << "[WARN] Region " << idx << ": ROI out of canvas, skipping. "
                  << "offset=(" << offset_x << "," << offset_y << "), tile=("
                  << birdview.cols << "x" << birdview.rows << "), canvas=("
                  << stitched.cols << "x" << stitched.rows << ")\n";
        ++idx; continue;
    }

    cv::Rect dst_roi(x0, y0, x1 - x0, y1 - y0);
    cv::Rect src_roi(x0 - offset_x, y0 - offset_y, dst_roi.width, dst_roi.height);
    birdview(src_roi).copyTo(stitched(dst_roi));

    std::cout << "Placed region " << idx++ << " at (" << offset_x << "," << offset_y
              << "), tile=" << birdview.cols << "x" << birdview.rows
              << ", dst_roi=" << dst_roi << "\n";
}

    // --- Place each region image
    // Build a color per region for nice visualization
    std::vector<cv::Scalar> region_colors;
    region_colors.reserve(regions.size());
    auto random_color = []() {
        static std::mt19937 rng(std::random_device{}());
        static std::uniform_int_distribution<int> dist(0, 255);
        return cv::Scalar(dist(rng), dist(rng), dist(rng));
    };
    for (size_t i = 0; i < regions.size(); ++i) region_colors.push_back(random_color());

    // Helper to test if a world point lies inside a region’s square FOV
    auto inside_region = [](const invertedai::Region& r, double x, double y) {
        double half = r.size * 0.5;
        return (x >= r.center.x - half && x <= r.center.x + half &&
                y >= r.center.y - half && y <= r.center.y + half);
    };

    // ---- draw real agent positions from the consolidated response ----
    int placed = 0;
    for (const auto& ag : response.agent_states()) {
        // find a region “owning” this agent for coloring (nearest square FOV match)
        int ridx = -1;
        for (int i = 0; i < static_cast<int>(regions.size()); ++i) {
            if (inside_region(regions[i], ag.x, ag.y)) { ridx = i; break; }
        }
        cv::Scalar color = (ridx >= 0) ? region_colors[ridx] : cv::Scalar(0, 0, 255);

        // world -> stitched pixel (must match how you pasted tiles)
        int u = static_cast<int>((ag.x - min_x) * scale);
        int v = static_cast<int>((max_y - ag.y) * scale); // y flipped

        if (u >= 0 && u < stitched.cols && v >= 0 && v < stitched.rows) {
            cv::circle(stitched, cv::Point(u, v), 4, color, -1);
            ++placed;
        }
    }
    std::cout << "Total Agents Placed " << placed << "\n";

    // // overlay agents (global placement)
    // for (const auto& agent : response.agent_states()) {
    //     int u = static_cast<int>((agent.x - min_x) * scale);
    //     int v = static_cast<int>((max_y - agent.y) * scale); // flip y
    //     cv::circle(stitched, cv::Point(u,v), 4, cv::Scalar(0,0,255), -1); // red dot
    // }

    // --- Save stitched image
    cv::imwrite("stitched_with_agents.png", stitched);
    std::cout << "Saved stitched_with_agents.png (" << canvas_w << "x" << canvas_h << ")\n";



    // --- Save initialized images
//     std::cout << "Saving initialized region images...\n";
//     int idx = 0;
//     for (auto& region : regions) {
//         // Build request
//         invertedai::LocationInfoRequest loc_info_req("{}");
//         loc_info_req.set_location(location);
//         loc_info_req.set_rendering_center(std::make_optional(std::make_pair(region.center.x, region.center.y)));
//         loc_info_req.set_rendering_fov((int)region.size);
//         loc_info_req.set_include_map_source(false);

//         // Fetch image
//         LocationInfoResponse loc_info_res = invertedai::location_info(loc_info_req, &session);
//         std::vector<unsigned char> img_bytes = loc_info_res.birdview_image();

//         // Decode and save
//         cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);

//         for (const auto& agent : response.agent_states()) {
//             cv::Point pt = world_to_pixel(
//                 agent.x, agent.y,
//                 region.center.x, region.center.y,
//                 birdview.cols, birdview.rows,
//                 region.size
//             );
//             cv::circle(birdview, pt, 5, cv::Scalar(0,0,255), -1);
//         }

// cv::imwrite("region_with_agents.png", birdview);
//         if (!birdview.empty()) {
//             std::string filename = "region_" + std::to_string(idx++) + ".png";
//             cv::imwrite(filename, birdview);
//             std::cout << "Saved " << filename << "\n";
//         } else {
//             std::cerr << "Failed to decode image for region " << idx << "\n";
//         }
//     }

    std::cout << "All done!\n";
    return 0;
}
