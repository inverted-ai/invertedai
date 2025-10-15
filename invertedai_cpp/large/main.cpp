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
#include "large_initialize.h"
#include "large_init_helpers.h"

using namespace invertedai;

// Clamp helper
int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

// ensure all regions are 100m x 100m 
auto validate_regions (const std::vector<invertedai::Region>& regs,
    double expected = 100.0,
    double tol = 1e-6,
    double px_per_meter = -1.0) {
    bool ok = true;
    for (size_t i = 0; i < regs.size(); ++i) {
        double s = regs[i].size;
        if (std::abs(s - expected) > tol) {
            std::cerr << "[WARN] Region " << i << " size=" << s
        << " (expected " << expected << ")\n";
        ok = false;
        }

        // Optional: world-space bbox sanity (useful for debugging/visual overlays)
        const double half = s * 0.5;
        const double L = regs[i].center.x - half;
        const double R = regs[i].center.x + half;
        const double B = regs[i].center.y - half;
        const double T = regs[i].center.y + half;
        (void)L; (void)R; (void)B; (void)T; // keep if you print later

        // Optional pixel-level check (after you've computed `scale`)
        if (px_per_meter > 0.0) {
            int expected_px = (int)std::llround(s * px_per_meter);
            if (expected_px <= 0) {
                std::cerr << "[WARN] Region " << i
                << " expected_px <= 0 (s=" << s
                << ", scale=" << px_per_meter << ")\n";
                ok = false;
            }
        }
    }
    if (ok) {
        std::cout << "[OK] All " << regs.size()
        << " regions are 100m x 100m (FOV=" << expected << ")\n";
    }
    return ok;
}

// Simple agent generator for testing
std::pair<std::vector<AgentState>, std::vector<AgentProperties>>
initialize_agents_for_region(
    invertedai::Session& session,
    const std::string& location,
    const Region& region,
    int num_agents,
    int random_seed = 1,
    bool get_birdview = false,
    bool get_infractions = false)
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

int main() {

// bazel build //large:large_main 
// ./bazel-bin/large/large_main
    std::string location = "carla:Town10HD";
    constexpr bool FLIP_X_FOR_THIS_DOMAIN = true; // set to true if using carla maps
    std::string API_KEY = "wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni"; // = getenv("INVERTEDAI_API_KEY"); or just paste here

    // controls for how many additional agents to add
    int total_num_agents = 90;

    // (used by get_regions_default)
    int width  = 1000;
    int height = 1000;

    // Random seed
    std::random_device rd;
    std::mt19937 gen(rd());
    int initialize_seed = std::uniform_int_distribution<>(1, 1000000)(gen); // or fixed for repeatability 

    // --- Session connection
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(API_KEY);
    session.connect();

    // --- Get map info (for map_center)
    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    LocationInfoResponse li_res = location_info(li_req, &session);

    std::pair<float,float> map_center{
        static_cast<float>(li_res.map_origin().x),
        static_cast<float>(li_res.map_origin().y)
    };

    // --- Generate default regions
    std::map<AgentType,int> agent_count_dict = {
        {AgentType::car, total_num_agents}
    };

    std::cout << "Generating default regions...\n";
    std::vector<Region> regions = get_regions_default(
        location,
        total_num_agents,                              // total_num_agents (legacy arg kept)
        agent_count_dict,                              // agent_count_dict
        session,
        std::pair<float,float>{width/2.f, height/2.f}, // area_shape / hint
        map_center,                                    // map center from location_info
        initialize_seed                               // random seed

    );

    // helper to check if regions have been validated currently
    validate_regions(regions, /*expected=*/100.0);
    std::cout << "Generated " << regions.size() << " regions.\n";

    // set up arguments for large initialize
    LargeInitializeConfig cfg(session);
    cfg.location = location;
    cfg.regions = regions;                // seed regions
    cfg.random_seed = initialize_seed;
    cfg.get_infractions = true;
    cfg.traffic_light_state_history = std::nullopt;
    cfg.return_exact_agents = true;
    cfg.api_model_version = std::nullopt;

    // Simple agent generator for testing - currently not being used
    auto [init_states, init_props] =
    initialize_agents_for_region(session, location, regions[0], 3, initialize_seed);

    cfg.agent_states     = init_states; //change to init_states to use the generator function
    cfg.agent_properties = init_props; 

    std::cout << "Calling large_initialize with " << regions.size() << " regions...\n";
    auto out = invertedai::large_initialize_with_regions(cfg);
    InitializeResponse response = std::move(out.response);
    std::vector<Region> outputed_regions = std::move(out.regions); // use this for drawing


    std::cout << "Number of agents (merged): " << response.agent_states().size() << "\n";
    // Use only final_regions from here on for geometry/rendering
    const std::vector<Region>& final_regions = outputed_regions;
    for(const auto& r : final_regions) {
        std::cout << " Region center=(" << r.center.x << "," << r.center.y << ")"
        << " size=" << r.size
        << " num_agents=" << r.agent_states.size() << "\n";
    }


    // --- Stitching and drawing workflow
    // 1) Bounds in world meters
    double min_x =  std::numeric_limits<double>::infinity();
    double min_y =  std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& r :  final_regions) {
        min_x = std::min(min_x, r.center.x - r.size * 0.5);
        max_x = std::max(max_x, r.center.x + r.size * 0.5);
        min_y = std::min(min_y, r.center.y - r.size * 0.5);
        max_y = std::max(max_y, r.center.y + r.size * 0.5);
    }

    // 2) Learn scale from the first final region
    auto probe_region =  final_regions.front();
    LocationInfoRequest probe("{}");
    probe.set_location(location);
    probe.set_rendering_center(std::make_pair(probe_region.center.x, probe_region.center.y));
    probe.set_rendering_fov(static_cast<int>(probe_region.size));
    probe.set_include_map_source(false);
    auto probe_res = location_info(probe, &session);
    cv::Mat probe_tile = cv::imdecode(probe_res.birdview_image(), cv::IMREAD_COLOR);
    if (probe_tile.empty()) throw std::runtime_error("Failed to decode probe birdview");
    const double scale = static_cast<double>(probe_tile.cols) / probe_region.size; // px per meter

    // 3) Canvas
    const int canvas_w = static_cast<int>(std::ceil((max_x - min_x) * scale));
    const int canvas_h = static_cast<int>(std::ceil((max_y - min_y) * scale));
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));

    auto clampi = [](int v, int lo, int hi){ return std::max(lo, std::min(v, hi)); };


    auto paste_region_tile = [&](const Region& r, int idx) {
        LocationInfoRequest req("{}");
        req.set_location(location);
        req.set_rendering_center(std::make_pair(r.center.x, r.center.y));
        req.set_rendering_fov(static_cast<int>(r.size));
        req.set_include_map_source(false);
    
        LocationInfoResponse res = location_info(req, &session);
        cv::Mat tile = cv::imdecode(res.birdview_image(), cv::IMREAD_COLOR);
        if (tile.empty()) { std::cerr << "[WARN] tile empty @ region " << idx << "\n"; return; }
    
        const int tile_px = static_cast<int>(std::llround(r.size * scale));
        if (tile.cols != tile_px || tile.rows != tile_px) {
            cv::resize(tile, tile, cv::Size(tile_px, tile_px), 0, 0, cv::INTER_LINEAR);
        }
   
        int offset_x, offset_y;
        if (FLIP_X_FOR_THIS_DOMAIN) {
            int num_cols = static_cast<int>(std::round((max_x - min_x) / r.size));
            int col = static_cast<int>(std::round((r.center.x - min_x) / r.size));
            int flipped_col = (num_cols - 1) - col;

            int num_rows = static_cast<int>(std::round((max_y - min_y) / r.size));
            int row = static_cast<int>(std::round((max_y - r.center.y) / r.size));

            // shift up and right by one
            flipped_col += 1;
            row -= 1;

            // clamp so we don’t go out of bounds
            flipped_col = std::max(0, std::min(flipped_col, num_cols - 1));
            row = std::max(0, std::min(row, num_rows - 1));
            // final pixel offsets
            offset_x = flipped_col * tile_px;
            offset_y = row * tile_px;
        } else {
            // top-left in canvas pixels (MUST match drawing math below)
            offset_y = static_cast<int>(std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));
            offset_x = static_cast<int>(std::floor((r.center.x - r.size*0.5 - min_x) * scale));
        }
        int x0 = clampi(offset_x, 0, stitched.cols);
        int y0 = clampi(offset_y, 0, stitched.rows);
        int x1 = clampi(offset_x + tile.cols, 0, stitched.cols);
        int y1 = clampi(offset_y + tile.rows, 0, stitched.rows);
        if (x1 <= x0 || y1 <= y0) return;
    
        cv::Rect dst(x0, y0, x1 - x0, y1 - y0);
        cv::Rect src(x0 - offset_x, y0 - offset_y, dst.width, dst.height);
        tile(src).copyTo(stitched(dst));
    };
    
    // Paste all tiles
    for (size_t i = 0; i <  final_regions.size(); ++i) paste_region_tile( final_regions[i], static_cast<int>(i));

    // color of borders and agents
    auto color_from_index = [&](size_t i) -> cv::Scalar {
        size_t N = final_regions.size();  // total number of regions
        if (N == 0) return cv::Scalar(255, 255, 255);
    
        // Evenly spaced hues across [0, 180) in OpenCV HSV
        double hue = (static_cast<double>(i) / N) * 180.0; 
        double sat = 255.0; // full saturation
        double val = 255.0; // full brightness
    
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    
        cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
        return cv::Scalar(c[0], c[1], c[2]); // OpenCV uses BGR order
    };

    int total_drawn = 0;
    for (size_t i = 0; i < final_regions.size(); ++i) {
        const Region& r = final_regions[i];
        const cv::Scalar color = color_from_index(i);
    
        // Compute tile size in pixels
        const int tile_px = static_cast<int>(std::llround(r.size * scale));
    
        int offset_x, offset_y;
        // carla map quirk
        if (FLIP_X_FOR_THIS_DOMAIN) {
            // Compute how many columns in total
            int num_cols = static_cast<int>(std::round((max_x - min_x) / r.size));
            // Region’s column index
            int col = static_cast<int>(std::floor((r.center.x - r.size * 0.5 - min_x) / r.size));
                        // Flip column across vertical axis
            int flipped_col = (num_cols - 1) - col;
    
            offset_x = flipped_col * tile_px;
            offset_y = static_cast<int>(
                std::floor((max_y - (r.center.y + r.size * 0.5)) * scale)
            );
        } else {
            // Default (unflipped) behavior
            offset_x = static_cast<int>(std::floor((r.center.x - r.size * 0.5 - min_x) * scale));
            offset_y = static_cast<int>(std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));
        }
    
        // Border rectangle
        int L  = clampi(offset_x, 0, stitched.cols) + 2;
        int T  = clampi(offset_y, 0, stitched.rows) + 2;
        int Rr = clampi(offset_x + tile_px - 1, 0, stitched.cols) - 2;
        int Bb = clampi(offset_y + tile_px - 1, 0, stitched.rows) - 2;
        if (Rr > L && Bb > T) {
            cv::rectangle(stitched, {L, T}, {Rr, Bb}, color, 2, cv::LINE_AA);
        }
    
        // Agents from THIS region
        for (const auto& s : r.agent_states) {
            int u = static_cast<int>(std::llround((s.x - min_x) * scale));
            int v = static_cast<int>(std::llround((max_y - s.y) * scale));
    
            if (FLIP_X_FOR_THIS_DOMAIN) {
                u = stitched.cols - u;
            }
    
            if ((unsigned)u < (unsigned)stitched.cols &&
                (unsigned)v < (unsigned)stitched.rows) {
                cv::circle(stitched, {u, v}, 4, color, cv::FILLED, cv::LINE_AA);
                ++total_drawn;
            }
        }
    }
    std::cout << "Total agents drawn from final_regions: " << total_drawn << "\n";

    // --- Save
    cv::imwrite("stitched_with_agents.png", stitched);
    std::cout << "Saved stitched_with_agents.png (" << stitched.cols << "x" << stitched.rows << ")\n";
    std::cout << "All done!\n";
    return 0;
}
