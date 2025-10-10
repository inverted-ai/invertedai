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
#include "large_initialize.h"
#include "large_init_helpers.h"
#include "common.h"
#include "large_drive.h"


using namespace invertedai;
// Parent lookup: which initial 100×100 region contains this point?
inline int parent_index_of_point(
    const std::vector<Region>& parents, double x, double y
){
    for (size_t i = 0; i < parents.size(); ++i) {
        const Region& p = parents[i];
        const double hx = p.size * 0.5, hy = p.size * 0.5;
        if (x >= p.center.x - hx && x <= p.center.x + hx &&
            y >= p.center.y - hy && y <= p.center.y + hy) {
            return static_cast<int>(i);
        }
    }
    // Fallback: nearest parent center (guards boundary edge cases)
    double best = std::numeric_limits<double>::infinity();
    int best_i = -1;
    for (size_t i = 0; i < parents.size(); ++i) {
        const auto& p = parents[i];
        double dx = x - p.center.x, dy = y - p.center.y;
        double d2 = dx*dx + dy*dy;
        if (d2 < best) { best = d2; best_i = static_cast<int>(i); }
    }
    return best_i; // could be -1 only if parents empty
}
inline cv::Scalar color_from_parent_index(size_t parent_idx, size_t parent_count) {
    if (parent_count == 0) return cv::Scalar(255,255,255);
    double hue = (static_cast<double>(parent_idx) / parent_count) * 180.0;
    cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto c = bgr.at<cv::Vec3b>(0,0);
    return cv::Scalar(c[0], c[1], c[2]); // BGR
}


auto leaf_color_at (double x, double y,
                         const std::vector<Region>& leaf_regions) -> cv::Scalar {
    // Find which quadtree leaf contains this point
    int leaf_idx = -1;
    for (size_t i = 0; i < leaf_regions.size(); ++i) {
        const Region& r = leaf_regions[i];
        double hx = r.size * 0.5, hy = r.size * 0.5;
        if (x >= r.center.x - hx && x <= r.center.x + hx &&
            y >= r.center.y - hy && y <= r.center.y + hy) {
            leaf_idx = static_cast<int>(i);
            break;
        }
    }

    if (leaf_idx < 0) {
        // Fallback: nearest leaf
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < leaf_regions.size(); ++i) {
            double dx = x - leaf_regions[i].center.x;
            double dy = y - leaf_regions[i].center.y;
            double d2 = dx * dx + dy * dy;
            if (d2 < best) {
                best = d2;
                leaf_idx = static_cast<int>(i);
            }
        }
    }

    // Now color by leaf index (ensures agents and borders align)
    return color_from_parent_index(static_cast<size_t>(leaf_idx),
                                   leaf_regions.size());
}; 

inline cv::Point world_to_canvas(
    double x, double y,
    double min_x, double max_y, double scale,
    int canvas_w, bool flip_x
){
    int u = static_cast<int>(std::llround((x - min_x) * scale));
    int v = static_cast<int>(std::llround((max_y - y) * scale));
    if (flip_x) u = canvas_w - u;              // same flip used in init agents
    return {u, v};
}
cv::Mat draw_quadtree_frame(const std::vector<Region>& regions,
    const std::vector<AgentState>& agents,
    double min_x, double max_x,
    double min_y, double max_y,
    double scale,
    int step)
{
    int canvas_w = static_cast<int>(std::ceil((max_x - min_x) * scale));
    int canvas_h = static_cast<int>(std::ceil((max_y - min_y) * scale));
    cv::Mat frame(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));

    // helper for coloring
    auto color_from_index = [&](size_t i) -> cv::Scalar {
        double hue = (static_cast<double>(i) / regions.size()) * 180.0;
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(hue,255,255));
        cv::Mat bgr; cv::cvtColor(hsv,bgr,cv::COLOR_HSV2BGR);
        auto c = bgr.at<cv::Vec3b>(0,0);
        return cv::Scalar(c[0], c[1], c[2]);
    };

    // Draw regions
    for (size_t i = 0; i < regions.size(); ++i) {
        const Region& r = regions[i];
        cv::Scalar color = color_from_index(i);

        int L = static_cast<int>((r.center.x - r.size/2 - min_x) * scale) + 1;
        int R = static_cast<int>((r.center.x + r.size/2 - min_x) * scale) - 1;
        int T = static_cast<int>((max_y - (r.center.y + r.size/2)) * scale) + 1;
        int B = static_cast<int>((max_y - (r.center.y - r.size/2)) * scale) - 1;

        cv::rectangle(frame, {L,T}, {R,B}, color, 2);
    }

    // Draw agents
    for (const auto& s : agents) {
        int u = static_cast<int>((s.x - min_x) * scale);
        int v = static_cast<int>((max_y - s.y) * scale);
        cv::circle(frame, {u,v}, 4, cv::Scalar(0,0,0), cv::FILLED);
    }

    // Label frame number
    cv::putText(frame, "Step " + std::to_string(step),
    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
    cv::Scalar(255, 255, 255), 3);  // white outline
    cv::putText(frame, "Step " + std::to_string(step),
    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
    cv::Scalar(255, 0, 0), 2);      // blue fill

    return frame;
}

// Clamp helper
inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

// ensure all regions are 100m x 100m 
auto validate_regions_100x100 (const std::vector<invertedai::Region>& regs,
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
    };

// Simple agent generator for testing
    inline std::pair<std::vector<AgentState>, std::vector<AgentProperties>>
    generate_agents_for_region(const Region& region,
                               const std::map<AgentType, int>& agent_count_dict)
    {
        std::vector<AgentState> states;
        std::vector<AgentProperties> props;
    
        // simple RNG for placing agents inside the region bounds
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist_x(region.center.x - region.size/2.0,
                                                      region.center.x + region.size/2.0);
        std::uniform_real_distribution<double> dist_y(region.center.y - region.size/2.0,
                                                      region.center.y + region.size/2.0);
        std::uniform_real_distribution<double> dist_theta(0.0, 2*M_PI);
        double spawn_x = region.center.x + 40.0; // near top-right but inside
        double spawn_y = region.center.y + 40.0; // slightly offset downward
        for (auto& [atype, count] : agent_count_dict) {
            for (int i = 0; i < count; i++) {
                AgentState st;
                st.x = spawn_x; 
                st.y = spawn_y; // avoid overlap for testing
                // st.x = dist_x(rng);
                // st.y = dist_y(rng);
                st.orientation = dist_theta(rng);
                st.speed = 0.0; // start stationary
    
                AgentProperties pr;
                if (atype == AgentType::car) {
                    pr.length = 4.5;
                    pr.width = 1.8;
                    pr.rear_axis_offset = 1.0;
                    pr.agent_type = "car";
                    pr.waypoint = region.center;   // give it a target
                    pr.max_speed = 15.0;
                } else if (atype == AgentType::pedestrian) {
                    pr.length = 0.5;
                    pr.width = 0.5;
                    pr.rear_axis_offset = 0.0;
                    pr.agent_type = "pedestrian";
                    pr.waypoint = region.center;
                    pr.max_speed = 1.5;
                }
    
                //states.push_back(st);
                props.push_back(pr);
            }
        }
    
        return {states, props};
    }
    inline cv::Rect region_rect_pixels(
        const Region& r,
        double min_x, double max_y, double scale,
        int canvas_w, int canvas_h, bool flip_x)
    {
        // Round to 1 cm world precision to remove floating noise, not to a big grid
        const double eps = 1e-2;
    
        const double cx   = std::round(r.center.x / eps) * eps;
        const double cy   = std::round(r.center.y / eps) * eps;
        const double half = std::round(r.size / 2.0 / eps) * eps;
    
        const double left   = cx - half;
        const double right  = cx + half;
        const double top    = cy + half;
        const double bottom = cy - half;
    
        cv::Point tl = world_to_canvas(left,  top,    min_x, max_y, scale, canvas_w, flip_x);
        cv::Point br = world_to_canvas(right, bottom, min_x, max_y, scale, canvas_w, flip_x);
    
        // Clamp to canvas safely
        int L = std::clamp(std::min(tl.x, br.x), 0, canvas_w);
        int R = std::clamp(std::max(tl.x, br.x), 0, canvas_w);
        int T = std::clamp(std::min(tl.y, br.y), 0, canvas_h);
        int B = std::clamp(std::max(tl.y, br.y), 0, canvas_h);
    
        if (R <= L || B <= T) return {};
    
        return cv::Rect(L, T, R - L, B - T);
    }

    void draw_traffic_lights(cv::Mat& frame,
        const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
        const std::map<std::string, cv::Point>& light_positions_px)
    {
    if (!traffic_lights_states.has_value()) {
        std::cout << "[DEBUG] No traffic_lights_states value.\n";
        return;
    }

    for (const auto& [light_id, state] : traffic_lights_states.value()) {
        auto it = light_positions_px.find(light_id);
        if (it == light_positions_px.end()) continue;

        cv::Scalar color;
        if (state == "red")      color = cv::Scalar(0, 0, 255);
        else if (state == "yellow") color = cv::Scalar(0, 255, 255);
        else if (state == "green")  color = cv::Scalar(0, 255, 0);
        else                        color = cv::Scalar(128, 128, 128);  // off/unknown

        cv::circle(frame, it->second, 6, color, cv::FILLED);
        cv::circle(frame, it->second, 8, cv::Scalar(0,0,0), 1); // thin border
    }
}
cv::Point world_to_pixel(double x, double y,
    const invertedai::Point2d& origin_px, double px_per_meter = 2.0)
{
int px = static_cast<int>(origin_px.x + x * px_per_meter);
int py = static_cast<int>(origin_px.y - y * px_per_meter); // invert Y for image coords
return cv::Point(px, py);
}

// Helper: Paste all cached tiles into one big stitched canvas
cv::Mat build_full_map_from_cache(
    const std::unordered_map<int, cv::Mat>& cached_tiles,
    const std::vector<Region>& all_regions,
    const std::vector<Region>& final_regions,
    double min_x, double max_y, double scale,
    int canvas_w, int canvas_h,
    bool FLIP_X_FOR_THIS_DOMAIN)
{
    const double tile_size = final_regions[0].size;
    const double max_x = final_regions.back().max_x;  // derive if you already have it elsewhere
    const double min_y = final_regions.front().min_y;
    const int tile_px = static_cast<int>(std::round(tile_size * scale)); // pixels per tile
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));

    if (all_regions.empty()) {
        std::cerr << "[WARN] No regions provided.\n";
        return stitched;
    }


    // Estimate how many tiles horizontally and vertically
    int num_cols = static_cast<int>(std::ceil((max_x - min_x) / tile_size));
    int num_rows = static_cast<int>(std::ceil((max_y - min_y) / tile_size));

    for (size_t i = 0; i < all_regions.size(); ++i) {
        const Region& r = all_regions[i];
        auto it = cached_tiles.find(i);
        if (it == cached_tiles.end()) {
            std::cerr << "[WARN] Missing cached tile for region " << i << "\n";
            continue;
        }

        const cv::Mat& tile = it->second;
        const int tile_px = tile.cols;

        // --- Compute pixel offsets ---
        int col = static_cast<int>(std::round((r.center.x - min_x) / tile_size));
        int row = static_cast<int>(std::round((max_y - r.center.y) / tile_size));

        // Apply flip if needed
        if (FLIP_X_FOR_THIS_DOMAIN) {
            col = (num_cols - 1) - col;
        }

        // Convert to pixel offset
        int offset_x = col * tile_px;
        int offset_y = row * tile_px;

        // Clamp to bounds
        int x0 = clampi(offset_x, 0, stitched.cols);
        int y0 = clampi(offset_y, 0, stitched.rows);
        int x1 = clampi(offset_x + tile.cols, 0, stitched.cols);
        int y1 = clampi(offset_y + tile.rows, 0, stitched.rows);
        if (x1 <= x0 || y1 <= y0) continue;

        cv::Rect dst(x0, y0, x1 - x0, y1 - y0);
        cv::Rect src(x0 - offset_x, y0 - offset_y, dst.width, dst.height);
        tile(src).copyTo(stitched(dst));
    }

    std::cout << "[INFO] Built static stitched background using "
              << cached_tiles.size() << " tiles (" << num_cols << "×" << num_rows << ").\n";
    return stitched;
}



int main() {

// bazel build //large:large_main 
// ./bazel-bin/large/large_main
    const std::string location = "carla:Town03";
    bool FLIP_X_FOR_THIS_DOMAIN = false;

    if (location.rfind("carla:", 0) == 0) {
        FLIP_X_FOR_THIS_DOMAIN = true;
        std::cout << "Detected CARLA map → flipping axes\n";
    }
    //constexpr bool FLIP_X_FOR_THIS_DOMAIN = true; // set to true if using carla maps
    std::string API_KEY = "wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni"; // = getenv("INVERTEDAI_API_KEY"); or just paste here

    // controls for how many additional agents to add
    int total_num_agents = 305;

    int sim_length = 100;   // num steps for large_drive simulation

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
    validate_regions_100x100(regions, /*expected=*/100.0);
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
    auto [init_states, init_props] = generate_agents_for_region(cfg.regions.front(), {
        {AgentType::car, 10}, // change the 10 -> however many cars you want to initialize
    
    });
    cfg.agent_states     = std::nullopt;  //init_states; change to init_states to use the generator function
    cfg.agent_properties = std::nullopt; //init_props; 

    std::cout << "Calling large_initialize with " << regions.size() << " regions...\n";
    auto out = invertedai::large_initialize_with_regions(cfg);
    InitializeResponse response = std::move(out.response);
    std::vector<Region> outputed_regions = std::move(out.regions); // use this for drawing

    // Extract and keep the evolving state outside the InitializeResponse
    std::vector<AgentState> agent_states     = response.agent_states();
    std::vector<AgentProperties> agent_props = response.agent_properties();
    std::vector<std::vector<double>> recurrent    = response.recurrent_states();
    std::optional<std::map<std::string, std::string>> traffic_lights_states = response.traffic_lights_states();
    std::optional<std::vector<LightRecurrentState>> light_recurrent_states = response.light_recurrent_states();

    std::cout << "=== Raw InitializeResponse JSON ===" << std::endl;

    // Use only final_regions from here on for geometry/rendering
    const std::vector<Region>& final_regions = outputed_regions;


// convenience: color at any world (x,y) based on initial parents
auto parent_color_at = [&](double x, double y) -> cv::Scalar {
    int p = parent_index_of_point(final_regions, x, y);
    size_t parent_count = final_regions.size();
    size_t pi = (p < 0 ? 0 : static_cast<size_t>(p));
    return color_from_parent_index(pi, parent_count);
};

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
    LocationInfoResponse probe_res = location_info(probe, &session);
    cv::Mat probe_tile = cv::imdecode(probe_res.birdview_image(), cv::IMREAD_COLOR);
    if (probe_tile.empty()) throw std::runtime_error("Failed to decode probe birdview");
    const double scale = static_cast<double>(probe_tile.cols) / probe_region.size; // px per meter

    // cache the regions tiles 
    std::unordered_map<int, cv::Mat> cached_tiles;

    for (size_t i = 0; i < final_regions.size(); ++i) {
        const Region& r = final_regions[i];
        LocationInfoRequest req("{}");
        req.set_location(location);
        req.set_rendering_center(std::make_pair(r.center.x, r.center.y));
        req.set_rendering_fov(static_cast<int>(r.size));
        req.set_include_map_source(false);

        LocationInfoResponse res = location_info(req, &session);
        cv::Mat tile = cv::imdecode(res.birdview_image(), cv::IMREAD_COLOR);
        if (tile.empty()) {
            std::cerr << "[WARN] Tile " << i << " is empty.\n";
            continue;
        }

        const int tile_px = static_cast<int>(std::llround(r.size * scale));
        if (tile.cols != tile_px || tile.rows != tile_px) {
            cv::resize(tile, tile, cv::Size(tile_px, tile_px), 0, 0, cv::INTER_LINEAR);
        }

        cached_tiles[i] = tile;
    }
    // 3) Canvas
    const int canvas_w = static_cast<int>(std::ceil((max_x - min_x) * scale));
    const int canvas_h = static_cast<int>(std::ceil((max_y - min_y) * scale));
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));

    auto clampi = [](int v, int lo, int hi){ return std::max(lo, std::min(v, hi)); };
    std::map<std::string, cv::Point> traffic_light_positions_px;

    for (const auto& actor : probe_res.static_actors()) {
        if (actor.agent_type == "traffic_light") {
            cv::Point pt = world_to_canvas(
                actor.x, actor.y,
                min_x, max_y, scale,
                canvas_w, FLIP_X_FOR_THIS_DOMAIN
            );
            traffic_light_positions_px[std::to_string(actor.actor_id)] = pt;
        }
    }


    auto paste_region_tile = [&](const Region& r, int idx, cv::Mat& stitched) {
        auto it = cached_tiles.find(idx);
        if (it == cached_tiles.end()) {
            std::cerr << "[WARN] Missing cached tile for region " << idx << "\n";
            return;
        }
    
        cv::Mat tile = it->second;
        const int tile_px = tile.cols;
    
        int offset_x, offset_y;
        if (FLIP_X_FOR_THIS_DOMAIN) {
            int num_cols = static_cast<int>(std::round((max_x - min_x) / r.size));
            int col = static_cast<int>(std::round((r.center.x - min_x) / r.size));
            int flipped_col = (num_cols - 1) - col;
    
            int num_rows = static_cast<int>(std::round((max_y - min_y) / r.size));
            int row = static_cast<int>(std::round((max_y - r.center.y) / r.size));
    
            flipped_col = std::clamp(flipped_col + 1, 0, num_cols - 1);
            row = std::clamp(row - 1, 0, num_rows - 1);
    
            offset_x = flipped_col * tile_px;
            offset_y = row * tile_px;
        } else {
            offset_x = static_cast<int>(std::floor((r.center.x - r.size * 0.5 - min_x) * scale));
            offset_y = static_cast<int>(std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));
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
    for (size_t i = 0; i < final_regions.size(); ++i) {
        paste_region_tile(final_regions[i], i, stitched);
    }

    int total_drawn = 0;
    for (size_t i = 0; i < final_regions.size(); ++i) {
        const Region& r = final_regions[i];
        const cv::Scalar color = color_from_parent_index(i, final_regions.size());
    
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
    cv::imwrite("frame_0000.png", stitched);




        // --- Now start stepping with large_drive ---
        int drive_seed = std::uniform_int_distribution<>(1, 1000000)(gen);
    
        // Carry forward agent_properties (constant)
        // if(response.agent_properties().size() < response.agent_states().size()) {
        //     throw std::runtime_error("Fewer agent_properties than agent_states in InitializeResponse");
        // }
        std::cout << "Starting simulation for " << sim_length << " steps...\n";
        std::vector<Region> full_regions;
        // double tile_size = final_regions[0].size; // use same FOV as your initialize tiles

        // for (double y = min_y + tile_size / 2.0; y < max_y; y += tile_size) {
        //     for (double x = min_x + tile_size / 2.0; x < max_x; x += tile_size) {
        //         Region r(Point2d{x, y}, tile_size);
        //         full_regions.push_back(r);
        //     }
        // }

        // auto all_tiles = invertedai::get_all_map_tiles(
        //     session, location,
        //     min_x, max_x, min_y, max_y,
        //     100.0, 2.0, FLIP_X_FOR_THIS_DOMAIN
        // );
        
        // cv::Mat stitchedd = build_full_map_from_cache(
        //     all_tiles, get_regions_in_grid(max_x-min_x, max_y-min_y), final_regions,
        //     min_x, max_y, 2.0, canvas_w, canvas_h, FLIP_X_FOR_THIS_DOMAIN
        // );
        LargeDriveConfig drive_cfg(session);
        drive_cfg.location = location;
        drive_cfg.agent_states = agent_states;
        drive_cfg.agent_properties = agent_props;
        drive_cfg.recurrent_states = recurrent;
        drive_cfg.traffic_lights_states = traffic_lights_states;
        drive_cfg.light_recurrent_states = light_recurrent_states;
        drive_cfg.random_seed = drive_seed;
        drive_cfg.get_infractions = true;
        drive_cfg.single_call_agent_limit = 100;
        drive_cfg.async_api_calls = true;
        for (int step = 0; step < sim_length; ++step) {   
            auto out = large_drive_with_regions(drive_cfg);
            DriveResponse drive_response = std::move(out.response);
            std::vector<Region> leaf_regions = std::move(out.regions);
            // === update evolving states ===
            auto states = drive_response.agent_states();
            auto recur  = drive_response.recurrent_states();
            auto lights_recur = drive_response.light_recurrent_states();
            auto traff  = drive_response.traffic_lights_states();
            
            // update all drive_cfg fields in the same consistent order
            drive_cfg.agent_states           = states;
            drive_cfg.recurrent_states       = recur;
            drive_cfg.light_recurrent_states = lights_recur;
            drive_cfg.traffic_lights_states  = std::nullopt;
            drive_cfg.api_model_version      = drive_response.model_version();
            drive_cfg.random_seed            = std::nullopt;
            // --------------------------------

            // // === Build base birdview (like large_initialize did) ===
            cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
            for (size_t i = 0; i < final_regions.size(); ++i) {
                paste_region_tile(final_regions[i], i, stitched);
            }

            // cv::Mat stitched = stitchedd.clone();
            // === Overlay quadtree regions (children take their parent color) ===
            for (const Region& r : leaf_regions) {
                // int parent_idx = parent_index_of_point(leaf_regions, r.center.x, r.center.y);
                cv::Scalar color = parent_color_at(r.center.x, r.center.y);

                cv::Rect rr = region_rect_pixels(
                    r, min_x, max_y, scale,
                    stitched.cols, stitched.rows,
                    FLIP_X_FOR_THIS_DOMAIN
                );
                if (rr.area() > 0) {
                    // small inward inset to avoid covering seams
                    cv::rectangle(stitched,
                                {rr.x+1, rr.y+1},
                                {rr.x + rr.width - 2, rr.y + rr.height - 2},
                                color, 2, cv::LINE_AA);
                }

            // === Draw agents (color by their current parent region) ===
            for (const auto& s : r.agent_states) {
                cv::Point pt = world_to_canvas(
                    s.x, s.y, min_x, max_y, scale,
                    stitched.cols, FLIP_X_FOR_THIS_DOMAIN
                );
                if (0 <= pt.x && pt.x < stitched.cols &&
                    0 <= pt.y && pt.y < stitched.rows) {
                    cv::circle(stitched, pt, 4, color, cv::FILLED, cv::LINE_AA);
                }
            }
        }
        auto lights_opt = drive_response.traffic_lights_states();
        draw_traffic_lights(stitched, lights_opt, traffic_light_positions_px);

            // for (const auto& r : leaf_regions) {
            //     int p_r = parent_index_of_point(final_regions, r.center.x, r.center.y);
            //     std::cout << "Leaf " << &r - &leaf_regions[0]
            //               << " parent=" << p_r << " color=" << p_r % 10 << "\n";
            // }
            // for (const auto& s : response.agent_states()) {
            //     int p_a = parent_index_of_point(final_regions, s.x, s.y);
            //     std::cout << "Agent color parent=" << p_a << " color=" << p_a % 10 << "\n";
            // }
            // === Label step ===
            cv::putText(stitched, "Step " + std::to_string(step),
                        cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0,0,0), 2);
        
            // Save frame
            char buf[64];
            sprintf(buf, "frame_%04d.png", step);
            cv::imwrite(buf, stitched);
        
            // Carry forward states
            // response.set_agent_states(drive_response.agent_states());
            // response.set_recurrent_states(drive_response.recurrent_states());
            // if (drive_response.light_recurrent_states().has_value()) {
            //     response.set_light_recurrent_states(drive_response.light_recurrent_states().value());
            // }
            int total_agents = drive_response.agent_states().size();
            int num_leaves = leaf_regions.size();
            double avg_agents_per_leaf = double(total_agents) / num_leaves;

            std::cout << "[Step " << step << "] " << num_leaves
                    << " leaves, avg " << avg_agents_per_leaf << " agents/leaf\n";
        }
        
        // int canvas_w = static_cast<int>(std::ceil((max_x - min_x) * scale));
        // int canvas_h = static_cast<int>(std::ceil((max_y - min_y) * scale));
        
        cv::VideoWriter writer("quadtree_sim.avi",
                               cv::VideoWriter::fourcc('M','J','P','G'),
                               10, // fps
                               cv::Size(canvas_w, canvas_h));
        // Load the intro image (stitched_with_agents.png)
        cv::Mat intro = cv::imread("stitched_with_agents.png");
        // if (!intro.empty()) {
            // Resize it to match your canvas if necessary
            // if (intro.size() != cv::Size(canvas_w, canvas_h)) {
            //     cv::resize(intro, intro, cv::Size(canvas_w, canvas_h));
            // }

            // // Write it for the first second (10 frames at 10 fps)
            // for (int i = 0; i < 10; ++i) {
            //     writer.write(intro);
            // }
        // } else {
        //     std::cerr << "[WARN] Could not load stitched_with_agents.png — skipping intro.\n";
        // }
        for (int step = 0; step < sim_length; ++step) {
            char buf[64];
            sprintf(buf, "frame_%04d.png", step);
            cv::Mat img = cv::imread(buf);
            if (!img.empty()) {
                writer.write(img);
            }
        }
        writer.release();
        std::cout << "Simulation finished after " << sim_length << " steps.\n";
    
    return 0;
}
