#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <optional>
#include <opencv2/opencv.hpp>
#include <utility>
#include <ostream>


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
static cv::Point world_to_canvas(
    double x, double y,
    double min_x, double max_y, double scale,
    int canvas_w, bool flip_x
){
    int u = static_cast<int>(std::llround((x - min_x) * scale));
    int v = static_cast<int>(std::llround((max_y - y) * scale));
    if (flip_x) u = canvas_w - u;              // same flip used in init agents
    return {u, v};
}
// Clamp helper
int clampi(
    int v, 
    int lo, 
    int hi
) { 
    return std::max(lo, std::min(v, hi)); 
}
static cv::Rect region_rect_pixels(
    const Region& r,
    double min_x, 
    double max_y, 
    double scale,
    int canvas_w, 
    int canvas_h, 
    bool flip_x
) {
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

static void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
    const std::map<std::string, cv::Point>& light_positions_px,
    std::vector<StaticMapActor> static_actors,
    bool flip_x,
    int scale
) {
    if (!traffic_lights_states.has_value()) {
        std::cout << "[DEBUG] No traffic_lights_states value.\n";
        return;
    }  

    for (const auto& [light_id, state] : *traffic_lights_states) {

        // Convert state → color
        cv::Scalar color;
        if (state == "red") color = cv::Scalar(0,0,255);
        else if (state == "yellow") color = cv::Scalar(0,255,255);
        else if (state == "green") color = cv::Scalar(0,255,0);
        else color = cv::Scalar(128,128,128);

        // Get pixel center
        auto it = light_positions_px.find(light_id);
        if (it == light_positions_px.end()) continue;
        cv::Point center_px = it->second;

        // Find corresponding StaticMapActor
        const StaticMapActor* A = nullptr;
        for (const auto& act : static_actors) {
            if (act.agent_type == "traffic_light" &&
                std::to_string(act.actor_id) == light_id) {
                A = &act;
                break;
            }
        }
        if (!A) continue;

        double l_m = std::max(1.0, A->length.value_or(1.0));
        double w_m = std::max(1.0, A->width.value_or(1.0));

        double l_px = l_m * scale*1.2;
        double w_px = w_m * scale*1.2;

        double psi_deg = - A->orientation * 180.0 / CV_PI;
        if (flip_x) psi_deg = 180.0 -psi_deg;

        cv::RotatedRect box(center_px, cv::Size2f(l_px, w_px), psi_deg);
        cv::Point2f v[4];
        box.points(v);
        cv::fillConvexPoly(frame, std::vector<cv::Point>{v[0],v[1],v[2],v[3]}, color);


    // const int rect_w = static_actors[0].width.value_or(1.0) * scale;   // assuming all static actors have same length and width
    // const int rect_h = static_actors[0].length.value_or(1.0) * scale; 

    // for (const auto& [light_id, state] : traffic_lights_states.value()) {
    //     auto it = light_positions_px.find(light_id);
    //     if (it == light_positions_px.end()) continue;

    //     cv::Scalar color;
    //     if (state == "red")           color = cv::Scalar(0, 0, 255);
    //     else if (state == "yellow")   color = cv::Scalar(0, 255, 255);
    //     else if (state == "green")    color = cv::Scalar(0, 255, 0);
    //     else                          color = cv::Scalar(128, 128, 128);

    //     cv::Point center = it->second;

    //     // axis-aligned rectangle centered on the light
    //     cv::Rect box(
    //         center.x - rect_w/2,
    //         center.y - rect_h/2,
    //         rect_w,
    //         rect_h
    //     );

    //     // fill rectangle
    //     cv::rectangle(frame, box, color, cv::FILLED);

    //     // thin border
    //     cv::rectangle(frame, box, cv::Scalar(0,0,0), 1);
    }
}

static std::map<std::string, cv::Point> get_traffic_light_positions(
    const LocationInfoResponse& li_res,
    double min_x,
    double max_y,
    double scale,
    int canvas_w,
    bool flip_x
) {
    std::map<std::string, cv::Point> positions;
    for (const auto& actor : li_res.static_actors()) {
        if (actor.agent_type == "traffic_light") {
            cv::Point pt = world_to_canvas(
                actor.x, actor.y,
                min_x, max_y, scale,
                canvas_w, flip_x
            );
            positions[std::to_string(actor.actor_id)] = pt;
        }
    }
    return positions;
}

void paste_region_tile(
    const Region& r,
    const std::unordered_map<int, cv::Mat>& tiles,
    int idx,
    cv::Mat& stitched,
    double min_x,
    double max_y,
    double max_x,
    double min_y,
    bool flip_x,
    double scale
) {
    
    auto it = tiles.find(idx);
    if (it == tiles.end()) {
        std::cerr << "[WARN] Missing cached tile for region " << idx << "\n";
        return;
    }

    const cv::Mat& tile = it->second;
    const int tile_px = tile.cols;

    int offset_x = 0, offset_y = 0;

    if (flip_x) {
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
        offset_x = static_cast<int>(
            std::floor((r.center.x - r.size * 0.5 - min_x) * scale));
        offset_y = static_cast<int>(
            std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));
    }

    int x0 = clampi(offset_x, 0, stitched.cols);
    int y0 = clampi(offset_y, 0, stitched.rows);
    int x1 = clampi(offset_x + tile.cols, 0, stitched.cols);
    int y1 = clampi(offset_y + tile.rows, 0, stitched.rows);
    if (x1 <= x0 || y1 <= y0) return;

    cv::Rect dst(x0, y0, x1 - x0, y1 - y0);
    cv::Rect src(x0 - offset_x, y0 - offset_y, dst.width, dst.height);
    tile(src).copyTo(stitched(dst));
}

void paste_region_tile_drive(
    const Region& r,
    const std::unordered_map<std::pair<double,double>, cv::Mat, PairHash>& tiles,
    cv::Mat& stitched,
    double min_x,
    double max_y,
    double scale,
    bool flip_x
) {
    auto key = std::make_pair(r.center.x, r.center.y);
    auto it = tiles.find(key);
    if (it == tiles.end()) {
        std::cerr << "[WARN] Missing cached tile for region at ("
                  << r.center.x << ", " << r.center.y << ")\n";
        return;
    }

    const cv::Mat& tile = it->second;
    const int tile_px = tile.cols;

    int offset_x = static_cast<int>(
        std::floor((r.center.x - r.size * 0.5 - min_x) * scale));
    int offset_y = static_cast<int>(
        std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));

    if (flip_x) {
        offset_x = stitched.cols - offset_x - tile_px;
    }

    int x0 = clampi(offset_x, 0, stitched.cols);
    int y0 = clampi(offset_y, 0, stitched.rows);
    int x1 = clampi(offset_x + tile.cols, 0, stitched.cols);
    int y1 = clampi(offset_y + tile.rows, 0, stitched.rows);
    if (x1 <= x0 || y1 <= y0) return;

    cv::Rect dst(x0, y0, x1 - x0, y1 - y0);
    cv::Rect src(0, 0, dst.width, dst.height);
    tile(src).copyTo(stitched(dst));
}

/*                                                                                 
            HOW TO RUN EXECUTABLE:

            Join docker:
            docker compose build
            docker compose run --rm dev 

            Export your API key in the docker:
            export IAI_API_KEY="your_key_here"
            
            Build:
            bazel build //examples:log_replay_example

            To run:
            ./bazel-bin/examples/log_replay_example

*/
int main(int argc, char** argv) {
    LogReader log_reader;
    std::cout << "here\n";
    log_reader.read_log("invertedai/carla_Town03_log.json");
    std::cout << "read";
    const int total_num_agents = log_reader.get_total_num_agents();
    const std::string location = log_reader.get_location();
    const std::vector<AgentProperties> all_agent_properties = log_reader.get_agent_properties();
    const std::vector<std::vector<AgentState>> all_agent_states = log_reader.get_agent_states_over_time();
    const std::optional<std::vector<std::map<std::string, std::string>>> all_traffic_lights = log_reader.get_traffic_lights_states_over_time();
    const int sim_length = log_reader.get_scenario_length();
    bool FLIP_X_FOR_THIS_DOMAIN = false; 
    if (location.rfind("carla:", 0) == 0) {
        FLIP_X_FOR_THIS_DOMAIN = true;
    }
    const std::string API_KEY = getenv("IAI_API_KEY"); // in the docker - 'export IAI_API_KEY="your key here"'

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
        log_reader.get_location(),
        total_num_agents,                              
        agent_count_dict,                              
        session,
        std::make_pair(900/2.f, 900/2.f), 
        map_center,                                    
        seed                               
    );

    std::map<AgentType,int> agent_count_dict_drive = {
        {AgentType::car, 500} // a lot of agents to initialize every tile 
    };
    std::vector<Region> drive_tiles = get_regions_default(
        log_reader.get_location(),
        500,                               //  a lot of agents to initialize every tile 
        agent_count_dict_drive,                              
        session,
        std::make_pair(900/2.f, 900/2.f), 
        map_center,                        // map center from location_info
        seed                               // random seed

    );
    
    cv::Rect2d bounds = compute_bounds_rect(drive_tiles);
    const double scale = get_render_scale(li_res, drive_tiles.front());
    const int canvas_w = static_cast<int>(std::ceil(bounds.width * scale));
    const int canvas_h = static_cast<int>(std::ceil(bounds.height * scale));
    std::unordered_map<std::pair<double,double>, cv::Mat, PairHash> drive_cached_tiles = cache_region_tiles_for_drive(
        session, location, drive_tiles, scale
    );
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
    cv::VideoWriter writer("log_replay.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        10, // fps
        cv::Size(canvas_w, canvas_h));
        for (int step = 0; step < sim_length; ++step) {   

            const double scale = get_render_scale(li_res, drive_tiles.front());
            cv::Rect2d bounds = compute_bounds_rect(drive_tiles);
        
            const double min_x = bounds.x;
            const double min_y = bounds.y;
            const double max_x = bounds.x + bounds.width;
            const double max_y = bounds.y + bounds.height;
        
            const int canvas_w = static_cast<int>(std::ceil(bounds.width * scale));
            const int canvas_h = static_cast<int>(std::ceil(bounds.height * scale));
        
            cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
        

            for (size_t i = 0; i < drive_tiles.size(); ++i) {
                paste_region_tile_drive(
                    drive_tiles[i],
                    drive_cached_tiles,
                    stitched,
                    min_x,
                    max_y,
                    scale,
                    FLIP_X_FOR_THIS_DOMAIN
                );
            }
        
            cv::Scalar agent_color(0,0,0);  // black

            // take states directly from the log or from drive_cfg
            const std::vector<AgentState>& current_states = all_agent_states[step];
        
            for (size_t agent_idx = 0; agent_idx < current_states.size(); agent_idx++) {

                const AgentState& s = current_states[agent_idx];
                const AgentProperties& props = all_agent_properties[agent_idx];
                double l = props.length.value_or(5.0);
                double w = props.width.value_or(1.9);
                if (props.agent_type == "pedestrian") {
                    l = 1.5;
                    w = 1.5;
                }
                double psi = s.orientation;
                double hl = l * 0.5;
                double hw = w * 0.5;
            
                double c = std::cos(psi);
                double ss = std::sin(psi);
            
                auto rot = [&](double px, double py) {
                    return cv::Point2d(
                        s.x + c * px - ss * py,
                        s.y + ss * px + c * py
                    );
                };
            
                cv::Point2d FL = rot( hl,  hw);
                cv::Point2d FR = rot( hl, -hw);
                cv::Point2d RR = rot(-hl, -hw);
                cv::Point2d RL = rot(-hl,  hw);
            
                auto to_px = [&](const cv::Point2d& P) {
                    int u = static_cast<int>(std::llround((P.x - min_x) * scale));
                    int v = static_cast<int>(std::llround((max_y - P.y) * scale));
            
                    if (FLIP_X_FOR_THIS_DOMAIN) {
                        u = canvas_w - u; 
                    }
            
                    return cv::Point(
                        std::clamp(u, 0, canvas_w - 1),
                        std::clamp(v, 0, canvas_h - 1)
                    );
                };
            
                std::vector<cv::Point> poly(4);
                poly[0] = to_px(FL);
                poly[1] = to_px(FR);
                poly[2] = to_px(RR);
                poly[3] = to_px(RL);
            
                cv::fillConvexPoly(stitched, poly, cv::Scalar(255, 0, 0)); 
            }

            std::optional<std::map<std::string, std::string>> traffic_lights_states;
            auto traff_all = log_reader.get_traffic_lights_states_over_time();
            if (traff_all.has_value()) {
                traffic_lights_states = traff_all->at(step);
            } else {
                traffic_lights_states = std::nullopt;
            }
        
            std::map<std::string, cv::Point> traffic_light_positions_px =
                get_traffic_light_positions(li_res, min_x, max_y, scale, canvas_w, FLIP_X_FOR_THIS_DOMAIN);
        
            draw_traffic_lights(stitched, traffic_lights_states, traffic_light_positions_px, li_res.static_actors(), FLIP_X_FOR_THIS_DOMAIN, scale);

            cv::putText(stitched,
                        "Step " + std::to_string(step),
                        cv::Point(20,40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0,0,0),
                        2);
        

            writer.write(stitched);
        }
        std::cout << "Video saved to log_replay.avi\n";
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
    



