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
#include "large/visualizer/visualizer.h"
#include "large/visualizer/visualizer_helpers.h"


namespace invertedai {
static std::unordered_map<int, cv::Mat> cache_region_tiles_for_initialize(
    Session& session,
    const std::string& location,
    const std::vector<Region>& regions,
    double scale
) {
    std::unordered_map<int, cv::Mat> cached_tiles;
    for (size_t i = 0; i < regions.size(); ++i) {
        const Region& r = regions[i];
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
    return cached_tiles;

}


// parent lookup: which initial 100×100 region contains this point
static int parent_index_of_point(
    const std::vector<Region>& parents, 
    double x, 
    double y
){
    for (size_t i = 0; i < parents.size(); ++i) {
        const Region& p = parents[i];
        const double hx = p.size * 0.5, hy = p.size * 0.5;
        if (x >= p.center.x - hx && x <= p.center.x + hx &&
            y >= p.center.y - hy && y <= p.center.y + hy) {
            return static_cast<int>(i);
        }
    }
    // fallback: nearest parent center 
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
static cv::Scalar color_from_parent_index(
    size_t parent_idx, 
    size_t parent_count
) {
    if (parent_count == 0) return cv::Scalar(200, 200, 200); // neutral gray

    double hue = fmod((180.0 * parent_idx) / std::max<size_t>(1, parent_count), 180.0);
    int sat = 255; // fully saturated
    int val = 255; // maximum brightness

    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]); // BGR order
}

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
    const std::map<std::string, cv::Point>& light_positions_px
) {
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


void visualize_large_initialize(
    const std::string& location,
    Session& session,
    const std::vector<invertedai::Region> final_regions,
    const std::vector<Region>& all_tiles,
    const LocationInfoResponse& li_res,
    bool flip_x
) {
    const double scale = get_render_scale(li_res, final_regions.front());

    // cache the regions tiles 
    auto cached_tiles = cache_region_tiles_for_initialize(session, location, final_regions, scale);

    // canvas construction
    // compute bounds in world meters
    cv::Rect2d bounds = compute_bounds_rect(all_tiles);
    const double min_x = bounds.x;
    const double min_y = bounds.y;
    const double max_x = bounds.x + bounds.width;
    const double max_y = bounds.y + bounds.height;

    const int canvas_w = static_cast<int>(std::ceil(bounds.width * scale));
    const int canvas_h = static_cast<int>(std::ceil(bounds.height * scale));
    cv::Mat stitched(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255,255,255));
    auto clampi = [](int v, int lo, int hi){ return std::max(lo, std::min(v, hi)); };
   
    // paste all tiles for initilize
    std::cerr << "Pasting " << final_regions.size() << " tiles for large_initialize visualization...\n";
    for (size_t i = 0; i < final_regions.size(); ++i) {
        paste_region_tile(
            final_regions[i],
            cached_tiles,
            static_cast<int>(i),
            stitched,
            min_x,
            max_y,
            max_x,
            min_y,
            flip_x,
            scale
        );
    }

    int total_drawn = 0;
    for (size_t i = 0; i < final_regions.size(); ++i) {
        const Region& r = final_regions[i];
        const cv::Scalar color = color_from_parent_index(i, final_regions.size());
    
        // compute tile size in pixels
        const int tile_px = static_cast<int>(std::llround(r.size * scale));
    
        int offset_x, offset_y;
        if (flip_x) {
            int num_cols = static_cast<int>(std::round((max_x - min_x) / r.size));
            int col = static_cast<int>(std::floor((r.center.x - r.size * 0.5 - min_x) / r.size));
            int flipped_col = (num_cols - 1) - col;
    
            offset_x = flipped_col * tile_px;
            offset_y = static_cast<int>(
                std::floor((max_y - (r.center.y + r.size * 0.5)) * scale)
            );
        } else {
            // unflipped behavior
            offset_x = static_cast<int>(std::floor((r.center.x - r.size * 0.5 - min_x) * scale));
            offset_y = static_cast<int>(std::floor((max_y - (r.center.y + r.size * 0.5)) * scale));
        }
    
        int L  = clampi(offset_x, 0, stitched.cols) + 2;
        int T  = clampi(offset_y, 0, stitched.rows) + 2;
        int Rr = clampi(offset_x + tile_px - 1, 0, stitched.cols) - 2;
        int Bb = clampi(offset_y + tile_px - 1, 0, stitched.rows) - 2;
        if (Rr > L && Bb > T) {
            cv::rectangle(stitched, {L, T}, {Rr, Bb}, color, 2, cv::LINE_AA);
        }
    
        for (const auto& s : r.agent_states) {
            int u = static_cast<int>(std::llround((s.x - min_x) * scale));
            int v = static_cast<int>(std::llround((max_y - s.y) * scale));
    
            if (flip_x) {
                u = stitched.cols - u;
            }
    
            if ((unsigned)u < (unsigned)stitched.cols &&
                (unsigned)v < (unsigned)stitched.rows) {
                cv::circle(stitched, {u, v}, 4, color, cv::FILLED, cv::LINE_AA);
                ++total_drawn;
            }
        }
    }
    std::cout << "Total agents drawn from large_initialize: " << total_drawn << "\n";

    // save large_initialization
    cv::imwrite("large_initialize_visualization.png", stitched);
    std::cout << "Saved large_initialize_visualization.png (" << stitched.cols << "x" << stitched.rows << ")\n";
    std::cout << "All done!\n";
}


void visualize_large_drive(
    const LargeDriveConfig& drive_cfg,
    const std::vector<invertedai::Region> leaf_regions,
    const std::vector<invertedai::Region> final_regions,
    const LocationInfoResponse& li_res,
    const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
    const std::vector<Region>& drive_tiles,
    const std::unordered_map<std::pair<double,double>, cv::Mat, invertedai::PairHash>& drive_cached_tiles,
    cv::VideoWriter writer,
    bool flip_x,
    int step
)  {
    const double scale = get_render_scale(
        li_res,
        drive_tiles.front()
    );
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
            flip_x
        );
    }

    auto parent_color_at = [&](double x, double y) -> cv::Scalar {
        int p = parent_index_of_point(final_regions, x, y);
        size_t parent_count = final_regions.size();
        size_t pi = (p < 0 ? 0 : static_cast<size_t>(p));
        return color_from_parent_index(pi, parent_count);
    };
    // overlay quadtree regions, children take their parent color
    for (const Region& r : leaf_regions) {
        cv::Scalar color = 
        parent_color_at(r.center.x, r.center.y);

        cv::Rect rr = region_rect_pixels(
            r, min_x, max_y, scale,
            stitched.cols, stitched.rows,
            flip_x
        );
        if (rr.area() > 0) {
            // small inward inset to avoid covering seams
            cv::rectangle(stitched,
                        {rr.x+1, rr.y+1},
                        {rr.x + rr.width - 2, rr.y + rr.height - 2},
                        color, 2, cv::LINE_AA);
        }

        // draw agents (color by their current parent region) 
        for (const auto& s : r.agent_states) {
            cv::Point pt = world_to_canvas(
                s.x, s.y, min_x, max_y, scale,
                stitched.cols, flip_x
            );
            if (0 <= pt.x && pt.x < stitched.cols &&
                0 <= pt.y && pt.y < stitched.rows) {
                cv::circle(stitched, pt, 4, color, cv::FILLED, cv::LINE_AA);
            }
        }
    }

    std::map<std::string, cv::Point> traffic_light_positions_px = get_traffic_light_positions(li_res, min_x, max_y, scale, canvas_w, flip_x);
    draw_traffic_lights(stitched, traffic_lights_states, traffic_light_positions_px);

    cv::putText(stitched, "Step " + std::to_string(step),
                cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255,0,0), 2);


    // save frame to video
    writer.write(stitched);
}

}