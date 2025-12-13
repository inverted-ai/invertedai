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
#include "large/visualizer/visualizer_helpers.h"   // defines PairHash



namespace invertedai{
cv::Point world_to_canvas(
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
cv::Rect region_rect_pixels(
    const Region& r,
    double min_x, 
    double max_y, 
    double scale,
    int canvas_w, 
    int canvas_h, 
    bool flip_x
) {
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

void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
    const std::map<std::string, cv::Point>& light_positions_px,
    const std::vector<StaticMapActor>& static_actors,
    bool flip_x,
    double scale
) {
    if (!traffic_lights_states.has_value()) {
        std::cout << "[DEBUG] No traffic_lights_states value.\n";
        return;
    }  

    for (const auto& [light_id, state] : *traffic_lights_states) {
     
        cv::Scalar color;
        if (state == "red") color = cv::Scalar(0,0,255);
        else if (state == "yellow") color = cv::Scalar(0,255,255);
        else if (state == "green") color = cv::Scalar(0,255,0);
        else color = cv::Scalar(128,128,128);

        auto it = light_positions_px.find(light_id);
        if (it == light_positions_px.end()) continue;
        cv::Point center_px = it->second;

        // find corresponding StaticMapActor
        const StaticMapActor* actor = nullptr;
        for (const auto& act : static_actors) {
            if (act.agent_type == "traffic_light" &&
                std::to_string(act.actor_id) == light_id) {
                actor = &act;
                break;
            }
        }
        if (!actor) continue;

        double l_m = std::max(1.0, actor->length.value_or(1.0));
        double w_m = std::max(1.0, actor->width.value_or(1.0));

        double l_px = l_m * scale*1.2;
        double w_px = w_m * scale*1.2;

        double psi_deg = - actor->orientation * 180.0 / CV_PI;
        if (flip_x) psi_deg = 180.0 -psi_deg;

        cv::RotatedRect box(center_px, cv::Size2f(l_px, w_px), psi_deg);
        cv::Point2f v[4];
        box.points(v);
        cv::fillConvexPoly(frame, std::vector<cv::Point>{v[0],v[1],v[2],v[3]}, color);

    }
}

std::map<std::string, cv::Point> get_traffic_light_positions(
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

}