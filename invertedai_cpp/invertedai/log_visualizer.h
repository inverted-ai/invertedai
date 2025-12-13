#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <optional>
#include <vector>

#include "invertedai/location_info_response.h"
#include "large/visualizer/visualizer_helpers.h"

namespace invertedai {

cv::Point world_to_canvas(
    double x, double y,
    double min_x, double max_y, double scale,
    int canvas_w, bool flip_x);

int clampi(int v, int lo, int hi);

cv::Rect region_rect_pixels(
    const Region& r,
    double min_x,
    double max_y,
    double scale,
    int canvas_w,
    int canvas_h,
    bool flip_x);

void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& traffic_lights_states,
    const std::map<std::string, cv::Point>& light_positions_px,
    const std::vector<StaticMapActor>& static_actors,
    bool flip_x,
    double scale);

std::map<std::string, cv::Point> get_traffic_light_positions(
    const LocationInfoResponse& li_res,
    double min_x,
    double max_y,
    double scale,
    int canvas_w,
    bool flip_x);

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
    double scale);

void paste_region_tile_drive(
    const Region& r,
    const std::unordered_map<std::pair<double,double>, cv::Mat, PairHash>& tiles,
    cv::Mat& stitched,
    double min_x,
    double max_y,
    double scale,
    bool flip_x);

} // namespace invertedai
