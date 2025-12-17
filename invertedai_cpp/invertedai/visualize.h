#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/opencv.hpp>
#include <optional>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "invertedai/location_info_response.h"
#include "invertedai/initialize_response.h"
#include "invertedai/drive_response.h"

struct WorldToPixelProjector {
    double cx, cy;
    double min_x, max_y;
    double scale;
    bool flip_x;
    int width, height;

    cv::Point operator()(double x, double y) const;
};

std::map<std::string, cv::Point> get_traffic_light_positions(
    const std::vector<invertedai::StaticMapActor>& actors,
    const WorldToPixelProjector& world_to_pixel
);

void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& tl_states,
    const std::map<std::string, cv::Point>& tl_positions_px,
    const std::vector<invertedai::StaticMapActor>& actors,
    const WorldToPixelProjector& world_to_pixel,
    bool flip_x
);

void draw_agent(
    cv::Mat& frame,
    const invertedai::AgentState& s,
    const invertedai::AgentProperties& p,
    const WorldToPixelProjector& world_to_pixel
);

#endif // VISUALIZE_H
