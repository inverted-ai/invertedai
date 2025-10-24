#pragma once
#include <opencv2/opencv.hpp>
#include "visualizer_helpers.h"

namespace invertedai {
struct PairHash {
    size_t operator()(const std::pair<double,double>& p) const noexcept {
        auto h1 = std::hash<double>{}(p.first);
        auto h2 = std::hash<double>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

inline cv::Rect2d compute_bounds_rect(const std::vector<Region>& regions) {
    double min_x =  std::numeric_limits<double>::infinity();
    double min_y =  std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto& r : regions) {
        min_x = std::min(min_x, r.center.x - r.size * 0.5);
        max_x = std::max(max_x, r.center.x + r.size * 0.5);
        min_y = std::min(min_y, r.center.y - r.size * 0.5);
        max_y = std::max(max_y, r.center.y + r.size * 0.5);
    }

    return cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
}

inline double get_render_scale(
    const LocationInfoResponse& li_res,
    const Region& region
) {
    cv::Mat probe_tile = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    if (probe_tile.empty()) {
        throw std::runtime_error("Failed to decode probe birdview for get_render_scale()");
    }

    // Compute pixels per meter
    double scale = static_cast<double>(probe_tile.cols) / region.size;
    if (scale <= 0.0) {
        throw std::runtime_error("Invalid scale computed in get_render_scale()");
    }

    return scale;
}

}