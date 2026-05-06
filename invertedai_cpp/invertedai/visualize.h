#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/opencv.hpp>
#include <optional>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "invertedai/location_info_response.h"
#include "invertedai/data_utils.h"
namespace invertedai {
struct WorldToPixelProjector {
    double cx, cy;
    double min_x, max_y;
    double scale;
    bool flip_x;
    int width, height;

    cv::Point operator()(double x, double y) const {
        int u = int((x - min_x) * scale);
        if (flip_x)
            u = width - u;
        int v = int((max_y - y) * scale);
        return cv::Point(
            std::clamp(u, 0, width  - 1),
            std::clamp(v, 0, height - 1)
        );
    }
};

class ScenePlotter {
    public:
        // rendering_fov_override: the FOV (in metres) the birdview was actually
        // rendered at. LocationInfoResponse::rendering_fov() returns the map's
        // natural map_fov, which is not generally the rendering FOV requested
        // when fetching the birdview, so callers should pass the value they
        // used in LocationInfoRequest::set_rendering_fov() to keep agents
        // aligned with the background.
        // target_resolution: if > 0, the birdview is upscaled to this square
        // resolution so agent rectangles render with sharper edges. Defaults
        // to 0 (use the API's native birdview size).
        // rendering_center_override: the world-coords center the birdview was
        // actually rendered at. LocationInfoResponse::rendering_center() returns
        // the map's natural map_center, which is not generally the center
        // requested when fetching the birdview, so callers should pass the
        // value they used in LocationInfoRequest::set_rendering_center() to
        // keep agents aligned with the background.
        ScenePlotter(
            const LocationInfoResponse& li_res,
            bool flip_x = false,
            std::optional<double> rendering_fov_override = std::nullopt,
            int target_resolution = 0,
            std::optional<Point2d> rendering_center_override = std::nullopt
        );

        void initialize_video(const std::string& filename, int fps = 10);
        void render_step(
            const std::vector<AgentState>& agent_states,
            const std::vector<AgentProperties>& agent_properties,
            const std::optional<std::map<std::string, std::string>>& tl_states
        );
        void close();

    private:
        WorldToPixelProjector projector_;
        cv::Mat background_;
        cv::VideoWriter writer_;
        bool flip_x_;
        double rendering_fov_;
        std::vector<StaticMapActor> static_actors_;
        std::map<std::string, cv::Point> traffic_light_positions_;
        LocationInfoResponse li_res_;
        void compute_traffic_light_positions();
        void draw_agent(cv::Mat&, const AgentState&, const AgentProperties&);
        void draw_traffic_lights(cv::Mat&, const std::optional<std::map<std::string,std::string>>&);
    };
}
#endif // VISUALIZE_H
