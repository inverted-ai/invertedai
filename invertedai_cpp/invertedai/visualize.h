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
        ScenePlotter(
            const LocationInfoResponse& li_res, 
            bool flip_x=false
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
        std::vector<StaticMapActor> static_actors_;
        std::map<std::string, cv::Point> traffic_light_positions_;
        LocationInfoResponse li_res_;
        void compute_traffic_light_positions();
        void draw_agent(cv::Mat&, const AgentState&, const AgentProperties&);
        void draw_traffic_lights(cv::Mat&, const std::optional<std::map<std::string,std::string>>&);
    };
}
#endif // VISUALIZE_H
