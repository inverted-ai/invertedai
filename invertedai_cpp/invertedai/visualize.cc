#include "visualize.h"

namespace invertedai {
void ScenePlotter::close() {
    if (writer_.isOpened())
        writer_.release();
}
void ScenePlotter::compute_traffic_light_positions() {
    traffic_light_positions_.clear();
    for (const auto& a : static_actors_) {
        if (a.agent_type != "traffic_light")
            continue;
        cv::Point px = projector_(a.x, a.y);
        traffic_light_positions_[std::to_string(a.actor_id)] = px;
    }
}


void ScenePlotter::draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& tl_states
) {
    if (!tl_states.has_value() || tl_states->empty())
        return;
    for (const auto& [light_id, state] : *tl_states) {
        cv::Scalar color(128, 128, 128);
        if (state == "red")    color = cv::Scalar(0,0,255);
        if (state == "yellow") color = cv::Scalar(0,255,255);
        if (state == "green")  color = cv::Scalar(0,255,0);
        auto pos_it = traffic_light_positions_.find(light_id);
        if (pos_it == traffic_light_positions_.end())
            continue;
        cv::Point center_px = pos_it->second;
        const invertedai::StaticMapActor* actor = nullptr;
        for (const auto& a : static_actors_) {
            if (a.agent_type == "traffic_light" &&
                std::to_string(a.actor_id) == light_id) {
                actor = &a;
                break;
            }
        }
        if (!actor) continue;
        double length_m = std::max(1.0, actor->length.value_or(1.0));
        double width_m = std::max(1.0, actor->width.value_or(1.0));
        double pixel_length = length_m * 3.5;
        double pixel_width = width_m * 3.5;
        double psi_deg = -actor->orientation * 180.0 / CV_PI;
        if (flip_x_)
            psi_deg = 180.0 - psi_deg;
        cv::RotatedRect box(center_px, cv::Size2f(pixel_length, pixel_width), psi_deg);
        cv::Point2f pts[4];
        box.points(pts);
        cv::fillConvexPoly(
            frame,
            std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
            color
        );
    }
}

void ScenePlotter::draw_agent(
    cv::Mat& frame,
    const invertedai::AgentState& s,
    const invertedai::AgentProperties& p
) {
    double x = s.x;
    double y = s.y;
    double psi = s.orientation;
    double length_m = p.length.value_or(5.0);
    double width_m = p.width.value_or(2.0);
    if (p.agent_type == "pedestrian") {
        length_m = width_m = 1.5;
    }
    double half_length = length_m * 0.5;
    double half_width = width_m * 0.5;
    double cos_h  = std::cos(psi);
    double sin_h = std::sin(psi);
    auto rot = [&](double px, double py){
        return cv::Point2d(
            x + cos_h*px - sin_h*py,
            y + sin_h*px + cos_h*py
        );
    };
    cv::Point2d front_left_world  = rot( half_length,  half_width );
    cv::Point2d front_right_world = rot( half_length, -half_width );
    cv::Point2d rear_right_world  = rot(-half_length, -half_width );
    cv::Point2d rear_left_world   = rot(-half_length,  half_width );
    cv::Point polygon[4] = {
        projector_(front_left_world.x,  front_left_world.y),
        projector_(front_right_world.x, front_right_world.y),
        projector_(rear_right_world.x,  rear_right_world.y),
        projector_(rear_left_world.x,   rear_left_world.y)
    };
    cv::fillConvexPoly(frame, polygon, 4, cv::Scalar(255,0,0));
}

ScenePlotter::ScenePlotter(
    const LocationInfoResponse& li_res, 
    int fov, 
    std::pair<double, double> rendering_center, 
    bool flip_x
) {
    flip_x_ = flip_x;
    static_actors_ = li_res.static_actors();
    background_ = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    int image_height = background_.rows;
    int image_width  = background_.cols;
    cv::cvtColor(background_, background_, cv::COLOR_BGR2RGB);

    double center_x = rendering_center.first;
    double center_y = rendering_center.second;
    double half = fov * 0.5;

    projector_ = {
        .cx    = center_x,
        .cy    = center_y,
        .min_x = center_x - half,
        .max_y = center_y + half,
        .scale = double(image_height) / fov,
        .flip_x = flip_x,
        .width  = image_width,
        .height = image_height
    };
    compute_traffic_light_positions();
}

void ScenePlotter::initialize_video(
    const std::string& filename, 
    int fps
) {
    if (background_.empty())
        throw std::runtime_error("ScenePlotter: background image is empty.");
    writer_ = cv::VideoWriter(
        filename,
        cv::VideoWriter::fourcc('M','J','P','G'),
        fps,
        cv::Size(background_.cols, background_.rows)
    );
    if (!writer_.isOpened())
        throw std::runtime_error("ScenePlotter: Failed to open video writer in initialize_video().");
}

void ScenePlotter::render_step(
    const std::vector<AgentState>& agent_states,
    const std::vector<AgentProperties>& agent_properties,
    const std::optional<std::map<std::string, std::string>>& tl_states
) {
    if (!writer_.isOpened())
        throw std::runtime_error("ScenePlotter: Failed to open video writer in render_step()");

    cv::Mat frame = background_.clone();

    for (size_t i = 0; i < agent_states.size(); i++) {
        draw_agent(frame, agent_states[i], agent_properties[i]);
    }

    draw_traffic_lights(frame, tl_states);

    writer_.write(frame);
}
} // namespace invertedai