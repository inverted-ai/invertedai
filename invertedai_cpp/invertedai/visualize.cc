#include "visualize.h"

namespace invertedai {
void Visualizer::close() {
    if (writer_.isOpened())
        writer_.release();
}
void Visualizer::compute_traffic_light_positions() {
    traffic_light_positions_.clear();
    for (const auto& a : static_actors_) {
        if (a.agent_type != "traffic_light")
            continue;
        cv::Point px = projector_(a.x, a.y);
        traffic_light_positions_[std::to_string(a.actor_id)] = px;
    }
}


void Visualizer::draw_traffic_lights(
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
        double L = std::max(1.0, actor->length.value_or(1.0));
        double W = std::max(1.0, actor->width.value_or(1.0));
        double px_L = L * 3.5;
        double px_W = W * 3.5;
        double psi_deg = -actor->orientation * 180.0 / CV_PI;
        if (flip_x_)
            psi_deg = 180.0 - psi_deg;
        cv::RotatedRect box(center_px, cv::Size2f(px_L, px_W), psi_deg);
        cv::Point2f pts[4];
        box.points(pts);
        cv::fillConvexPoly(
            frame,
            std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
            color
        );
    }
}

void Visualizer::draw_agent(
    cv::Mat& frame,
    const invertedai::AgentState& s,
    const invertedai::AgentProperties& p
) {
    double x = s.x;
    double y = s.y;
    double psi = s.orientation;
    double L = p.length.value_or(5.0);
    double W = p.width.value_or(2.0);
    if (p.agent_type == "pedestrian") {
        L = W = 1.5;
    }
    double hl = L * 0.5;
    double hw = W * 0.5;
    double c  = std::cos(psi);
    double sn = std::sin(psi);
    auto rot = [&](double px, double py){
        return cv::Point2d(
            x + c*px - sn*py,
            y + sn*px + c*py
        );
    };
    cv::Point2d FLw = rot( hl,  hw);
    cv::Point2d FRw = rot( hl, -hw);
    cv::Point2d RRw = rot(-hl, -hw);
    cv::Point2d RLw = rot(-hl,  hw);
    cv::Point poly[4] = {
        projector_(FLw.x, FLw.y),
        projector_(FRw.x, FRw.y),
        projector_(RRw.x, RRw.y),
        projector_(RLw.x, RLw.y)
    };
    cv::fillConvexPoly(frame, poly, 4, cv::Scalar(255,0,0));
}

Visualizer::Visualizer(
    const LocationInfoResponse& li_res, 
    int fov, 
    std::pair<double, double> rendering_center, 
    bool flip_x
) {
    flip_x_ = flip_x;
    static_actors_ = li_res.static_actors();
    background_ = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    int H = background_.rows;
    int W = background_.cols;
    cv::cvtColor(background_, background_, cv::COLOR_BGR2RGB);

    double cx  = rendering_center.first;
    double cy  = rendering_center.second;
    double half = fov * 0.5;

    projector_ = {
        .cx    = cx,
        .cy    = cy,
        .min_x = cx - half,
        .max_y = cy + half,
        .scale = double(H) / fov,
        .flip_x = flip_x,
        .width  = W,
        .height = H
    };
    compute_traffic_light_positions();
}

void Visualizer::initialize_video(const std::string& filename, int fps) {
    if (background_.empty())
        throw std::runtime_error("Visualizer: background image is empty.");
    writer_ = cv::VideoWriter(
        filename,
        cv::VideoWriter::fourcc('M','J','P','G'),
        fps,
        cv::Size(background_.cols, background_.rows)
    );
    if (!writer_.isOpened())
        throw std::runtime_error("Visualizer: Failed to open video writer in initialize_video().");
}

void Visualizer::render_step(
    const std::vector<AgentState>& agent_states,
    const std::vector<AgentProperties>& agent_properties,
    const std::optional<std::map<std::string, std::string>>& tl_states
) {
    if (!writer_.isOpened())
        throw std::runtime_error("Visualizer: Failed to open video writer in render_step()");

    cv::Mat frame = background_.clone();

    for (size_t i = 0; i < agent_states.size(); i++) {
        draw_agent(frame, agent_states[i], agent_properties[i]);
    }

    draw_traffic_lights(frame, tl_states);

    writer_.write(frame);
}
} // namespace invertedai