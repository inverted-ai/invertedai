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
        double pixel_length = length_m * projector_.scale;
        double pixel_width = width_m * projector_.scale;
        double psi_deg = -actor->orientation * 180.0 / CV_PI;
        if (flip_x_)
            psi_deg = 180.0 - psi_deg;
        cv::RotatedRect box(center_px, cv::Size2f(pixel_length, pixel_width), psi_deg);
        cv::Point2f pts[4];
        box.points(pts);
        cv::fillConvexPoly(
            frame,
            std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
            color,
            cv::LINE_AA
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
    const bool is_pedestrian = (p.agent_type == "pedestrian");
    if (is_pedestrian) {
        length_m = width_m = 1.5;
    }
    double half_length = length_m * 0.5;
    double half_width = width_m * 0.5;
    double cos_h = std::cos(psi);
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
    std::vector<cv::Point> body{
        projector_(front_left_world.x,  front_left_world.y),
        projector_(front_right_world.x, front_right_world.y),
        projector_(rear_right_world.x,  rear_right_world.y),
        projector_(rear_left_world.x,   rear_left_world.y)
    };

    // Colors are written into a buffer that has already been cvtColor'd
    // BGR->RGB, but the VideoWriter / MP4 player interpret the encoded bytes
    // as BGR, so the net rendering convention here is BGR. To match Python's
    // matplotlib RGB tuples, swap (R,G,B)->(B,G,R) when constructing cv::Scalar.
    // Python (utils.py): agent_c = (0.125, 0.29, 0.529), agent_ped_c = (1.0, 0.75, 0.8).
    cv::Scalar fill_color = is_pedestrian
        ? cv::Scalar(204, 191, 255)
        : cv::Scalar(135, 74, 32);
    cv::Scalar edge_color(20, 20, 20);
    int edge_thickness = std::max(1, static_cast<int>(std::round(projector_.scale * 0.08)));

    cv::fillConvexPoly(frame, body, fill_color, cv::LINE_AA);
    cv::polylines(frame, body, true, edge_color, edge_thickness, cv::LINE_AA);

    if (!is_pedestrian) {
        // Equilateral direction marker centered at length/4 forward of the
        // agent centroid, side proportional to width, pointing along heading.
        // Matches the Python visualizer (utils.py: marker_offset = length / 4,
        // numsides=3, c = dir_c = (0.392, 1.0, 1.0) cyan).
        const double centroid_offset = length_m * 0.25;
        const double side = width_m * 0.8;
        const double sqrt3 = std::sqrt(3.0);
        const double tip_dx = side * sqrt3 / 3.0;
        const double base_dx = -side * sqrt3 / 6.0;
        const double base_dy = side * 0.5;
        cv::Point2d tip_w = rot(centroid_offset + tip_dx,  0.0);
        cv::Point2d bl_w  = rot(centroid_offset + base_dx,  base_dy);
        cv::Point2d br_w  = rot(centroid_offset + base_dx, -base_dy);
        std::vector<cv::Point> tri{
            projector_(tip_w.x, tip_w.y),
            projector_(bl_w.x,  bl_w.y),
            projector_(br_w.x,  br_w.y)
        };
        // Python utils.py: dir_c = (0.392, 1.0, 1.0); written here as BGR.
        cv::fillConvexPoly(frame, tri, cv::Scalar(255, 255, 100), cv::LINE_AA);
    }
}

ScenePlotter::ScenePlotter(
    const LocationInfoResponse& li_res,
    bool flip_x,
    std::optional<double> rendering_fov_override,
    int target_resolution
) :
    flip_x_(flip_x),
    li_res_(li_res)
{
    static_actors_ = li_res.static_actors();
    background_ = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(background_, background_, cv::COLOR_BGR2RGB);

    if (target_resolution > 0 &&
        (background_.rows != target_resolution || background_.cols != target_resolution)) {
        cv::Mat resized;
        cv::resize(
            background_,
            resized,
            cv::Size(target_resolution, target_resolution),
            0, 0,
            cv::INTER_CUBIC
        );
        background_ = resized;
    }

    rendering_fov_ = rendering_fov_override.value_or(
        static_cast<double>(li_res.rendering_fov())
    );

    int image_height = background_.rows;
    int image_width  = background_.cols;
    double center_x = li_res.rendering_center().x;
    double center_y = li_res.rendering_center().y;
    double half = rendering_fov_ * 0.5;

    projector_ = {
        .cx     = center_x,
        .cy     = center_y,
        .min_x  = center_x - half,
        .max_y  = center_y + half,
        .scale  = double(image_height) / rendering_fov_,
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
        cv::VideoWriter::fourcc('m','p','4','v'),
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