#include "visualize.h"


cv::Point WorldToPixelProjector::operator()(double x, double y) const {
    int u = int((x - min_x) * scale);
    if (flip_x)
        u = width - u;

    int v = int((max_y - y) * scale);

    return cv::Point(
        std::clamp(u, 0, width - 1),
        std::clamp(v, 0, height - 1)
    );
}

std::map<std::string, cv::Point> get_traffic_light_positions(
    const std::vector<invertedai::StaticMapActor>& actors,
    const WorldToPixelProjector& world_to_pixel
) {
    std::map<std::string, cv::Point> out;

    for (const auto& a : actors) {
        if (a.agent_type != "traffic_light")
            continue;

        cv::Point px = world_to_pixel(a.x, a.y);
        out[std::to_string(a.actor_id)] = px;
    }

    return out;
}

void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& tl_states,
    const std::map<std::string, cv::Point>& tl_positions_px,
    const std::vector<invertedai::StaticMapActor>& actors,
    const WorldToPixelProjector& world_to_pixel,
    bool flip_x
) {
    if (!tl_states.has_value() || tl_states->empty())
        return;

    for (const auto& [light_id, state] : *tl_states) {

        cv::Scalar color(128, 128, 128);
        if (state == "red")    color = cv::Scalar(0,0,255);
        if (state == "yellow") color = cv::Scalar(0,255,255);
        if (state == "green")  color = cv::Scalar(0,255,0);

        auto pos_it = tl_positions_px.find(light_id);
        if (pos_it == tl_positions_px.end())
            continue;

        cv::Point center_px = pos_it->second;

        const invertedai::StaticMapActor* actor = nullptr;
        for (const auto& a : actors) {
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
        if (flip_x)
            psi_deg = 180.0 - psi_deg;

        cv::RotatedRect box(center_px, cv::Size2f(px_L, px_W), psi_deg);
        cv::Point2f pts[4];
        box.points(pts);

        cv::fillConvexPoly(frame,
            std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
            color);
    }
}


void draw_agent(
    cv::Mat& frame,
    const invertedai::AgentState& s,
    const invertedai::AgentProperties& p,
    const WorldToPixelProjector& world_to_pixel
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
        world_to_pixel(FLw.x, FLw.y),
        world_to_pixel(FRw.x, FRw.y),
        world_to_pixel(RRw.x, RRw.y),
        world_to_pixel(RLw.x, RLw.y)
    };

    cv::fillConvexPoly(frame, poly, 4, cv::Scalar(255,0,0));
}
