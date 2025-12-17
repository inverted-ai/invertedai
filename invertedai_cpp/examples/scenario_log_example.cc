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
#include "invertedai/initialize_request.h"
#include "invertedai/initialize_response.h"
#include "invertedai/drive_request.h"
#include "invertedai/drive_response.h"

using namespace invertedai;

/*                                                                                 
            HOW TO RUN EXECUTABLE:

            1. cd into invertedai_cpp folder

            2. Join docker:
            docker compose build
            docker compose run --rm dev 

            3. Export your API key in the docker:
            export IAI_API_KEY="your_key_here"
            
            4. Build:
            bazel build //examples:scenario_log_example

            5. To run:
            ./bazel-bin/examples/scenario_log_example

*/

void draw_traffic_lights(
    cv::Mat& frame,
    const std::optional<std::map<std::string, std::string>>& tl_states,
    const std::map<std::string, cv::Point>& tl_positions_px,
    const std::vector<StaticMapActor>& actors,
    std::function<cv::Point(double,double)> world_to_pixel,
    bool flip_x
) {
    if (!tl_states.has_value() || tl_states->empty())
        return;

    for (const auto& [light_id, state] : *tl_states) {

        cv::Scalar color = cv::Scalar(128, 128, 128);
        if (state == "red")    color = cv::Scalar(0,0,255);
        if (state == "yellow") color = cv::Scalar(0,255,255);
        if (state == "green")  color = cv::Scalar(0,255,0);

        auto pos_it = tl_positions_px.find(light_id);
        if (pos_it == tl_positions_px.end())
            continue;

        cv::Point center_px = pos_it->second;
        const StaticMapActor* actor = nullptr;
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
        cv::fillConvexPoly(
            frame,
            std::vector<cv::Point>{pts[0], pts[1], pts[2], pts[3]},
            color
        );
    }
}


std::map<std::string, cv::Point> get_traffic_light_positions(
    const std::vector<StaticMapActor>& actors,
    std::function<cv::Point(double,double)> world_to_pixel
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

int main(int argc, char** argv) {
    const std::string API_KEY = getenv("IAI_API_KEY"); 
    LogReader log_reader("examples/can_appleby_line_and_dryden_ave_canada_log.json");
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    invertedai::Session session(ioc, ctx);
    session.set_api_key(API_KEY);
    session.connect();

    const std::string location = log_reader.get_location();
    bool FLIP_X_FOR_THIS_DOMAIN = false; 
    if (location.rfind("carla:", 0) == 0) {
        FLIP_X_FOR_THIS_DOMAIN = true;
    }
    LocationInfoRequest li_req("{}");
    li_req.set_location(location);
    li_req.set_include_map_source(true);
    li_req.set_rendering_fov(log_reader.get_fov());
    li_req.set_rendering_center(log_reader.get_rendering_center());
    LocationInfoResponse li_res = location_info(li_req, &session);
    auto image = cv::imdecode(li_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int frame_width  = image.cols;
    int frame_height = image.rows;
    
    cv::VideoWriter video(
        "scenario_log_replay.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        10,  
        cv::Size(frame_width, frame_height)
    );

    auto rc = log_reader.get_scenario_log().rendering_center;
    if (!rc) {
        std::cerr << "please provide a rendering center in JSON logs\n";
        return 1;
    }

    double cx = rc->first;
    double cy = rc->second;
    double FOV = log_reader.get_fov();
    double half = FOV * 0.5;
    double min_x = cx - half;
    double max_y = cy + half;
    double scale = image.rows / FOV;
    auto world_to_pixel = [&](double x, double y) -> cv::Point {
        int u = int((x - min_x) * scale);
        if (FLIP_X_FOR_THIS_DOMAIN) {
            u = frame_width - u;
        }
        int v = int((max_y - y) * scale);
        return cv::Point(
            std::clamp(u, 0, image.cols - 1),
            std::clamp(v, 0, image.rows - 1)
        );
    };
    
    auto draw_agent = [&](cv::Mat &frame,
        const AgentState &s,
        const AgentProperties &p
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
    };
    log_reader.reset_log();
    do {
        cv::Mat frame = image.clone();

        const auto& states = log_reader.current_agent_states();
        const auto  props  = log_reader.current_agent_properties();

        for (size_t i = 0; i < states.size(); ++i) {
            draw_agent(frame, states[i], *props[i]);
        }

        auto traffic_lights_states = log_reader.current_traffic_lights();

        if (traffic_lights_states.has_value()) {
            std::map<std::string, cv::Point> traffic_light_positions_px =
                get_traffic_light_positions(
                    li_res.static_actors(),
                    world_to_pixel
                );

            draw_traffic_lights(
                frame,
                traffic_lights_states,
                traffic_light_positions_px,
                li_res.static_actors(),
                world_to_pixel,
                FLIP_X_FOR_THIS_DOMAIN
            );
        }

        video.write(frame);

    } while (log_reader.next());

    video.release();

}