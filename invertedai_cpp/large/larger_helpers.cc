#include "large_helpers.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include "invertedai/error.h"
#include "invertedai/api.h" // location_info
#include "data_utils.h"    // decode birdview image
#include "invertedai/location_info_response.h" // LocationInfoResponse
#include "invertedai/location_info_request.h"  // LocationInfoRequest
#include <opencv2/opencv.hpp>


namespace invertedai {

std::vector<Region> get_regions_default(
    const std::string& location,
    std::optional<int> total_num_agents,
    std::optional<std::map<AgentType,int>> agent_count_dict,
    std::optional<std::pair<float,float>> area_shape,
    std::pair<float,float> map_center,
    std::optional<int> random_seed,
    bool display_progress_bar
) {
    if (!agent_count_dict.has_value()) {
        if (!total_num_agents.has_value()) {
            throw InvertedAIError("Must specify a number of agents within the regions.");
        } else {
            agent_count_dict = std::map<AgentType,int>{{AgentType::car, total_num_agents.value()}};
        }
    }

    if (!area_shape.has_value()) {
        area_shape = {50.0f, 50.0f}; // default 100/2
    }

    auto regions = get_regions_in_grid(
        area_shape->first,
        area_shape->second,
        map_center
    );

    auto new_regions = get_number_of_agents_per_region_by_drivable_area(
        location,
        regions,
        total_num_agents,
        agent_count_dict,
        random_seed,
        display_progress_bar
    );

    return new_regions;
}

std::vector<Region> get_regions_in_grid(
    float width,
    float height,
    std::pair<float,float> map_center,
    float stride
) {
    auto check_valid_center = [&](const std::pair<float,float>& center) {
        return (map_center.first - width) < center.first && center.first < (map_center.first + width) &&
               (map_center.second - height) < center.second && center.second < (map_center.second + height);
    };

    auto get_neighbors = [&](const std::pair<float,float>& center) {
        return std::vector<std::pair<float,float>>{
            {center.first + stride, center.second + stride},
            {center.first - stride, center.second + stride},
            {center.first + stride, center.second - stride},
            {center.first - stride, center.second - stride}
        };
    };

    std::vector<std::pair<float,float>> queue = {map_center};
    std::vector<std::pair<float,float>> centers;

    while (!queue.empty()) {
        auto center = queue.back();
        queue.pop_back();

        for (auto neighbor : get_neighbors(center)) {
            if (check_valid_center(neighbor) &&
                std::find(queue.begin(), queue.end(), neighbor) == queue.end() &&
                std::find(centers.begin(), centers.end(), neighbor) == centers.end()) {
                queue.push_back(neighbor);
            }
        }

        if (check_valid_center(center) &&
            std::find(centers.begin(), centers.end(), center) == centers.end()) {
            centers.push_back(center);
        }
    }

    std::vector<Region> regions;
    regions.reserve(centers.size());
    for (auto& center : centers) {
        Point2d p{center.first, center.second};
        regions.push_back(Region::create_square_region(p));
    }

    return regions;
}

std::vector<Region> get_number_of_agents_per_region_by_drivable_area(
    const std::string& location,
    const std::vector<Region>& regions,
    std::optional<int> total_num_agents,
    std::optional<std::map<AgentType,int>> agent_count_dict,
    std::optional<int> random_seed,
    bool display_progress_bar
) {
    if (!agent_count_dict.has_value()) {
        if (!total_num_agents.has_value()) {
            throw InvertedAIError("Must specify a number of agents within the regions.");
        } else {
            agent_count_dict = std::map<AgentType,int>{{AgentType::car, total_num_agents.value()}};
        }
    }

    // Flatten agent types into a list
    std::vector<AgentType> agent_list_types;
    for (auto& [atype, num] : agent_count_dict.value()) {
        for (int i = 0; i < num; i++) {
            agent_list_types.push_back(atype);
        }
    }

    std::vector<Region> new_regions = regions; // shallow copy
    std::vector<double> region_road_area;
    double total_drivable_area_ratio = 0.0;

    std::mt19937 rng(random_seed.value_or(std::random_device{}()));
    // session configuration
    boost::asio::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key("wIvOHtKln43XBcDtLdHdXR3raX81mUE1Hp66ZRni");
    session.connect();

    for (auto& region : new_regions) {
        std::string loc_body = "{}";
        invertedai::LocationInfoRequest loc_info_req(loc_body);
        loc_info_req.set_location(location);
        loc_info_req.set_rendering_center({region.center.x, region.center.y});
        loc_info_req.set_rendering_fov(static_cast<int>(region.size));
        loc_info_req.set_include_map_source(false);
        LocationInfoResponse loc_info_res = invertedai::location_info(loc_info_req, &session);


        // Suppose loc_info_res is your LocationInfoResponse
        std::vector<unsigned char> img_bytes = loc_info_res.birdview_image();

        // Decode the compressed image bytes into a cv::Mat
        cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);

        // birdview is now an H×W×3 BGR image (like NumPy array in Python)
        int total_pixels = birdview.rows * birdview.cols;

        // Count black pixels (same as np.sum(birdview.sum(axis=-1) == 0))
        cv::Mat gray;
        cv::cvtColor(birdview, gray, cv::COLOR_BGR2GRAY);
        int number_of_black_pix = total_pixels - cv::countNonZero(gray);

        double drivable_area_ratio =
            static_cast<double>(total_pixels - number_of_black_pix) / total_pixels;
        cv::Mat birdview = cv::imdecode(birdview_bytes, cv::IMREAD_COLOR);

        int total_pixels = birdview.rows * birdview.cols;
        int number_of_black_pix = cv::countNonZero(
            (birdview.reshape(1) == cv::Vec3b(0,0,0))
        );

        int h = birdview.rows();
        int w = birdview.cols();
        int total_pixels = h * w;

        int num_black = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                auto pixel = birdview.at(y, x);
                if (pixel.r == 0 && pixel.g == 0 && pixel.b == 0) {
                    num_black++;
                }
            }
        }

        double drivable_area_ratio = (double)(total_pixels - num_black) / (double)total_pixels;
        region_road_area.push_back(drivable_area_ratio);
        total_drivable_area_ratio += drivable_area_ratio;
    }

    // Weighted random assignment
    std::discrete_distribution<> dist(
        region_road_area.begin(), region_road_area.end()
    );

    for (auto atype : agent_list_types) {
        int ind = dist(rng);
        auto props = get_default_agent_properties({{atype, 1}});
        new_regions[ind].agent_properties.insert(
            new_regions[ind].agent_properties.end(),
            props.begin(),
            props.end()
        );
    }

    // Filter out empty regions
    std::vector<Region> filtered;
    for (auto& region : new_regions) {
        if (!region.agent_properties.empty()) {
            filtered.push_back(region);
        }
    }

    return filtered;
}

} // namespace invertedai
