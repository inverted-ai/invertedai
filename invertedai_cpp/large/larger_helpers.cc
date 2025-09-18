#include "large_helpers.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include "invertedai/error.h"
#include "invertedai/api.h" // location_info
#include "invertedai/data_utils.h"    // decode birdview image
#include "invertedai/location_info_response.h" // LocationInfoResponse
#include "invertedai/location_info_request.h"  // LocationInfoRequest
#include <opencv2/opencv.hpp>


namespace invertedai {

    std::vector<AgentProperties> get_default_agent_properties(size_t num_agents) {
        std::vector<AgentProperties> props;
        props.reserve(num_agents);
        for (size_t i = 0; i < num_agents; i++) {
            AgentProperties car;
            car.agent_type = "car";   // this has been hardcoded!!! may be a problem for future extension
            //can also set dimensions if you want (length, width, rear_axis_offset)
            props.push_back(car);
        }
        return props;
    }

    invertedai::AgentProperties make_default_car() {
        invertedai::AgentProperties car;
        car.agent_type = "car";
        car.length = 4.5;           // ??? not sure if these are
        car.width = 2.0;
        car.rear_axis_offset = 1.5;
        return car;
    }

    std::vector<Region> get_regions_default(
        const std::string& location,
        int total_num_agents,
        Session& session,
        std::optional<std::pair<float,float>> area_shape,
        std::pair<float,float> map_center,
        std::optional<int> random_seed,
        bool display_progress_bar
    ) {
        if (total_num_agents <= 0) {
            throw std::invalid_argument("Must request at least 1 agent.");
        }
    
        // Default to 50x50 half-widths if not provided
        if (!area_shape.has_value()) {
            area_shape = {50.0f, 50.0f};
        }
    
        // if caller didn’t pass a real map_center, set one relative to area_shape
        if (map_center.first == 0.0f && map_center.second == 0.0f) {
            map_center = { area_shape->first / 2.0f, area_shape->second / 2.0f };
        }
    
        // Step 1: Create grid regions
        std::vector<Region> regions = get_regions_in_grid(
            area_shape->first,
            area_shape->second,
            map_center,
            50.0f  // stride
        );
    
        // Step 2: Assign agents proportionally to drivable area
        return get_number_of_agents_per_region_by_drivable_area(
            location,
            regions,
            total_num_agents,
            session,
            random_seed,
            display_progress_bar
        );
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
    int total_num_agents,
    Session& session,
    std::optional<int> random_seed,
    bool display_progress_bar
) {
    // Step 1: Build agent list
    std::map<AgentType,int> agent_count_dict;
    agent_count_dict.emplace(AgentType::car, total_num_agents);

    std::vector<AgentType> agent_list_types;
    for (auto& [atype, num] : agent_count_dict) {
        for (int i = 0; i < num; i++) {
            agent_list_types.push_back(atype);
        }
    }

    std::vector<Region> new_regions = regions;
    std::vector<double> region_road_area;
    double total_drivable_area_ratio = 0.0;

    std::mt19937 rng(random_seed.value_or(std::random_device{}()));

    // Use the existing session passed in
    for (auto& region : new_regions) {
        std::string loc_body = "{}";
        invertedai::LocationInfoRequest loc_info_req(loc_body);
        loc_info_req.set_location(location);
        loc_info_req.set_rendering_center(std::make_optional(
            std::make_pair(region.center.x, region.center.y)));
        loc_info_req.set_rendering_fov(static_cast<int>(region.size));
        loc_info_req.set_include_map_source(false);
        LocationInfoResponse loc_info_res =
            invertedai::location_info(loc_info_req, &session);

        std::vector<unsigned char> img_bytes = loc_info_res.birdview_image();
        cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);

        int total_pixels = birdview.rows * birdview.cols;

        cv::Mat gray;
        cv::cvtColor(birdview, gray, cv::COLOR_BGR2GRAY);
        int number_of_black_pix = total_pixels - cv::countNonZero(gray);

        double drivable_area_ratio =
            static_cast<double>(total_pixels - number_of_black_pix) / total_pixels;

        region_road_area.push_back(drivable_area_ratio);
        total_drivable_area_ratio += drivable_area_ratio;
    }

    std::discrete_distribution<> dist(
        region_road_area.begin(), region_road_area.end()
    );

    for (auto atype : agent_list_types) {
        int ind = dist(rng);
        AgentProperties minimal_prop;
        minimal_prop.agent_type = agent_type_to_string(atype);  
        new_regions[ind].agent_properties.push_back(minimal_prop);
    }
    for (size_t i = 0; i < new_regions.size(); i++) {
        std::cout << "Region " << i 
                  << " drivable_ratio=" << region_road_area[i] 
                  << " num_props=" << new_regions[i].agent_properties.size() << "\n";
    }
    
    std::vector<Region> filtered;
    for (size_t i = 0; i < new_regions.size(); i++) {
        if (region_road_area[i] > 0.0 && !new_regions[i].agent_properties.empty()) {
            filtered.push_back(new_regions[i]);
        }
    }
    

    return filtered;
}


} // namespace invertedai
