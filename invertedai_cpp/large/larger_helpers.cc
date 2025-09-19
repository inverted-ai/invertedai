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

    AgentProperties make_default_properties(AgentType type) {
        AgentProperties props;
    
        switch (type) {
            case AgentType::car:
                props.agent_type = "car";
                break;
            case AgentType::pedestrian:
                props.agent_type = "pedestrian";
                break;
            default:
                props.agent_type = "car";
                break;
        }
    
        return props;
    }

    std::vector<AgentProperties> get_default_agent_properties(
        const std::map<AgentType,int>& agent_count_dict
    ) {
        std::vector<AgentProperties> result;
        for (const auto& [atype, count] : agent_count_dict) {
            for (int i = 0; i < count; i++) {
                result.push_back(make_default_properties(atype));
            }
        }
        return result;
    }

    /**
     *
     * Mirrors the Python get_default_agent_properties function.
     *
     * @param agent_count_dict A map of AgentType → number of agents of that type.
     * @param use_agent_properties If true, returns AgentProperties; otherwise returns AgentAttributes.
     * @return std::vector<std::variant<AgentAttributes, AgentProperties>>
     */
    // std::vector<std::variant<AgentAttributes, AgentProperties>>
    // get_default_agent_properties(
    //     const std::map<AgentType,int>& agent_count_dict,
    //     bool use_agent_properties
    // ) {
    //     std::vector<std::variant<AgentAttributes, AgentProperties>> result;

    //     for (const auto& [atype, count] : agent_count_dict) {
    //         for (int i = 0; i < count; i++) {
    //             if (use_agent_properties) {
    //                 // Construct default AgentProperties from type
    //                 AgentProperties props = make_default_agent_properties(atype);
    //                 result.emplace_back(props);
    //             } else {
    //                 // Use legacy AgentAttributes
    //                 // AgentAttributes attrs;
    //                 // attrs = AgentAttributes::from_type(atype); // or .fromlist({atype}) depending on API
    //                 // result.emplace_back(attrs);
    //             }
    //         }
    //     }

    //     return result;
    // }

    std::vector<Region> get_regions_default(
        const std::string& location,
        std::optional<int> total_num_agents,
        std::optional<std::map<AgentType,int>> agent_count_dict,  
        Session& session,
        std::optional<std::pair<float,float>> area_shape,
        std::pair<float,float> map_center,
        std::optional<int> random_seed,
        bool display_progress_bar
    ) {
        if (!agent_count_dict.has_value()) {
            if (!total_num_agents.has_value()) {
                throw InvertedAIError("Must specify either total_num_agents or agent_count_dict.");
            } else {
                // Default to cars
                agent_count_dict = std::map<AgentType,int>{{AgentType::car, total_num_agents.value()}};
            }
        }
    
        // Default area shape if not given
        if (!area_shape.has_value()) {
            area_shape = std::make_pair(50.0f, 50.0f); // 100/2 in Python
        }
    
        // Create regions in a grid
        std::vector<Region> regions = get_regions_in_grid(
            area_shape->first,
            area_shape->second,
            map_center
        );
    
        // Assign agents to regions proportional to drivable area
        std::vector<Region> new_regions = get_number_of_agents_per_region_by_drivable_area(
            location,
            regions,
            total_num_agents,         // total_num_agents (deprecated)
            agent_count_dict,        // agent_count_dict
            session,
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
   
    Session& session,
    std::optional<int> random_seed,
    bool display_progress_bar
)  {
    if (!agent_count_dict.has_value()) {
        if (!total_num_agents.has_value()) {
            throw InvertedAIError("Must specify total_num_agents or agent_count_dict.");
        } else {
            agent_count_dict = std::map<AgentType,int>{{AgentType::car, total_num_agents.value()}};
        }
    }

    // Expand dict into flat agent type list
    std::vector<AgentType> agent_types;
    for (auto& [type, count] : agent_count_dict.value()) {
        for (int i = 0; i < count; ++i) {
            agent_types.push_back(type);
        }
    }
    std::vector<Region> new_regions = regions;
    std::vector<double> region_road_area;
    double total_drivable_area_ratio = 0.0;

    if (random_seed.has_value()) {
        std::srand(random_seed.value());
    }

    std::mt19937 rng(
        random_seed.has_value() ? static_cast<unsigned int>(random_seed.value())
                                : std::random_device{}()
    );

    // Loop over regions and compute drivable area ratio
    for (size_t i = 0; i < new_regions.size(); ++i) {
        const Region& region = new_regions[i];
        std::pair<float, float> center_tuple(region.center.x, region.center.y);

        // Call location_info for this region
        invertedai::LocationInfoRequest loc_req("{}");
        loc_req.set_location(location);
        loc_req.set_rendering_center(center_tuple);
        loc_req.set_rendering_fov(static_cast<int>(region.size));

        LocationInfoResponse loc_res = invertedai::location_info(loc_req, &session);

        // Decode birdview image using OpenCV
        std::vector<unsigned char> img_bytes = loc_res.birdview_image();
        cv::Mat birdview = cv::imdecode(img_bytes, cv::IMREAD_COLOR);

        if (birdview.empty()) {
            throw std::runtime_error("Failed to decode birdview image.");
        }

        long total_num_pixels = birdview.rows * birdview.cols;
        long number_of_black_pix = 0;

        for (int y = 0; y < birdview.rows; ++y) {
            for (int x = 0; x < birdview.cols; ++x) {
                cv::Vec3b pixel = birdview.at<cv::Vec3b>(y, x);
                if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
                    number_of_black_pix++;
                }
            }
        }

        double drivable_area_ratio =
            static_cast<double>(total_num_pixels - number_of_black_pix) /
            static_cast<double>(total_num_pixels);

        total_drivable_area_ratio += drivable_area_ratio;
        region_road_area.push_back(drivable_area_ratio);
    }

    // Compute weights for each region
    std::vector<double> weights;
    weights.reserve(region_road_area.size());
    for (double r : region_road_area) {
        weights.push_back(r / total_drivable_area_ratio);
    }
    //  Distribution across regions
    std::discrete_distribution<int> dist(weights.begin(), weights.end());

   // Assign agents into regions stochastically (Python equivalent)
    for (AgentType t : agent_types) {
        int region_idx = dist(rng);
        new_regions[region_idx].agent_properties.push_back(
            make_default_properties(t)
        );
    }

    // Filter out regions with no agents
    std::vector<Region> filtered_regions;
    for (auto& region : new_regions) {
        if (!region.agent_properties.empty()) {
            filtered_regions.push_back(region);
        }
    }

    if (total_drivable_area_ratio <= 0.0) {
        throw InvertedAIError("No drivable area found in any region.");
    }

    return filtered_regions;
    }


} // namespace invertedai
