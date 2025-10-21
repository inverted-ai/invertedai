#include "large_init_helpers.h"
#include <cmath>
#include <iostream>
#include "invertedai/error.h"
#include "invertedai/api.h" 
#include "invertedai/data_utils.h"    
#include "invertedai/location_info_response.h" 
#include "invertedai/location_info_request.h"  
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

    std::vector<Region> get_regions_default(
        const std::string& location,
        std::optional<int> total_num_agents,
        std::optional<std::map<AgentType,int>> agent_count_dict,  
        Session& session,
        std::optional<std::pair<float,float>> area_shape,
        std::pair<float,float> map_center,
        std::optional<int> random_seed
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
            area_shape = std::make_pair(50.0f, 50.0f); 
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
            total_num_agents,        
            agent_count_dict,        
            session,
            random_seed
        );
    
        return new_regions;
    }
    
    

    std::vector<Region> get_regions_in_grid(
        float width,
        float height,
        std::pair<float,float> map_center,
        float stride
    ) {
        float half_width  = width  / 2.0f;
        float half_height = height / 2.0f;
        std::vector<Region> regions;

        const float x0 = map_center.first  - half_width;
        const float x1 = map_center.first  + half_width;
        const float y0 = map_center.second - half_height;
        const float y1 = map_center.second + half_height;

        // make stepping robust against float fuzz
        auto nsteps = [](float a, float b, float h) {
            return std::max(0, (int)std::floor((b - a) / h + 0.5f));
        };
        const int nx = nsteps(x0, x1, stride);
        const int ny = nsteps(y0, y1, stride);

        for (int j = 0; j <= ny; ++j) {
            float y = y0 + j * stride;
            for (int i = 0; i <= nx; ++i) {
                float x = x0 + i * stride;
                regions.push_back(Region::create_square_region(Point2d{x, y}));
            }
        }
        return regions;
    }


    std::vector<Region> get_number_of_agents_per_region_by_drivable_area(
        const std::string& location,
        const std::vector<Region>& regions,
        std::optional<int> total_num_agents,
        std::optional<std::map<AgentType,int>> agent_count_dict,
    
        Session& session,
        std::optional<int> random_seed
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
