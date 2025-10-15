#include "quadtree.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace invertedai {

QuadTree::QuadTree(int capacity, const Region& region)
    : capacity_(capacity),
      region_(region),
      region_buffer_(Region::create_square_region(region.center,
                                                  region.size + 2 * BUFFER_FOV)),
      leaf_(true),
      northWest_(nullptr),
      northEast_(nullptr),
      southWest_(nullptr),
      southEast_(nullptr) {}

void QuadTree::subdivide() {
    Region parent = region_;
    double new_size = region_.size / 2.0;
    double new_center_dist = new_size / 2.0;
    double parent_x = parent.center.x;
    double parent_y = parent.center.y;

    Region region_nw = Region::create_square_region({parent_x - new_center_dist,
                                                     parent_y + new_center_dist},
                                                    new_size);
    Region region_ne = Region::create_square_region({parent_x + new_center_dist,
                                                     parent_y + new_center_dist},
                                                    new_size);
    Region region_sw = Region::create_square_region({parent_x - new_center_dist,
                                                     parent_y - new_center_dist},
                                                    new_size);
    Region region_se = Region::create_square_region({parent_x + new_center_dist,
                                                     parent_y - new_center_dist},
                                                    new_size);

    northWest_ = std::make_unique<QuadTree>(capacity_, region_nw);
    northEast_ = std::make_unique<QuadTree>(capacity_, region_ne);
    southWest_ = std::make_unique<QuadTree>(capacity_, region_sw);
    southEast_ = std::make_unique<QuadTree>(capacity_, region_se);

    leaf_ = false;
    region_.clear_agents();
    region_buffer_.clear_agents();

    for (const auto& p : particles_)
        insert_particle_in_leaf_nodes(p, false);
    for (const auto& p : particles_buffer_)
        insert_particle_in_leaf_nodes(p, true);

    particles_.clear();
    particles_buffer_.clear();
}

bool QuadTree::insert_particle_in_leaf_nodes(
    const QuadTreeAgentInfo& particle, 
    bool is_inserted
) {
    bool inserted = northWest_->insert(particle, is_inserted);
    inserted = northEast_->insert(particle, inserted || is_inserted) || inserted;
    inserted = southWest_->insert(particle, inserted || is_inserted) || inserted;
    inserted = southEast_->insert(particle, inserted || is_inserted) || inserted;
    return inserted;
}

bool QuadTree::insert(
    const QuadTreeAgentInfo& particle, 
    bool is_particle_placed
) {
    bool is_in_region = region_.is_inside({particle.agent_state.x, particle.agent_state.y});
    bool is_in_buffer = region_buffer_.is_inside({particle.agent_state.x, particle.agent_state.y});

    if (!is_in_region && !is_in_buffer)
        return false;

    if ((particles_.size() + particles_buffer_.size()) < static_cast<size_t>(capacity_) && leaf_) {
        if (is_in_region && !is_particle_placed) {
            particles_.push_back(particle);
            region_.insert_all_agent_details(particle.agent_state,
                                             particle.agent_properties,
                                             particle.recurrent_state.value_or(std::vector<double>{}));
            return true;
        } else {
            particles_buffer_.push_back(particle);
            region_buffer_.insert_all_agent_details(particle.agent_state,
                                                    particle.agent_properties,
                                                    particle.recurrent_state.value_or(std::vector<double>{}));
            return false;
        }
    }

    if (leaf_) {
        subdivide();
    }

    bool inserted = insert_particle_in_leaf_nodes(particle, is_particle_placed);
    return inserted;
}

std::vector<Region> QuadTree::get_regions() const {
    if (leaf_) return {region_};

    auto nw = northWest_->get_regions();
    auto ne = northEast_->get_regions();
    auto sw = southWest_->get_regions();
    auto se = southEast_->get_regions();

    nw.insert(nw.end(), ne.begin(), ne.end());
    nw.insert(nw.end(), sw.begin(), sw.end());
    nw.insert(nw.end(), se.begin(), se.end());
    return nw;
}

std::vector<QuadTree*> QuadTree::get_leaf_nodes() {
    if (leaf_) return {this};

    std::vector<QuadTree*> result;
    auto nw = northWest_->get_leaf_nodes();
    auto ne = northEast_->get_leaf_nodes();
    auto sw = southWest_->get_leaf_nodes();
    auto se = southEast_->get_leaf_nodes();
    result.insert(result.end(), nw.begin(), nw.end());
    result.insert(result.end(), ne.begin(), ne.end());
    result.insert(result.end(), sw.begin(), sw.end());
    result.insert(result.end(), se.begin(), se.end());
    return result;
}

size_t QuadTree::get_number_of_agents_in_node() const {
    return particles_.size();
}


} // namespace invertedai
