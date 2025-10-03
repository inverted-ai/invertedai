#include "quadtree.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace invertedai {

// -------- QuadTree --------
QuadTree::QuadTree(int capacity, const Region& region)
    : capacity_(capacity),
      region_(region),
      region_buffer_(Region::create_square_region(region.center, region.size + 2 * BUFFER_FOV)),
      leaf_(true),
      northWest_(nullptr),
      northEast_(nullptr),
      southWest_(nullptr),
      southEast_(nullptr) {}

void QuadTree::subdivide() {
    if (region_.size <= 2.0) {  // tweak threshold if needed
        leaf_ = true;
        return;
    }
    // Each new quadrant is half the size of the parent
    double child_size = region_.size / 2.0;
    double half_child = child_size / 2.0;

    // Parent center coordinates
    double cx = region_.center.x;
    double cy = region_.center.y;

    northWest_ = std::make_unique<QuadTree>(
        capacity_, Region::create_square_region({cx - half_child, cy + half_child}, child_size));
    northEast_ = std::make_unique<QuadTree>(
        capacity_, Region::create_square_region({cx + half_child, cy + half_child}, child_size));
    southWest_ = std::make_unique<QuadTree>(
        capacity_, Region::create_square_region({cx - half_child, cy - half_child}, child_size));
    southEast_ = std::make_unique<QuadTree>(
        capacity_, Region::create_square_region({cx + half_child, cy - half_child}, child_size));

    leaf_ = false;
    region_.clear_agents();
    region_buffer_.clear_agents();

    for (const auto& particle : particles_)
        insert_particle_in_leaf_nodes(particle, false);
    for (const auto& particle : particles_buffer_)
        insert_particle_in_leaf_nodes(particle, true);

    particles_.clear();
    particles_buffer_.clear();
}

bool QuadTree::insert_particle_in_leaf_nodes(const QuadTreeAgentInfo& p, bool was_core_inserted) {
    // Decide primary child by position relative to parent center
    auto pick_primary = [&](const Point2d& c, const AgentState& s) -> QuadTree* {
        bool east = (s.x >= c.x);
        bool north = (s.y >= c.y);
        if ( north && !east) return northWest_.get();
        if ( north &&  east) return northEast_.get();
        if (!north && !east) return southWest_.get();
        return                  southEast_.get();
    };

    QuadTree* primary = pick_primary(region_.center, p.agent_state);

    bool inserted = false;

    auto try_child = [&](QuadTree* child, bool as_core) {
        if (!child) return false;
        return child->insert(p, as_core ? false : true);
    };

    // Insert as CORE into exactly one child if we haven't already placed as core
    if (!was_core_inserted) {
        inserted = try_child(primary, /*as_core*/ true) || inserted;
    }

    // Insert into buffers of all children whose BUFFER contains the point
    auto in_buf = [&](const std::unique_ptr<QuadTree>& child) {
        return child && child->region_buffer_.is_inside({p.agent_state.x, p.agent_state.y});
    };

    if (in_buf(northWest_)) inserted = northWest_->insert(p, /*is_particle_placed*/ true) || inserted;
    if (in_buf(northEast_)) inserted = northEast_->insert(p, /*is_particle_placed*/ true) || inserted;
    if (in_buf(southWest_)) inserted = southWest_->insert(p, /*is_particle_placed*/ true) || inserted;
    if (in_buf(southEast_)) inserted = southEast_->insert(p, /*is_particle_placed*/ true) || inserted;

    return inserted;
}

bool QuadTree::insert(const QuadTreeAgentInfo& particle, bool is_particle_placed) {
    const bool is_in_region = region_.is_inside({particle.agent_state.x, particle.agent_state.y});
    const bool is_in_buffer = region_buffer_.is_inside({particle.agent_state.x, particle.agent_state.y});

    if (!is_in_region && !is_in_buffer) {
        return false; // particle doesn't belong to this node at all
    }

    // Helper lambdas to place locally into this node
    auto place_in_region_here = [&]() {
        particles_.push_back(particle);
        region_.insert_agent(
            particle.agent_state,
            particle.agent_properties,
            particle.recurrent_state.has_value() ? *particle.recurrent_state : std::vector<double>{}
        );
        return true; // placed as a primary region particle
    };

    auto place_in_buffer_here = [&]() {
        particles_buffer_.push_back(particle);
        region_buffer_.insert_agent(
            particle.agent_state,
            particle.agent_properties,
            particle.recurrent_state.has_value() ? *particle.recurrent_state : std::vector<double>{}
        );
        return false; // recorded as buffer (for neighbor context)
    };

    // If we're a leaf and still under capacity, place locally.
    if (leaf_ && (particles_.size() + particles_buffer_.size()) < static_cast<size_t>(capacity_)) {
        if (is_in_region && !is_particle_placed) {
            return place_in_region_here();
        } else {
            return place_in_buffer_here();
        }
    }

    // Either we're over capacity or not a leaf: try to subdivide if we’re a leaf.
    if (leaf_) {
        subdivide();

        // IMPORTANT GUARD:
        // If subdivide() couldn't split (size too small), children remain null.
        // In that case, force place locally to avoid dereferencing null children.
        if (leaf_) {
            // We’re still a terminal leaf. Accept the particle here.
            if (is_in_region && !is_particle_placed) {
                return place_in_region_here();
            } else {
                return place_in_buffer_here();
            }
        }
    }

    // Reaching here means we have children; pass particle down.
    return insert_particle_in_leaf_nodes(particle, is_particle_placed);
}
std::vector<Region> QuadTree::get_regions() const {
    if (leaf_)
        return {region_};
    else {
        auto nw = northWest_->get_regions();
        auto ne = northEast_->get_regions();
        auto sw = southWest_->get_regions();
        auto se = southEast_->get_regions();
        nw.insert(nw.end(), ne.begin(), ne.end());
        nw.insert(nw.end(), sw.begin(), sw.end());
        nw.insert(nw.end(), se.begin(), se.end());
        return nw;
    }
}

std::vector<QuadTree*> QuadTree::get_leaf_nodes() {
    if (leaf_) {
        return {this};
    } else {
        std::vector<QuadTree*> result;

        if (northWest_) {
            auto nw = northWest_->get_leaf_nodes();
            result.insert(result.end(), nw.begin(), nw.end());
        }
        if (northEast_) {
            auto ne = northEast_->get_leaf_nodes();
            result.insert(result.end(), ne.begin(), ne.end());
        }
        if (southWest_) {
            auto sw = southWest_->get_leaf_nodes();
            result.insert(result.end(), sw.begin(), sw.end());
        }
        if (southEast_) {
            auto se = southEast_->get_leaf_nodes();
            result.insert(result.end(), se.begin(), se.end());
        }

        return result;
    }
}


size_t QuadTree::get_number_of_agents_in_node() const {
    return particles_.size();
}

// // -------- utility --------
// template <typename T>
// std::vector<T> flatten_and_sort(const std::vector<std::vector<T>>& nested_list,
//                                 const std::vector<int>& index_list) {
//     std::vector<T> flat_list;
//     for (const auto& sublist : nested_list)
//         flat_list.insert(flat_list.end(), sublist.begin(), sublist.end());

//     std::vector<std::pair<int, T>> indexed;
//     int idx = 0;
//     for (const auto& item : flat_list)
//         indexed.emplace_back(index_list[idx++], item);

//     std::sort(indexed.begin(), indexed.end(),
//               [](const auto& a, const auto& b) { return a.first < b.first; });

//     std::vector<T> sorted_list;
//     for (const auto& [i, val] : indexed)
//         sorted_list.push_back(val);

//     return sorted_list;
// }

} // namespace invertedai
