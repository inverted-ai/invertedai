#include "quadtree.h"
#include <cmath>
#include <algorithm>

namespace invertedai {

// -------- QuadTreeAgentInfo --------
std::vector<std::variant<AgentState, AgentProperties, std::optional<RecurrentState>, int>>
QuadTreeAgentInfo::tolist() const {
    return {agent_state, agent_properties, recurrent_state, agent_id};
}

QuadTreeAgentInfo QuadTreeAgentInfo::fromlist(const AgentState& state,
                                              const AgentProperties& props,
                                              const std::optional<RecurrentState>& recur,
                                              int id) {
    return {state, props, recur, id};
}

// -------- QuadTree --------
QuadTree::QuadTree(int capacity, const Region& region)
    : capacity_(capacity),
      region_(region),
      region_buffer_(Region::createSquareRegion(region.center, region.size + 2 * BUFFER_FOV)),
      leaf_(true),
      northWest_(nullptr),
      northEast_(nullptr),
      southWest_(nullptr),
      southEast_(nullptr) {}

void QuadTree::subdivide() {
    // Each new quadrant is half the size of the parent
    double child_size = region_.size / 2.0;
    double half_child = child_size / 2.0;

    // Parent center coordinates
    double cx = region_.center.x;
    double cy = region_.center.y;

    northWest_ = std::make_unique<QuadTree>(
        capacity_, Region::createSquareRegion({cx - half_child, cy + half_child}, child_size));
    northEast_ = std::make_unique<QuadTree>(
        capacity_, Region::createSquareRegion({cx + half_child, cy + half_child}, child_size));
    southWest_ = std::make_unique<QuadTree>(
        capacity_, Region::createSquareRegion({cx - half_child, cy - half_child}, child_size));
    southEast_ = std::make_unique<QuadTree>(
        capacity_, Region::createSquareRegion({cx + half_child, cy - half_child}, child_size));

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

bool QuadTree::insert_particle_in_leaf_nodes(const QuadTreeAgentInfo& particle, bool is_inserted) {
    bool inserted = false;
    inserted = northWest_->insert(particle, is_inserted) || inserted;
    inserted = northEast_->insert(particle, is_inserted) || inserted;
    inserted = southWest_->insert(particle, is_inserted) || inserted;
    inserted = southEast_->insert(particle, is_inserted) || inserted;
    return inserted;
}

bool QuadTree::insert(const QuadTreeAgentInfo& particle, bool is_particle_placed) {
    bool is_in_region = region_.is_inside({particle.agent_state.x, particle.agent_state.y});
    bool is_in_buffer = region_buffer_.is_inside({particle.agent_state.x, particle.agent_state.y});

    if (!is_in_region && !is_in_buffer)
        return false;

    if ((particles_.size() + particles_buffer_.size()) < (size_t)capacity_ && leaf_) {
        if (is_in_region && !is_particle_placed) {
            particles_.push_back(particle);
            region_.insert_agent(particle.agent_state, particle.agent_properties,
                                 particle.recurrent_state.value_or(RecurrentState()));
            return true;
        } else {
            particles_buffer_.push_back(particle);
            region_buffer_.insert_agent(particle.agent_state, particle.agent_properties,
                                        particle.recurrent_state.value_or(RecurrentState()));
            return false;
        }
    } else {
        if (leaf_)
            subdivide();
        return insert_particle_in_leaf_nodes(particle, is_particle_placed);
    }
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
    if (leaf_)
        return {this};
    else {
        auto nw = northWest_->get_leaf_nodes();
        auto ne = northEast_->get_leaf_nodes();
        auto sw = southWest_->get_leaf_nodes();
        auto se = southEast_->get_leaf_nodes();
        nw.insert(nw.end(), ne.begin(), ne.end());
        nw.insert(nw.end(), sw.begin(), sw.end());
        nw.insert(nw.end(), se.begin(), se.end());
        return nw;
    }
}

size_t QuadTree::get_number_of_agents_in_node() const {
    return particles_.size();
}

// -------- utility --------
template <typename T>
std::vector<T> flatten_and_sort(const std::vector<std::vector<T>>& nested_list,
                                const std::vector<int>& index_list) {
    std::vector<T> flat_list;
    for (const auto& sublist : nested_list)
        flat_list.insert(flat_list.end(), sublist.begin(), sublist.end());

    std::vector<std::pair<int, T>> indexed;
    int idx = 0;
    for (const auto& item : flat_list)
        indexed.emplace_back(index_list[idx++], item);

    std::sort(indexed.begin(), indexed.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<T> sorted_list;
    for (const auto& [i, val] : indexed)
        sorted_list.push_back(val);

    return sorted_list;
}

} // namespace invertedai
