#define INVERTEDAI_QUADTREE_H

#include <vector>
#include <optional>
#include "common.h"
#include "invertedai/data_utils.h"    
namespace invertedai {

constexpr double BUFFER_FOV = 35.0;
constexpr double QUADTREE_SIZE_BUFFER = 1.0;

struct QuadTreeAgentInfo {
    AgentState agent_state;
    std::optional<std::vector<double>> recurrent_state;  // C++ analog of Optional[RecurrentState]
    AgentProperties agent_properties;
    int agent_id;

    // Return as vector<variant> like Python's tolist()
    std::vector<std::variant<AgentState, AgentProperties, std::optional<std::vector<double>>, int>> 
    tolist() const {
        return {agent_state, agent_properties, recurrent_state, agent_id};
    }

    // Construct from vector<variant> like Python's fromlist()
    static QuadTreeAgentInfo fromlist(
        const std::vector<std::variant<AgentState, AgentProperties, std::optional<std::vector<double>>, int>>& l
    ) {
        if (l.size() != 4) {
            throw std::invalid_argument("fromlist requires exactly 4 elements");
        }
        QuadTreeAgentInfo q;
        q.agent_state      = std::get<AgentState>(l[0]);
        q.agent_properties = std::get<AgentProperties>(l[1]);
        q.recurrent_state  = std::get<std::optional<std::vector<double>>>(l[2]);
        q.agent_id         = std::get<int>(l[3]);
        return q;
    }
};


class QuadTree {
public:
    QuadTree(int capacity, const Region& region);

    bool insert(const QuadTreeAgentInfo& particle, bool is_particle_placed = false);
    std::vector<invertedai::Region> get_regions() const;
    std::vector<QuadTree*> get_leaf_nodes();
    size_t get_number_of_agents_in_node() const;
    const std::vector<QuadTreeAgentInfo>& particles() const { return particles_; }
    const std::vector<QuadTreeAgentInfo>& particles_buffer() const { return particles_buffer_; }
    const Region& region() const { return region_; }

private:
    int capacity_;
    invertedai::Region region_;
    Region region_buffer_;
    bool leaf_;

    std::vector<QuadTreeAgentInfo> particles_;
    std::vector<QuadTreeAgentInfo> particles_buffer_;
    std::unique_ptr<QuadTree> northWest_;
    std::unique_ptr<QuadTree> northEast_;
    std::unique_ptr<QuadTree> southWest_;
    std::unique_ptr<QuadTree> southEast_;

    void subdivide();
    bool insert_particle_in_leaf_nodes(const QuadTreeAgentInfo& particle, bool is_inserted);
};

/// Utility function
template <typename T>
std::vector<T> flatten_and_sort(const std::vector<std::vector<T>>& nested_list,
                                const std::vector<int>& index_list) {
    std::vector<T> result(index_list.size());
    size_t pos = 0;
    for (const auto& sublist : nested_list) {
        for (const auto& elem : sublist) {
            if (pos >= index_list.size()) break;
            result[index_list[pos++]] = elem;
        }
    }
    return result;
}

} // namespace invertedai
