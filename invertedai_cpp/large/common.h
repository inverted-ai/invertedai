#include "data_utils.h"

#include <vector>
#include <optional>

namespace invertedai {
    constexpr int RECURRENT_SIZE = 152;
    constexpr double REGION_MAX_SIZE = 100.0;
    constexpr double AGENT_SCOPE_FOV_BUFFER = 60.0;
    constexpr int ATTEMPT_PER_NUM_REGIONS = 15;
    constexpr int RECURRENT_SIZE = 152;
    // Recurrent state used in drive
    struct RecurrentState {
        std::vector<float> packed;

        RecurrentState() : packed(RECURRENT_SIZE, 0.0f) {}
        explicit RecurrentState(const std::vector<float>& vals) : packed(vals) {
            if (vals.size() != RECURRENT_SIZE) {
                throw std::invalid_argument("RecurrentState must have size 152");
            }
        }
    };
    struct Region {
        Point2d center;
        double size;
        double min_x, max_x, min_y, max_y;
        // std::vector<Agent> agents;

        std::vector<AgentState> agent_states;
        std::vector<AgentProperties> agent_properties;
        std::vector<RecurrentState> recurrent_states;

        // Region copy() const {
        //     Region r(center, size);
        //     r.agents = agents; 
        //     return r;
        // }
        Region copy() const {
            Region r(center, size);
            r.agent_states = agent_states;
            r.agent_properties = agent_properties;
            r.recurrent_states = recurrent_states;
            return r;
        }


        static Region createSquareRegion(
            const Point2d& center,
            double size = 100.0,
            const std::vector<AgentState>& states = {},
            const std::vector<AgentProperties>& props = {},
            const std::vector<RecurrentState>& recurs = {}
        ) {
            if (states.size() != props.size()) {
                throw std::invalid_argument("states and props must have same length");
            }
            if (!recurs.empty() && recurs.size() != states.size()) {
                throw std::invalid_argument("recurs must be empty or same length as states");
            }

            Region region(center, size);
            for (size_t i = 0; i < states.size(); i++) {
                if (!region.is_inside({states[i].x, states[i].y})) {
                    throw std::invalid_argument("Agent state outside region.");
                }
                region.agent_states.push_back(states[i]);
                region.agent_properties.push_back(props[i]);
                if (!recurs.empty()) {
                    region.recurrent_states.push_back(recurs[i]);
                } else {
                    region.recurrent_states.emplace_back(); // default recurrent
                }
            }
            return region;
        }
        Region(
            const Point2d& c, 
            double s
        ) : center(c), size(s) {
            min_x = c.x - s/2;
            max_x = c.x + s/2;
            min_y = c.y - s/2;
            max_y = c.y + s/2;
        }

        //check if point is within an X-Y axis aligned square region
        bool is_inside(const Point2d& p) const {
            return (min_x <= p.x && p.x <= max_x &&
                    min_y <= p.y && p.y <= max_y);
        }
    
    //     void insert_agents(Agent&& agent) {
    //         if (!is_inside({agent.state.x, agent.state.y})) {
    //             throw std::invalid_argument("Agent state outside region");
    //         }
    //         agents.push_back(std::move(agent));
    //     }
    //     void clear_agents() {
    //         agents.clear();
    //     }
        
    // };
    void insert_agent(
        const AgentState& state, 
        const AgentProperties& props, 
        const RecurrentState& recur = RecurrentState()
    ) {
        if (!is_inside({state.x, state.y})) {
            throw std::invalid_argument("Agent state outside region");
        }
        agent_states.push_back(state);
        agent_properties.push_back(props);
        recurrent_states.push_back(recur);
    }
    void clear_agents() {
        agent_states.clear();
        agent_properties.clear();
        recurrent_states.clear();
    }

    size_t size_agents() const {
        return agent_states.size();
    }
};
    
    struct Agent {
        AgentState state;
        AgentProperties properties;
        RecurrentState recurrent;
    };
    
} // namespace invertedai