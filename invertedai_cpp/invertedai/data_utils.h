#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <map>
#include <string>

namespace invertedai {

const std::map<std::string, int> kControlType = {
    {"traffic-light", 0},       {"yield", 1},       {"stop-sign", 2},
    {"traffic-light-actor", 3}, {"yield-actor", 4}, {"stop-sign-actor", 5}};

struct Point2d {
  double x, y;
};

struct AgentState {
  double x, y, orientation, speed;
};

struct AgentAttributes {
  double length, width, rear_axis_offset;
};

struct TrafficLightState {
  std::string id, value;
};

struct InfractionIndicator {
  bool collisions, offroad, wrong_way;
};

struct StaticMapActor {
  int actor_id, agent_type;
  double x, y, orientation, length, width;
  int dependant;
};

std::string read_file(const char *path);

} // namespace invertedai

#endif
