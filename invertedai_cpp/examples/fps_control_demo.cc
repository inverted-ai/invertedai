#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define _USE_MATH_DEFINES
#include <future> //includes std::async library
#include <chrono>
#include <stdexcept>
#include <cmath>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../invertedai/api.h"
#include "../invertedai/data_utils.h"

using tcp = net::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
using json = nlohmann::json; // from <json.hpp>

const unsigned int IAI_FPS = 10;

struct EgoAgentInput {
    std::vector<invertedai::AgentState> ego_states;
    std::vector<invertedai::AgentAttributes> ego_attributes;
};

json get_ego_log(const std::string& file_path) {
  std::ifstream ego_log_file(file_path);
  json example_ego_agent_log = json::parse(ego_log_file);
  return example_ego_agent_log;
}

EgoAgentInput get_ego_agents(const json& example_ego_agent_log){
  EgoAgentInput output_struct;

  invertedai::AgentState current_ego_state;
  current_ego_state.x = example_ego_agent_log.at(0)["json"]["agent_states"][0][0];
  current_ego_state.y = example_ego_agent_log.at(0)["json"]["agent_states"][0][1];
  current_ego_state.orientation = example_ego_agent_log.at(0)["json"]["agent_states"][0][2];
  current_ego_state.speed = example_ego_agent_log.at(0)["json"]["agent_states"][0][3];
  output_struct.ego_states = {current_ego_state};

  invertedai::AgentAttributes ego_attributes(example_ego_agent_log.at(0)["json"]["agent_attributes"][0]);
  output_struct.ego_attributes = {ego_attributes};

  return output_struct;
}

std::vector<std::vector<invertedai::AgentState>> linear_interpolation(const std::vector<invertedai::AgentState>& current_agent_states, const std::vector<invertedai::AgentState>& next_agent_states, const int number_steps) {
  if (current_agent_states.size() != next_agent_states.size()) {
    throw std::runtime_error("Size of vector arguements for interpolation does not match.");
  }

  std::vector<std::vector<invertedai::AgentState>> interpolated_states;
  double number_steps_div = (double)number_steps;
  
  for (int tt = 0; tt < number_steps; tt++) {
    std::vector<invertedai::AgentState> timestep_states;
    for(int i = 0; i < (int)next_agent_states.size(); i++){
      invertedai::AgentState agent_state_interpolated;
      invertedai::AgentState agent_state_current = current_agent_states[i];
      invertedai::AgentState agent_state_next = next_agent_states[i];

      agent_state_interpolated.x = agent_state_current.x + (agent_state_next.x - agent_state_current.x)*(tt/number_steps_div);
      agent_state_interpolated.y = agent_state_current.y + (agent_state_next.y - agent_state_current.y)*(tt/number_steps_div);
      agent_state_interpolated.orientation = agent_state_current.orientation + (agent_state_next.orientation - agent_state_current.orientation)*(tt/number_steps_div);
      agent_state_interpolated.speed = agent_state_current.speed + (agent_state_next.speed - agent_state_current.speed)*(tt/number_steps_div);
      timestep_states.push_back(agent_state_interpolated);
    }
    interpolated_states.push_back(timestep_states);
  }

  return interpolated_states;
}

double angle_wrap(double angle) {
  //Assume angles are given in radians
  const int MAX_ITERATIONS = 1000;
  int num_iteration = 0;
  while ((angle > M_PI) || (angle < -M_PI)) {
    if (angle > M_PI) {
      angle -= 2*M_PI;
    }
    else if (angle < -M_PI) {
      angle += 2*M_PI;
    }
    num_iteration += 1;
    if (num_iteration >= MAX_ITERATIONS) {
      throw std::runtime_error("Exceeded maximum allowable iterations.");
    }
  }

  return angle;
}

double get_angle_difference(double a, double b){
  //Assume angles are wrapped
  //Assume angles are in radians
  //Assume the equation is a - b
  double sub = a -b;
  if (sub > M_PI) {
    sub = 2*M_PI - sub;
  }
  else if (sub < -M_PI) {
    sub = -2*M_PI + sub;
  }

  return sub;
}

std::vector<invertedai::AgentState> extrapolate_ego_agents(const std::vector<invertedai::AgentState>& current_ego_states, const std::vector<invertedai::AgentState>& previous_ego_states){
  std::vector<invertedai::AgentState> estimated_ego_states;
  for(int ag = 0; ag < (int)current_ego_states.size(); ag++){
    invertedai::AgentState ego_agent;
    double angle_difference = get_angle_difference(angle_wrap(current_ego_states[ag].orientation),angle_wrap(previous_ego_states[ag].orientation));
    ego_agent.orientation = current_ego_states[ag].orientation + angle_difference; //Assume constant angular velocity over time period from previous time period
    ego_agent.orientation = angle_wrap(ego_agent.orientation);
    ego_agent.speed = current_ego_states[ag].speed + (current_ego_states[ag].speed - previous_ego_states[ag].speed); //Assume constant acceleration over time period from previous time period

    //Estimate future position of agent based on estimated average speed between start and beginning of time period
    double avg_speed = (current_ego_states[ag].speed + ego_agent.speed)/2;
    ego_agent.x = current_ego_states[ag].x + avg_speed*sin(current_ego_states[ag].orientation)/IAI_FPS;
    ego_agent.y = current_ego_states[ag].y + avg_speed*cos(current_ego_states[ag].orientation)/IAI_FPS;

    estimated_ego_states.push_back(ego_agent);
  }

  return estimated_ego_states;
}

std::vector<invertedai::AgentState> split_npc_and_ego_states(std::vector<invertedai::AgentState>& combined_agent_vector, const int num_ego_agents){
  //Returns the vector of ego states
  //Assume the ego agents are at the beginning of the vector

  std::vector<invertedai::AgentState> ego_states;
  for (int a = 0; a < num_ego_agents; a++) {
    ego_states.push_back(combined_agent_vector.front());
    combined_agent_vector.erase(combined_agent_vector.begin());
  }

  return ego_states;

}


// usage: ./fps_control_demo $location $agent_num $timestep $api_key $FPS
int main(int argc, char **argv) {
  try {
    const std::string location(argv[1]);
    const unsigned int agent_num = std::stoi(argv[2]);
    const unsigned int timestep = std::stoi(argv[3]);
    const std::string api_key(argv[4]);
    const unsigned int FPS = std::stoi(argv[5]);
    if (FPS % IAI_FPS != 0) {
      throw std::invalid_argument("FPS argument must be a multiple of 10.");
    }
    int NUM_INTERP_STEPS = (int) FPS/IAI_FPS;

    net::io_context ioc;
    ssl::context ctx(ssl::context::tlsv12_client);
    // configure connection setting
    invertedai::Session session(ioc, ctx);
    session.set_api_key(api_key);
    session.connect();

    // construct request for getting information about the location
    invertedai::LocationInfoRequest loc_info_req(
        invertedai::read_file("examples/location_info_body.json"));
    loc_info_req.set_location(location);

    // get response of location information
    invertedai::LocationInfoResponse loc_info_res =
        invertedai::location_info(loc_info_req, &session);

    // use opencv to decode and save the bird's eye view image of the simulation
    auto image = cv::imdecode(loc_info_res.birdview_image(), cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int frame_width = image.rows;
    int frame_height = image.cols;
    cv::VideoWriter video("iai-demo.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                          cv::Size(frame_width, frame_height));

    
    //////////////////////////////////////////////////////////////////////////////
    //REPLACE THIS BLOCK OF CODE WITH YOUR OWN EGO AGENT MODEL 
    //Get ego agent initial state and attributes to set into initial conditions
    json example_ego_agent_log = get_ego_log("examples/ubc_roundabout_ego_agent_log.json");
    EgoAgentInput ego_agent_struct = get_ego_agents(example_ego_agent_log);
    std::vector<invertedai::AgentState> current_ego_states = ego_agent_struct.ego_states;
    std::vector<invertedai::AgentAttributes> all_ego_attributes = ego_agent_struct.ego_attributes;
    //////////////////////////////////////////////////////////////////////////////
    const int NUMBER_EGO_AGENTS = (int)all_ego_attributes.size();

    // construct request for initializing the simulation (placing NPCs on the map)
    invertedai::InitializeRequest init_req(invertedai::read_file("examples/conditional_initialize_body.json"));
    // set the location
    init_req.set_location(location);
    // set the number of agents
    init_req.set_num_agents_to_spawn(agent_num);
    std::vector<std::vector<invertedai::AgentState>> current_ego_states_history = {current_ego_states};
    init_req.set_states_history(current_ego_states_history);
    init_req.set_agent_attributes(all_ego_attributes);

    // get the response of simulation initialization
    invertedai::InitializeResponse init_res =
        invertedai::initialize(init_req, &session);
    std::vector<invertedai::AgentState> current_agent_states = init_res.agent_states(); //Should be NPC states only
    split_npc_and_ego_states(current_agent_states,NUMBER_EGO_AGENTS);


    // construct request for stepping the simulation (driving the NPCs)
    invertedai::DriveRequest drive_req(
        invertedai::read_file("examples/drive_body.json"));
    drive_req.set_location(location);
    drive_req.update(init_res);

    //Acquire the next drive states based on the current NPC and ego agent states
    invertedai::DriveResponse next_drive_res = invertedai::drive(drive_req, &session);
    std::vector<invertedai::AgentState> next_agent_states = next_drive_res.agent_states(); //Should be NPC states only
    split_npc_and_ego_states(next_agent_states,NUMBER_EGO_AGENTS);
    drive_req.update(next_drive_res);

    for (int t = 0; t < (int)timestep; t++) {
      std::vector<invertedai::AgentState> drive_states = drive_req.agent_states();
      std::vector<invertedai::AgentState> previous_ego_states = split_npc_and_ego_states(drive_states,NUMBER_EGO_AGENTS);
      std::vector<invertedai::AgentState> estimated_ego_states = extrapolate_ego_agents(current_ego_states, previous_ego_states);
      estimated_ego_states.insert(estimated_ego_states.end(), drive_states.begin(), drive_states.end()); //Concatenate ego states to NPC states
      drive_req.set_agent_states(estimated_ego_states);

      //Get the future IAI agent states while stepping through the higher FPS timesteps
      std::future<invertedai::DriveResponse> drive_res = std::async (std::launch::async,invertedai::drive,std::ref(drive_req),&session);
      std::vector<std::vector<invertedai::AgentState>> interpolated_states = linear_interpolation(current_agent_states,next_agent_states,NUM_INTERP_STEPS);
      
      //////////////////////////////////////////////////////////////////////////////
      //REPLACE THIS BLOCK OF CODE WITH YOUR OWN EGO AGENT MODEL 
      //Example acquiring the ego agent states between current and next IAI timestep
      invertedai::AgentState next_ego_state;
      next_ego_state.x = example_ego_agent_log.at(t+1)["cars"][0]["x"];
      next_ego_state.y = example_ego_agent_log.at(t+1)["cars"][0]["y"];
      next_ego_state.orientation = example_ego_agent_log.at(t+1)["cars"][0]["orientation"];
      std::vector<invertedai::AgentState> next_ego_states = {next_ego_state};
      std::vector<std::vector<invertedai::AgentState>> ego_states = linear_interpolation(current_ego_states,next_ego_states,NUM_INTERP_STEPS);
      
      //Show the time steps between IAI time steps
      std::cout << "Time step: " << t << std::endl;
      for(int i = 0; i < NUM_INTERP_STEPS; i++) {
        std::cout << "Sub time step: " << i << std::endl;
        for(int j = 0; j < (int)ego_states[i].size(); j++){
          std::vector<invertedai::AgentState> timestep_states = ego_states[i];
          std::cout << "Ego Agent State " << j << ": [x: " << timestep_states[j].x << ", y: " << timestep_states[j].y << ", orientation: " << timestep_states[j].orientation << "]" << std::endl;
        }
        for(int j = 0; j < (int)interpolated_states[i].size(); j++){
          std::vector<invertedai::AgentState> timestep_states = interpolated_states[i];
          std::cout << "NPC Agent State " << j << ": [x: " << timestep_states[j].x << ", y: " << timestep_states[j].y << ", orientation: " << timestep_states[j].orientation << "]" << std::endl;
        }
      }

      for(int ag = 0; ag < (int)current_ego_states.size(); ag++) {
        double speed = sqrt(std::pow(next_ego_states[ag].x-current_ego_states[ag].x,2) + std::pow(next_ego_states[ag].y-current_ego_states[ag].y,2))*IAI_FPS;
        current_ego_states[ag] = next_ego_states[ag];
        current_ego_states[ag].speed = speed;
      }
      //////////////////////////////////////////////////////////////////////////////
      
      current_agent_states = next_agent_states;
      next_drive_res = drive_res.get();
      
      auto image = cv::imdecode(next_drive_res.birdview(), cv::IMREAD_COLOR);
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      video.write(image);

      next_agent_states = next_drive_res.agent_states();
      next_agent_states.pop_back(); //Remove the ego agent from these states
      drive_req.update(next_drive_res);
      
    }
    video.release();

  } catch (std::exception const &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;

  }
  return EXIT_SUCCESS;

}
