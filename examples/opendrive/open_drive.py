import sys
import csv
import math
from matplotlib.patches import Rectangle
import numpy as np
H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7

# file_name = "Town02.csv"
# file_name = "/home/alireza/iai/drive-sdk/foretelix/examples/opendrive/Town02.csv"
# with open(sys.argv[1]) as f:


def draw_map(plt, file_name):
    with open(file_name) as f:
        reader = csv.reader(f, skipinitialspace=True)
        positions = list(reader)

    ref_x = []
    ref_y = []
    ref_z = []
    ref_h = []

    lane_x = []
    lane_y = []
    lane_z = []
    lane_h = []

    border_x = []
    border_y = []
    border_z = []
    border_h = []

    road_id = []
    road_id_x = []
    road_id_y = []

    road_start_dots_x = []
    road_start_dots_y = []

    road_end_dots_x = []
    road_end_dots_y = []

    lane_section_dots_x = []
    lane_section_dots_y = []

    arrow_dx = []
    arrow_dy = []

    current_road_id = None
    current_lane_id = None
    current_lane_section = None
    new_lane_section = False

    for i in range(len(positions) + 1):

        if i < len(positions):
            pos = positions[i]

        # plot road id before going to next road
        if i == len(positions) or (pos[0] == 'lane' and i > 0 and current_lane_id == '0'):

            if current_lane_section == '0':
                road_id.append(int(current_road_id))
                index = int(len(ref_x[-1])/3.0)
                h = ref_h[-1][index]
                road_id_x.append(
                    ref_x[-1][index] + (text_x_offset * math.cos(h) - text_y_offset * math.sin(h)))
                road_id_y.append(
                    ref_y[-1][index] + (text_x_offset * math.sin(h) + text_y_offset * math.cos(h)))
                road_start_dots_x.append(ref_x[-1][0])
                road_start_dots_y.append(ref_y[-1][0])
                if len(ref_x) > 0:
                    arrow_dx.append(ref_x[-1][1]-ref_x[-1][0])
                    arrow_dy.append(ref_y[-1][1]-ref_y[-1][0])
                else:
                    arrow_dx.append(0)
                    arrow_dy.append(0)

            lane_section_dots_x.append(ref_x[-1][-1])
            lane_section_dots_y.append(ref_y[-1][-1])

        if i == len(positions):
            break

        if pos[0] == 'lane':
            current_road_id = pos[1]
            current_lane_section = pos[2]
            current_lane_id = pos[3]
            if pos[3] == '0':
                ltype = 'ref'
                ref_x.append([])
                ref_y.append([])
                ref_z.append([])
                ref_h.append([])

            elif pos[4] == 'no-driving':
                ltype = 'border'
                border_x.append([])
                border_y.append([])
                border_z.append([])
                border_h.append([])
            else:
                ltype = 'lane'
                lane_x.append([])
                lane_y.append([])
                lane_z.append([])
                lane_h.append([])
        else:
            if ltype == 'ref':
                ref_x[-1].append(float(pos[0]))
                ref_y[-1].append(float(pos[1]))
                ref_z[-1].append(float(pos[2]))
                ref_h[-1].append(float(pos[3]))

            elif ltype == 'border':
                border_x[-1].append(float(pos[0]))
                border_y[-1].append(float(pos[1]))
                border_z[-1].append(float(pos[2]))
                border_h[-1].append(float(pos[3]))
            else:
                lane_x[-1].append(float(pos[0]))
                lane_y[-1].append(float(pos[1]))
                lane_z[-1].append(float(pos[2]))
                lane_h[-1].append(float(pos[3]))

    # p1 = plt.figure(1)

    # plot road ref line segments
    for i in range(len(ref_x)):
        plt.plot(ref_x[i], ref_y[i], linewidth=2.0, color='#BB5555')

    # plot driving lanes in blue
    for i in range(len(lane_x)):
        plt.plot(lane_x[i], lane_y[i], linewidth=1.0, color='#3333BB')

    return None


class ScenePlotter:
    def __init__(self):  # , map_image, fov, xy_offset, static_actors):
        # self.conditional_agents = None
        # self.agent_attributes = None
        # self.traffic_lights_history = None
        # self.agent_states_history = None
        # self.map_image = map_image
        # self.fov = fov
        # self.extent = (- self.fov / 2 + xy_offset[0], self.fov / 2 + xy_offset[0]) + \
        # (- self.fov / 2 + xy_offset[1], self.fov / 2 + xy_offset[1])

        # self.traffic_lights = {static_actor.actor_id: static_actor
        #    for static_actor in static_actors
        #    if static_actor.agent_type == 'traffic-light'}

        # self.traffic_light_colors = {
        # 'red':    (1.0, 0.0, 0.0),
        # 'green':  (0.0, 1.0, 0.0),
        # 'yellow': (1.0, 0.8, 0.0)
        # }

        self.agent_c = (0.2, 0.2, 0.7)
        self.cond_c = (0.75, 0.35, 0.35)
        self.dir_c = (0.9, 0.9, 0.9)
        self.v_c = (0.2, 0.75, 0.2)

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.frame_label = None
        self.current_ax = None

        self.reset_recording()

        self.numbers = False

    def initialize_recording(self, agent_states, agent_attributes, traffic_light_states=None, conditional_agents=None):
        self.agent_states_history = [agent_states]
        self.traffic_lights_history = [traffic_light_states]
        self.agent_attributes = agent_attributes
        if conditional_agents is not None:
            self.conditional_agents = conditional_agents
        else:
            self.conditional_agents = []

    def reset_recording(self):
        self.agent_states_history = []
        self.traffic_lights_history = []
        self.agent_attributes = None
        self.conditional_agents = []

    def record_step(self, agent_states, traffic_light_states=None):
        self.agent_states_history.append(agent_states)
        self.traffic_lights_history.append(traffic_light_states)

    def plot_scene(self, agent_states, agent_attributes, traffic_light_states=None, conditional_agents=None,
                   ax=None, numbers=False, direction_vec=True, velocity_vec=False):
        self.initialize_recording(agent_states, agent_attributes,
                                  traffic_light_states=traffic_light_states,
                                  conditional_agents=conditional_agents)

        self.plot_frame(idx=0, ax=ax, numbers=numbers, direction_vec=direction_vec,
                        velocity_vec=velocity_vec, plot_frame_number=False)

        self.reset_recording()

    def plot_frame(self, idx, ax=None, numbers=False, direction_vec=False, velocity_vec=False, plot_frame_number=False):
        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        self._update_frame_to(idx)

    def animate_scene(self, output_name=None, start_idx=0, end_idx=-1, ax=None,
                      numbers=False, direction_vec=True, velocity_vec=False,
                      plot_frame_number=False):
        self._initialize_plot(ax=ax, numbers=numbers, direction_vec=direction_vec,
                              velocity_vec=velocity_vec, plot_frame_number=plot_frame_number)
        end_idx = len(self.agent_states_history) if end_idx == -1 else end_idx
        fig = self.current_ax.figure

        def animate(i):
            self._update_frame_to(i)

        ani = animation.FuncAnimation(
            fig, animate, np.arange(start_idx, end_idx), interval=100)
        if output_name is not None:
            ani.save(f'{output_name}', writer='pillow')
        return ani

    def _initialize_plot(self, ax=None, numbers=False, direction_vec=True, velocity_vec=False, plot_frame_number=False):
        # if ax is None:
        #     plt.clf()
        #     ax = plt.gca()
        # self.current_ax = ax
        # ax.imshow(self.map_image, extent=self.extent)
        self.current_ax = ax

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.frame_label = None

        self.numbers = numbers
        self.direction_vec = direction_vec
        self.velocity_vec = velocity_vec
        self.plot_frame_number = plot_frame_number

        self._update_frame_to(0)

    def _update_frame_to(self, frame_idx):
        for i, (agent, agent_attribute) in enumerate(zip(self.agent_states_history[frame_idx], self.agent_attributes)):
            self._update_agent(i, agent, agent_attribute)

        if self.traffic_lights_history[frame_idx] is not None:
            for light_id, light_state in self.traffic_lights_history[frame_idx].items():
                self._plot_traffic_light(light_id, light_state)

        if self.plot_frame_number:
            if self.frame_label is None:
                self.frame_label = self.current_ax.text(
                    self.extent[0], self.extent[2], str(frame_idx), c='r', fontsize=18)
            else:
                self.frame_label.set_text(str(frame_idx))

        self.current_ax.set_xlim(*self.extent[0:2])
        self.current_ax.set_ylim(*self.extent[2:4])

    def _update_agent(self, agent_idx, agent, agent_attribute):
        l, w = agent_attribute.length, agent_attribute.width
        x, y = agent.center.x, agent.center.y
        v = agent.speed
        psi = agent.orientation
        box = np.array([
            [0, 0], [l * 0.5, 0],  # direction vector
            [0, 0], [v * 0.5, 0],  # speed vector at (0.5 m / s ) / m
        ])
        box = np.matmul(rot(psi), box.T).T + np.array([[x, y]])
        if self.direction_vec:
            if agent_idx not in self.dir_lines:
                self.dir_lines[agent_idx] = self.current_ax.plot(
                    box[0:2, 0], box[0:2, 1], lw=2.0, c=self.dir_c)[0]  # plot the direction vector
            else:
                self.dir_lines[agent_idx].set_xdata(box[0:2, 0])
                self.dir_lines[agent_idx].set_ydata(box[0:2, 1])

        if self.velocity_vec:
            if agent_idx not in self.v_lines:
                self.v_lines[agent_idx] = self.current_ax.plot(
                    box[2:4, 0], box[2:4, 1], lw=1.5, c=self.v_c)[0]  # plot the speed
            else:
                self.v_lines[agent_idx].set_xdata(box[2:4, 0])
                self.v_lines[agent_idx].set_ydata(box[2:4, 1])
        if self.numbers:
            if agent_idx not in self.box_labels:
                self.box_labels[agent_idx] = self.current_ax.text(
                    x, y, str(agent_idx), c='r', fontsize=18)
                self.box_labels[agent_idx].set_clip_on(True)
            else:
                self.box_labels[agent_idx].set_x(x)
                self.box_labels[agent_idx].set_y(y)

        if agent_idx in self.conditional_agents:
            c = self.cond_c
        else:
            c = self.agent_c

        rect = Rectangle((x - l / 2, y - w / 2), l, w, angle=psi *
                         180 / np.pi, rotation_point='center', fc=c, lw=0)
        if agent_idx in self.actor_boxes:
            self.actor_boxes[agent_idx].remove()
        self.actor_boxes[agent_idx] = rect
        self.actor_boxes[agent_idx].set_clip_on(True)
        self.current_ax.add_patch(self.actor_boxes[agent_idx])

    def _plot_traffic_light(self, light_id, light_state):
        light = self.traffic_lights[light_id]
        x, y = light.center.x, light.center.y
        psi = light.orientation
        l, w = light.length, light.width

        rect = Rectangle((x - l / 2, y - w / 2), l, w, angle=psi * 180 / np.pi,
                         rotation_point='center',
                         fc=self.traffic_light_colors[light_state], lw=0)
        if light_id in self.traffic_light_boxes:
            self.traffic_light_boxes[light_id].remove()
        self.current_ax.add_patch(rect)
        self.traffic_light_boxes[light_id] = rect


def rot(rot):
    """Rotate in 2d"""
    return np.array([[np.cos(rot), -np.sin(rot)],
                     [np.sin(rot),  np.cos(rot)]])
