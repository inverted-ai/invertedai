import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from invertedai.common import AgentData, AgentID, StaticMapActor, TrafficLightState
from invertedai.utils import (
    AgentTag,
    AgentTagStyle,
    TagStyleConfig,
    FrameData,
)

def rot(rotation):
    """Rotate in 2d"""
    return np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])


@dataclass
class SceneVisualizerConfig:
    """
    Configuration for :class:`SceneVisualizer`.

    resolution:
        Desired output image size as ``(width, height)`` in pixels.
    dpi:
        Dots per inch for the rendered figure.
    left_hand_coordinates:
        Set to ``True`` for CARLA maps where the X-axis is flipped.
    plot_frame_number:
        Whether to overlay the frame index on each animation frame.
    direction_vec:
        Whether to draw a directional arrow on each agent.
    velocity_vec:
        Whether to draw a velocity arrow on each agent.
    display_agent_ids:
        Agent IDs whose label should be rendered. ``None`` means all agents;
        pass an empty list to suppress all labels.
    fov:
        Field of view in metres. When provided alongside ``location``, a new
        ``location_info`` call is made to fetch a correctly-cropped birdview image.
    tag_styles:
        Colour configuration for each :class:`AgentTag`.
    xy_offset:
        Map centre coordinates ``(x, y)`` in metres. Same behaviour as ``fov``.
    location:
        IAI formatted map location string. Required when ``fov`` or ``xy_offset``
        are provided so that a new ``location_info`` call can be made.
    display_waypoints:
        Whether to draw waypoint markers for car agents.
        Set to ``False`` to not displaywaypoints.
    ax:
        An optional ``matplotlib.axes.Axes`` to draw into. A new figure/axes is
        created when ``None``.
    """
    resolution: Tuple[int, int] = (2048, 2048)
    dpi: float = 100
    left_hand_coordinates: bool = False
    plot_frame_number: bool = True
    direction_vec: bool = True
    velocity_vec: bool = False
    fov: float = 100.0
    display_agent_ids: Optional[List[str]] = None
    display_waypoints: bool = True
    tag_styles: TagStyleConfig = field(default_factory=TagStyleConfig)
    xy_offset: Optional[Tuple[float, float]] = None
    location: Optional[str] = None
    ax: Optional[Axes] = None


class SceneVisualizer:
    """
    Lightweight, stateless visualization tool for IAI simulation data.

    Unlike :class:`ScenePlotter`, this class does not record frames internally.
    Instead, the caller passes a fully-assembled ``List[FrameData]`` to
    :func:`visualize`, which animates multi-frame sequences or renders a still
    image when only one frame is provided.

    Parameters
    ----------
    map_image:
        Background image decoded from the birdview map returned by
        :func:`location_info`.
    fov:
        Field of view in metres from :func:`location_info`.
    xy_offset:
        Map centre coordinates ``(x, y)`` in metres from :func:`location_info`.
    static_actors:
        List of :class:`StaticMapActor` objects (e.g. traffic lights) from
        :func:`location_info`.
    cfg:
        Optional :class:`SceneVisualizerConfig`. Defaults are used when ``None``.

    See Also
    --------
    :func:`location_info`
    """

    def __init__(
        self,
        map_image: np.ndarray,
        xy_offset: Tuple[float, float],
        static_actors: List[StaticMapActor],
        fov: Optional[float],
        cfg: Optional[SceneVisualizerConfig] = None,
    ):
        self._cfg = cfg if cfg is not None else SceneVisualizerConfig()

        self._left_hand_coordinates = self._cfg.left_hand_coordinates
        self.tag_styles = self._cfg.tag_styles
        self._dpi = self._cfg.dpi
        self._dpi_scale = 100 / self._dpi
        self._resolution = self._cfg.resolution

        self.map_image = map_image
        self.fov = fov if fov is not None else self._cfg.fov
        self.xy_offset = xy_offset if xy_offset is not None else self._cfg.xy_offset
        self.static_actors = static_actors

        self.traffic_lights = {
            actor.actor_id: actor
            for actor in self.static_actors
            if actor.agent_type == "traffic_light"
        }

        self.extent = (
            -self.fov / 2 + self.xy_offset[0],
            self.fov / 2 + self.xy_offset[0],
            -self.fov / 2 + self.xy_offset[1],
            self.fov / 2 + self.xy_offset[1],
        )

        self.traffic_light_colors = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "yellow": (1.0, 0.8, 0.0),
        }

        self.agent_c = (0.125, 0.29, 0.529)
        self.agent_ped_c = (1.0, 0.75, 0.8)
        self.dir_c = (0.392, 1.0, 1.0)
        self.v_c = (0.2, 0.75, 0.2)

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.waypoint_markers = {}
        self.frame_label = None
        self.current_ax = None

    # Public API
    def visualize(
        self,
        frames: List[FrameData],
        output_name: Optional[str] = None,
    ) -> Optional[FuncAnimation]:
        """
        Render a list of :class:`FrameData` objects.

        When ``frames`` contains a single frame a still image is produced.
        When it contains multiple frames a :class:`FuncAnimation` is returned
        and optionally saved.

        Parameters
        ----------
        frames:
            Ordered list of frames to render. Each frame holds the agent dict,
            optional traffic-light states, and optional per-agent tags.
        output_name:
            Path to save the animation (``'.gif'`` or ``'.mp4'``). Ignored for
            single-frame renders. If ``None`` the animation is returned but not
            saved.

        Returns
        -------
        FuncAnimation or None
        """
        if not frames:
            raise ValueError("frames list is empty, nothing to visualize.")

        if len(frames) == 1:
            self._plot_single_frame(frames[0], output_name=output_name)
            return None

        self._initialize_plot(self._cfg.ax)
        fig = self.current_ax.figure
        fig.set_size_inches(
            w=self._resolution[0] / self._dpi,
            h=self._resolution[1] / self._dpi,
            forward=True,
        )

        def init_func():
            return []

        def animate_fn(i):
            return self._update_frame_to(i, frames[i])

        ani = FuncAnimation(
            fig=fig,
            func=animate_fn,
            frames=np.arange(len(frames)),
            init_func=init_func,
            interval=100,
            blit=True,
        )
        if output_name is not None:
            ext = os.path.splitext(output_name)[1].lower()
            writer = "ffmpeg" if ext == ".mp4" else "pillow"
            ani.save(
                filename=output_name,
                writer=writer,
                dpi=self._dpi,
            )
        return ani

    # Private helpers
    def _plot_single_frame(
        self,
        frame: Optional[FrameData] = None,
        output_name: Optional[str] = None,
    ):
        self._initialize_plot(self._cfg.ax)
        if frame is not None:
            self._update_frame_to(0, frame)
        plt.savefig(
            fname=output_name if output_name is not None else self._cfg.location + "_single_frame.png",
        )

    def _initialize_plot(self, ax=None):
        if ax is None:
            plt.clf()
            ax = plt.gca()

        ax.imshow(
            X=self.map_image,
            extent=self.extent,
        )
        ax.set_xlim(
            left=self.extent[0],
            right=self.extent[1],
        )
        ax.set_ylim(
            bottom=self.extent[2],
            top=self.extent[3],
        )
        self.current_ax = ax

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.waypoint_markers = {}
        self.frame_label = None

    def _update_frame_to(self, frame_idx: int, frame: FrameData):
        self._remove_everything()
        self._update_agents(frame)
        if frame.traffic_lights is not None:
            self._update_traffic_lights(frame)
        self._update_frame_label(frame_idx)
        return self._collect_artists()

    def _remove_everything(self):
        for rect in self.actor_boxes.values():
            rect.set_visible(False)
        for marker in self.waypoint_markers.values():
            elem = marker["marker"]
            if isinstance(elem, list):
                for m in elem:
                    m.set_visible(False)
            else:
                elem.set_visible(False)
        for lines in self.dir_lines.values():
            if isinstance(lines, list):
                for line in lines:
                    line.set_visible(False)
            else:
                lines.set_visible(False)
        for lines in self.v_lines.values():
            if isinstance(lines, list):
                for line in lines:
                    line.set_visible(False)
            else:
                lines.set_visible(False)
        for label in self.box_labels.values():
            label.set_visible(False)
        for rect in self.traffic_light_boxes.values():
            rect.set_visible(False)

    def _update_agents(self, frame: FrameData):
        label_ids = self._cfg.display_agent_ids if self._cfg.display_agent_ids is not None else frame.agents.keys()
        for agent_id, agent_data in frame.agents.items():
            self._update_agent(
                agent_id=agent_id,
                agent_data=agent_data,
                agent_tags=frame.agent_tags,
                show_label=agent_id in label_ids,
            )

    def _update_traffic_lights(self, frame: FrameData):
        for light_id, light_state in frame.traffic_lights.items():
            self._plot_traffic_light(
                light_id=light_id,
                light_state=light_state,
            )

    def _update_frame_label(self, frame_idx: int):
        if self._cfg.plot_frame_number:
            if self.frame_label is None:
                self.frame_label = self.current_ax.text(
                    x=self.extent[0],
                    y=self.extent[2],
                    s=str(frame_idx),
                    c="r",
                    fontsize=18 * self._dpi_scale,
                )
            else:
                self.frame_label.set_text(str(frame_idx))

    def _collect_artists(self) -> list:
        artists = list(self.actor_boxes.values())
        artists.extend(self.traffic_light_boxes.values())
        for lines in self.dir_lines.values():
            if isinstance(lines, list):
                artists.extend(lines)
            else:
                artists.append(lines)
        for lines in self.v_lines.values():
            if isinstance(lines, list):
                artists.extend(lines)
            else:
                artists.append(lines)
        artists.extend(self.box_labels.values())
        for m in self.waypoint_markers.values():
            elem = m["marker"]
            if isinstance(elem, list):
                artists.extend(elem)
            else:
                artists.append(elem)
            artists.append(m["text"])
        if self.frame_label is not None:
            artists.append(self.frame_label)
        return artists

    def _update_agent(
        self,
        agent_id: str,
        agent_data: AgentData,
        agent_tags: Optional[Dict[str, AgentTag]] = None,
        show_label: bool = True,
    ):
        agent = agent_data.state
        agent_properties = agent_data.properties

        l, w = agent_properties.length, agent_properties.width
        if agent_properties.agent_type == "pedestrian":
            l, w = 1.5, 1.5
        x, y = agent.center.x, agent.center.y
        v = agent.speed
        psi = agent.orientation

        if self._left_hand_coordinates:
            x, psi = self._transform_point_to_left_hand_coordinate_frame(
                x=x,
                orientation=psi,
            )

        if self._cfg.velocity_vec:
            box = np.array([
                [0, 0], [l * 0.5, 0],
                [0, 0], [v * 0.5, 0],
            ])
            box = np.matmul(rot(psi), box.T).T + np.array([[x, y]])

        if self._cfg.direction_vec:
            marker_offset = agent_properties.length / 4
            x_data = x + marker_offset * math.cos(psi)
            y_data = y + marker_offset * math.sin(psi)
            marker_data = (3, 0, (-90 + 180 * psi / math.pi))

            if agent_id not in self.dir_lines:
                self.dir_lines[agent_id] = self.current_ax.plot(
                    x_data,
                    y_data,
                    marker=marker_data,
                    markersize=agent_properties.width * (400 / self.fov) * self._dpi_scale,
                    linestyle="None",
                    c=self.dir_c,
                )
            else:
                self.dir_lines[agent_id][0].set_xdata([x_data])
                self.dir_lines[agent_id][0].set_ydata([y_data])
                self.dir_lines[agent_id][0].set_marker(marker_data)

            self.dir_lines[agent_id][0].set_visible(True)

        if self._cfg.velocity_vec:
            if agent_id not in self.v_lines:
                self.v_lines[agent_id] = self.current_ax.plot(
                    box[2:4, 0],
                    box[2:4, 1],
                    lw=1.5,
                    c=self.v_c,
                )[0]
            else:
                self.v_lines[agent_id].set_xdata(box[2:4, 0])
                self.v_lines[agent_id].set_ydata(box[2:4, 1])

            self.v_lines[agent_id].set_visible(True)

        if show_label:
            if agent_id not in self.box_labels:
                self.box_labels[agent_id] = self.current_ax.text(
                    x=x,
                    y=y,
                    s=agent_id,
                    c="w",
                    ha="center",
                    va="center",
                    fontsize=18 * self._dpi_scale * (150 / self.fov),
                )
                self.box_labels[agent_id].set_clip_on(True)
            else:
                self.box_labels[agent_id].set_x(x)
                self.box_labels[agent_id].set_y(y)

            self.box_labels[agent_id].set_visible(True)

        if show_label and self._cfg.display_waypoints and agent_properties.agent_type != "pedestrian":
            self._plot_waypoint(
                agent_id=agent_id,
                agent_data=agent_data,
            )

        lw = 1
        fc = None
        ec = None

        if fc is None or ec is None:
            tag_style = self._resolve_tag_style(agent_id, agent_tags)
            if fc is None:
                if tag_style is not None:
                    fc = tag_style.face_color
                elif agent_properties.agent_type == "pedestrian":
                    fc = self.agent_ped_c
                else:
                    fc = self.agent_c
            if ec is None:
                if tag_style is not None and tag_style.edge_color is not None:
                    ec = tag_style.edge_color
                else:
                    lw = 0
                    ec = fc

        if agent_id in self.actor_boxes:
            rect = self.actor_boxes[agent_id]
            rect.set_xy((x - l / 2, y - w / 2))
            rect.set_width(l)
            rect.set_height(w)
            rect.set_angle(psi * 180 / np.pi)
            rect.set_facecolor(fc)
            rect.set_edgecolor(ec)
            rect.set_linewidth(lw)
        else:
            rect = Rectangle(
                xy=(x - l / 2, y - w / 2),
                width=l,
                height=w,
                angle=psi * 180 / np.pi,
                rotation_point="center",
                fc=fc,
                ec=ec,
                lw=lw,
            )
            rect.set_clip_on(True)
            self.current_ax.add_patch(rect)
            self.actor_boxes[agent_id] = rect

        rect.set_visible(True)

    def _plot_waypoint(self, agent_id: str, agent_data: AgentData):
        pos = self._get_waypoint_position(agent_data)
        if pos is None:
            return
        x_data, y_data = pos
        if agent_id not in self.waypoint_markers:
            self._create_waypoints(agent_id, x_data, y_data)
        else:
            self._update_waypoints(agent_id, x_data, y_data)

    def _get_waypoint_position(self, agent_data: AgentData) -> Optional[Tuple[float, float]]:
        wps = agent_data.properties.waypoints if agent_data.properties else None
        if not wps:
            return None
        wp = wps[0]
        x = float(wp.x)
        y = float(wp.y)
        psi = 0.0
        if self._left_hand_coordinates:
            x, psi = self._transform_point_to_left_hand_coordinate_frame(
                x=x,
                orientation=psi,
            )
        marker_offset = 0.0
        x_data = x + marker_offset * math.cos(psi)
        y_data = y + marker_offset * math.sin(psi)
        return x_data, y_data

    def _create_waypoints(self, agent_id: str, x_data: float, y_data: float):
        self.waypoint_markers[agent_id] = dict()
        self.waypoint_markers[agent_id]["marker"] = self.current_ax.plot(
            x_data,
            y_data,
            marker="o",
            color="saddlebrown",
            markersize=17.0 * self._dpi_scale * (150 / self.fov),
            linestyle="None",
            zorder=6,
        )[0]
        self.waypoint_markers[agent_id]["text"] = self.current_ax.text(
            x=x_data,
            y=y_data,
            s=agent_id,
            c="w",
            ha="center",
            va="center",
            fontsize=18 * self._dpi_scale * (150 / self.fov),
            zorder=6,
        )
        self.waypoint_markers[agent_id]["text"].set_clip_on(True)

    def _update_waypoints(self, agent_id: str, x_data: float, y_data: float):
        marker = self.waypoint_markers[agent_id]["marker"]
        marker.set_xdata([x_data])
        marker.set_ydata([y_data])
        marker.set_marker("o")
        marker.set_visible(True)

        text = self.waypoint_markers[agent_id]["text"]
        text.set_x(x_data)
        text.set_y(y_data)
        text.set_visible(True)

    def _plot_traffic_light(self, light_id, light_state):
        light = self.traffic_lights[light_id]
        x, y = light.center.x, light.center.y
        psi = light.orientation
        l, w = max(light.length, 1.0), max(light.width, 1.0)

        if self._left_hand_coordinates:
            x, psi = self._transform_point_to_left_hand_coordinate_frame(
                x=x,
                orientation=psi,
            )

        color = self.traffic_light_colors[light_state]
        if light_id in self.traffic_light_boxes:
            self.traffic_light_boxes[light_id].set_facecolor(color)
            self.traffic_light_boxes[light_id].set_visible(True)
        else:
            rect = Rectangle(
                xy=(x - l / 2, y - w / 2),
                width=l,
                height=w,
                angle=psi * 180 / np.pi,
                rotation_point="center",
                fc=color,
                lw=0,
            )
            self.current_ax.add_patch(rect)
            self.traffic_light_boxes[light_id] = rect

    def _resolve_tag_style(
        self,
        agent_id: str,
        agent_tags: Optional[Dict[str, AgentTag]],
    ) -> Optional[AgentTagStyle]:
        if agent_tags is None:
            return None
        tag = agent_tags.get(agent_id)
        if tag is None:
            return None
        return self.tag_styles.get(tag)

    def _transform_point_to_left_hand_coordinate_frame(self, x, orientation):
        t_x = 2 * self.xy_offset[0] - x
        if orientation >= 0:
            t_orientation = -orientation + math.pi
        else:
            t_orientation = -orientation - math.pi
        return t_x, t_orientation

