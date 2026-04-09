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
    Color,
    ColorList,
    ColorDict,
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
        Agent IDs whose label should be rendered. ``None`` means no labels.
    tag_styles:
        Colour configuration for each :class:`AgentTag`.
    fov:
        Field of view in metres. When provided alongside ``location``, a new
        ``location_info`` call is made to fetch a correctly-cropped birdview image.
    xy_offset:
        Map centre coordinates ``(x, y)`` in metres. Same behaviour as ``fov``.
    location:
        IAI formatted map location string. Required when ``fov`` or ``xy_offset``
        are provided so that a new ``location_info`` call can be made.
    """
    resolution: Tuple[int, int] = (640, 480)
    dpi: float = 100
    left_hand_coordinates: bool = False
    plot_frame_number: bool = False
    direction_vec: bool = True
    velocity_vec: bool = False
    display_agent_ids: Optional[List[str]] = None
    tag_styles: TagStyleConfig = field(default_factory=TagStyleConfig)
    fov: Optional[float] = None
    xy_offset: Optional[Tuple[float, float]] = None
    location: Optional[str] = None


class SceneVisualizer:
    """
    Lightweight, stateless visualization tool for IAI simulation data.

    Unlike :class:`ScenePlotter`, this class does not record frames internally.
    Instead, the caller passes a fully-assembled ``List[FrameData]`` to
    :meth:`animate` or a single :class:`FrameData` to :meth:`plot_single_frame`.

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
        fov: float,
        xy_offset: Tuple[float, float],
        static_actors: List[StaticMapActor],
        cfg: Optional[SceneVisualizerConfig] = None,
    ):
        self._cfg = cfg if cfg is not None else SceneVisualizerConfig()

        self._left_hand_coordinates = self._cfg.left_hand_coordinates
        self.tag_styles = self._cfg.tag_styles
        self._dpi = self._cfg.dpi
        self._dpi_scale = 100 / self._dpi
        self._resolution = self._cfg.resolution

        self.map_image = map_image
        self.fov = fov
        self.xy_offset = xy_offset
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

        # Transient color state (set during animate and/or plot_single_frame)
        self._agent_face_colors: Optional[List[Optional[ColorDict]]] = None
        self._agent_edge_colors: Optional[List[Optional[ColorDict]]] = None

    # Public API
    def animate(
        self,
        frames: List[FrameData],
        output_name: Optional[str] = None,
        ax: Optional[Axes] = None,
        agent_ids: Optional[List[str]] = None,
        agent_face_colors: Optional[Union[ColorDict, List[ColorDict], ColorList, List[ColorList]]] = None,
        agent_edge_colors: Optional[Union[ColorDict, List[ColorDict], ColorList, List[ColorList]]] = None,
    ) -> FuncAnimation:
        """
        Produce an animation from a list of :class:`FrameData` objects.

        Parameters
        frames:
            Ordered list of frames to animate. Each frame holds the agent dict,
            optional traffic-light states, and optional per-agent tags.
        output_name:
            Path to save the animation (``'.gif'`` or ``'.mp4'``). If ``None``
            the animation is returned but not saved.
        ax:
            An existing ``matplotlib.axes.Axes`` to draw into. A new figure/axes
            is created when ``None``.
        agent_ids:
            Agent IDs whose label should be rendered for this animation. Overrides
            ``display_agent_ids`` from :class:`SceneVisualizerConfig` when provided.
        agent_face_colors:
            Optional per-agent fill colours. Accepts a single :class:`ColorDict`
            (applied to every frame), a list of ``ColorDict`` (one per frame), or
            legacy ``ColorList`` / ``List[ColorList]`` formats.
        agent_edge_colors:
            Same format as *agent_face_colors* but for agent border colours.

        Returns
        FuncAnimation
        """
        if not frames:
            raise ValueError("frames list is empty, nothing to animate.")

        if agent_ids is not None:
            self._cfg.display_agent_ids = agent_ids

        agent_face_colors = self._normalize_color_input(agent_face_colors)
        agent_edge_colors = self._normalize_color_input(agent_edge_colors)
        self._validate_agent_style_data(frames, agent_face_colors, agent_edge_colors)

        self._initialize_plot(ax)
        fig = self.current_ax.figure
        fig.set_size_inches(self._resolution[0] / self._dpi, self._resolution[1] / self._dpi, True)

        def init_func():
            return []

        def animate_fn(i):
            return self._update_frame_to(i, frames[i])

        ani = FuncAnimation(
            fig, animate_fn, np.arange(len(frames)),
            init_func=init_func, interval=100, blit=True,
        )
        if output_name is not None:
            ext = os.path.splitext(output_name)[1].lower()
            writer = "ffmpeg" if ext == ".mp4" else "pillow"
            ani.save(output_name, writer=writer, dpi=self._dpi)
        return ani

    def plot_single_frame(
        self,
        frame: Optional[FrameData] = None,
        ax: Optional[Axes] = None,
    ):
        """
        Render a single frame onto a matplotlib axes.

        Parameters
        ----------
        frame:
            The frame to render. When ``None`` only the background map is drawn,
            which is useful for visualizing the map without any agents.
            A :class:`FrameData` with an empty ``agents`` dict is also valid.
        ax:
            An existing ``matplotlib.axes.Axes`` to draw into. A new figure/axes
            is created when ``None``.
        """
        self._agent_face_colors = [None]
        self._agent_edge_colors = [None]
        self._initialize_plot(ax)
        if frame is not None:
            self._update_frame_to(0, frame)

    # Private helpers
    def _initialize_plot(self, ax=None):
        if ax is None:
            plt.clf()
            ax = plt.gca()

        ax.imshow(self.map_image, extent=self.extent)
        ax.set_xlim(*self.extent[0:2])
        ax.set_ylim(*self.extent[2:4])
        self.current_ax = ax

        self.dir_lines = {}
        self.v_lines = {}
        self.actor_boxes = {}
        self.traffic_light_boxes = {}
        self.box_labels = {}
        self.waypoint_markers = {}
        self.frame_label = None

    def _update_frame_to(self, frame_idx: int, frame: FrameData):
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

        for agent_id, agent_data in frame.agents.items():
            self._update_agent(
                agent_id=agent_id,
                agent_data=agent_data,
                frame_idx=frame_idx,
                agent_tags=frame.agent_tags,
            )
            if self._cfg.display_agent_ids is not None and agent_id in self._cfg.display_agent_ids:
                self._plot_waypoint(agent_id, agent_data)

        if frame.traffic_lights is not None:
            for light_id, light_state in frame.traffic_lights.items():
                self._plot_traffic_light(light_id, light_state)

        if self._cfg.plot_frame_number:
            if self.frame_label is None:
                self.frame_label = self.current_ax.text(
                    self.extent[0],
                    self.extent[2],
                    str(frame_idx),
                    c="r",
                    fontsize=18 * self._dpi_scale,
                )
            else:
                self.frame_label.set_text(str(frame_idx))

        # Collect artists for blitting
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
        frame_idx: int,
        agent_tags: Optional[Dict[str, AgentTag]] = None,
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
            x, psi = self._transform_point_to_left_hand_coordinate_frame(x, psi)

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

        if self._cfg.display_agent_ids is not None and agent_id in self._cfg.display_agent_ids:
            if agent_id not in self.box_labels:
                self.box_labels[agent_id] = self.current_ax.text(
                    x,
                    y,
                    agent_id,
                    c="w",
                    ha="center",
                    va="center",
                    fontsize=18 * self._dpi_scale * (110 / self.fov),
                )
                self.box_labels[agent_id].set_clip_on(True)
            else:
                self.box_labels[agent_id].set_x(x)
                self.box_labels[agent_id].set_y(y)

            self.box_labels[agent_id].set_visible(True)

        lw = 1
        face_colors_for_frame = self._agent_face_colors[frame_idx] if self._agent_face_colors else None
        edge_colors_for_frame = self._agent_edge_colors[frame_idx] if self._agent_edge_colors else None
        fc = self._get_color(agent_id, face_colors_for_frame)
        ec = self._get_color(agent_id, edge_colors_for_frame)

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
                (x - l / 2, y - w / 2),
                l,
                w,
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
        wps = agent_data.properties.waypoints if agent_data.properties else None
        if wps is not None and wps:
            wp = wps[0]
            x = float(wp.x)
            y = float(wp.y)
            psi = 0.0
            if self._left_hand_coordinates:
                x, psi = self._transform_point_to_left_hand_coordinate_frame(x, psi)

            marker_offset = 0.0
            x_data = x + marker_offset * math.cos(psi)
            y_data = y + marker_offset * math.sin(psi)
            marker_data = "o"

            if agent_id not in self.waypoint_markers:
                self.waypoint_markers[agent_id] = dict()
                self.waypoint_markers[agent_id]["marker"] = self.current_ax.plot(
                    x_data,
                    y_data,
                    marker=marker_data,
                    color="saddlebrown",
                    markersize=17.0 * self._dpi_scale * (80 / self.fov),
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
                    fontsize=18 * self._dpi_scale * (80 / self.fov),
                    zorder=6,
                )
                self.waypoint_markers[agent_id]["text"].set_clip_on(True)
            else:
                marker = self.waypoint_markers[agent_id]["marker"]
                marker.set_xdata([x_data])
                marker.set_ydata([y_data])
                marker.set_marker(marker_data)
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
            x, psi = self._transform_point_to_left_hand_coordinate_frame(x, psi)

        color = self.traffic_light_colors[light_state]
        if light_id in self.traffic_light_boxes:
            self.traffic_light_boxes[light_id].set_facecolor(color)
            self.traffic_light_boxes[light_id].set_visible(True)
        else:
            rect = Rectangle(
                (x - l / 2, y - w / 2),
                l,
                w,
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

    def _get_color(self, agent_id: str, color_dict: Optional[ColorDict]):
        if color_dict is None or agent_id not in color_dict:
            return None
        c = color_dict[agent_id]
        if c is None:
            return None
        is_good = isinstance(c, tuple) and len(c) == 3 and all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in c)
        if not is_good:
            raise Exception(f"Expected color format is Tuple[float,float,float] with 0 <= float <= 1 but received {c}.")
        return c

    def _transform_point_to_left_hand_coordinate_frame(self, x, orientation):
        t_x = 2 * self.xy_offset[0] - x
        if orientation >= 0:
            t_orientation = -orientation + math.pi
        else:
            t_orientation = -orientation - math.pi
        return t_x, t_orientation

    def _normalize_color_input(self, colors):
        """Convert legacy ColorList (index-based) to ColorDict (keyed by agent ID)."""
        if colors is None:
            return None
        if isinstance(colors, dict):
            return colors
        if isinstance(colors, list) and len(colors) > 0:
            first = colors[0]
            if isinstance(first, dict):
                return colors
            if isinstance(first, list):
                return [
                    {str(i): c for i, c in enumerate(cl) if c is not None} if cl is not None else None
                    for cl in colors
                ]
            if first is None or isinstance(first, tuple):
                return {str(i): c for i, c in enumerate(colors) if c is not None}
        return colors

    def _normalize_colors(self, colors, label, n_frames):
        """Convert color input to a list of optional dicts, one per frame."""
        if colors is None:
            return [None] * n_frames
        if isinstance(colors, dict):
            return [colors] * n_frames
        if isinstance(colors, list):
            assert len(colors) == n_frames, f"Number of {label} time steps does not match number of frames."
            return colors
        raise ValueError(f"Unexpected type for {label}: {type(colors)}")

    def _validate_agent_style_data(self, frames, agent_face_colors, agent_edge_colors):
        """Normalize and store color inputs as List[Optional[ColorDict]], one per frame."""
        n = len(frames)
        self._agent_face_colors = self._normalize_colors(agent_face_colors, "agent face colors", n)
        self._agent_edge_colors = self._normalize_colors(agent_edge_colors, "agent edge colors", n)
