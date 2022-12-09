import requests
import json
import re
from typing import Dict, Optional
from requests.auth import AuthBase
import invertedai as iai
import invertedai.api
import invertedai.api.config
from invertedai import error, api
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import numpy as np

TIMEOUT_SECS = 600
MAX_RETRIES = 10

class Session:
    def __init__(self, api_token: str = ""):
        self.session = requests.Session()
        self.session.auth = APITokenAuth(api_token)
        self.session.mount(
            "https://",
            requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES),
        )
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                # "Connection": "keep-alive",
            }
        )
        self.base_url = self._get_base_url()

    def add_apikey(self, api_token: str = ""):
        if not iai.dev and not api_token:
            raise error.InvalidAPIKeyError("Empty API key received.")
        self.session.auth = APITokenAuth(api_token)
        response = self.session.request(method="get", url=self.base_url)
        if response.status_code != 200:
            url_acd = "https://api.inverted.ai/v0/academic/m1"
            response_acd = self.session.request(method="get", url=url_acd)
            if response_acd.status_code == 200:
                self.base_url = url_acd
                response = response_acd
            elif response_acd.status_code != 403:
                response = response_acd
        if response.status_code == 403:
            raise error.AuthenticationError(
                "Access denied. Please check the provided API key."
            )
        elif response.status_code != 200:
            raise error.APIError("The server is aware that it has erred or is incapable of performing the requested"
                                 " method.")

    def use_mock_api(self, use_mock: bool = True) -> None:
        invertedai.api.config.mock_api = use_mock
        if use_mock:
            iai.logger.warning(
                'Using mock Inverted AI API - predictions will be trivial'
            )

    def request(
        self, model: str, params: Optional[dict] = None, data: Optional[dict] = None
    ):
        method, relative_path = iai.model_resources[model]
        response = self._request(
            method=method,
            relative_path=relative_path,
            params=params,
            json_body=data,
        )

        return response

    def _request(
        self,
        method,
        relative_path: str = "",
        params=None,
        headers=None,
        json_body=None,
        data=None,
    ) -> Dict:
        try:
            result = self.session.request(
                method=method,
                params=params,
                url=self.base_url + relative_path,
                headers=headers,
                data=data,
                json=json_body,
            )
            result.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise error.APIConnectionError("Error communicating with IAI", should_retry=True)
        except requests.exceptions.Timeout as e:
            raise error.APIConnectionError("Error communicating with IAI")
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 403:
                raise error.AuthenticationError(
                    "Access denied. Please check the provided API key."
                )
            elif e.response.status_code in [400, 422]:
                raise error.InvalidRequestError(e.response.text, param="")
            elif e.response.status_code == 404:
                raise error.ResourceNotFoundError(e.response.text)
            elif e.response.status_code == 429:
                raise error.RateLimitError("Throttled")
            elif e.response.status_code == 503:
                raise error.ServiceUnavailableError("Service Unavailable")
            elif 400 <= e.response.status_code < 500:
                raise error.APIError("Invalid request. Please check the sent data again.")
            else:
                raise error.APIError("The server is aware that it has erred or is incapable of performing the requested"
                                     " method.")
        iai.logger.info(
            iai.logger.logfmt(
                "IAI API response",
                path=self.base_url,
                response_code=result.status_code,
            )
        )
        try:
            data = json.loads(result.content)
        except json.decoder.JSONDecodeError:
            raise error.APIError(
                f"HTTP code {result.status_code} from API ({result.content})", result.content, result.status_code,
                headers=result.headers
            )
        return data

    def _get_base_url(self) -> str:
        """
        This function returns the endpoint for API calls, which includes the
        version and other endpoint specifications.
        The method path should be appended to the base_url
        """
        if not iai.dev:
            base_url = "https://api.inverted.ai/v0/aws/m1"
        else:
            base_url = iai.dev_url
        # TODO: Add endpoint option and versioning to base_url
        return base_url

    def _handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
        try:
            error_data = resp["error"]
        except (KeyError, TypeError):
            raise error.APIError(
                "Invalid response object from API: %r (HTTP response code "
                "was %d)" % (rbody, rcode),
                rbody,
                rcode,
                resp,
            )

        if "internal_message" in error_data:
            error_data["message"] += "\n\n" + error_data["internal_message"]

        iai.logger.info(
            iai.logger.logfmt(
                "IAI API error received",
                error_code=error_data.get("code"),
                error_type=error_data.get("type"),
                error_message=error_data.get("message"),
                error_param=error_data.get("param"),
                stream_error=stream_error,
            )
        )

        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            return error.RateLimitError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode in [400, 404, 415]:
            return error.InvalidRequestError(
                error_data.get("message"),
                error_data.get("param"),
                error_data.get("code"),
                rbody,
                rcode,
                resp,
                rheaders,
            )
        elif rcode == 401:
            return error.AuthenticationError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 403:
            return error.PermissionError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 409:
            return error.TryAgain(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        else:
            return error.APIError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )

    def _interpret_response_line(self, result):
        rbody = result.content
        rcode = result.status_code
        rheaders = result.headers

        if rcode == 503:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            data = json.loads(rbody)
        except:
            raise error.APIError(
                f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders
            )
        if "error" in data or not 200 <= rcode < 300:
            raise self._handle_error_response(rbody, rcode, data, rheaders)

        return data


class APITokenAuth(AuthBase):
    def __init__(self, api_token):
        self.api_token = api_token

    def __call__(self, r):
        r.headers["x-api-key"] = self.api_token
        r.headers["api-key"] = self.api_token
        return r


def Jupyter_Render():
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np

    class Jupyter_Render(widgets.HBox):
        def __init__(self):
            super().__init__()
            output = widgets.Output()
            self.buffer = [np.zeros([128, 128, 3], dtype=np.uint8)]

            with output:
                self.fig, self.ax = plt.subplots(
                    constrained_layout=True, figsize=(5, 5)
                )
            self.im = self.ax.imshow(self.buffer[0])
            self.ax.set_axis_off()

            self.fig.canvas.toolbar_position = "bottom"

            self.max = 0
            # define widgets
            self.play = widgets.Play(
                value=0,
                min=0,
                max=self.max,
                step=1,
                description="Press play",
                disabled=False,
            )
            self.int_slider = widgets.IntSlider(
                value=0, min=0, max=self.max, step=1, description="Frame"
            )

            controls = widgets.HBox(
                [
                    self.play,
                    self.int_slider,
                ]
            )
            controls.layout = self._make_box_layout()
            widgets.jslink((self.play, "value"), (self.int_slider, "value"))
            output.layout = self._make_box_layout()

            self.int_slider.observe(self.update, "value")
            self.children = [controls, output]

        def update(self, change):
            self.im.set_data(self.buffer[self.int_slider.value])
            self.fig.canvas.draw()

        def add_frame(self, frame):
            self.buffer.append(frame)
            self.int_slider.max += 1
            self.play.max += 1
            self.int_slider.value = self.int_slider.max
            self.play.value = self.play.max

        def _make_box_layout(self):
            return widgets.Layout(
                border="solid 1px black",
                margin="0px 10px 10px 0px",
                padding="5px 5px 5px 5px",
            )

    return Jupyter_Render()


class IAILogger(logging.Logger):
    def __init__(
        self,
        name: str = "IAILogger",
        level: str = "WARNING",
        consoel: bool = True,
        log_file: bool = False,
    ) -> None:

        level = logging.getLevelName(level)
        log_level = level if type(level) == int else 30
        super().__init__(name, log_level)
        if consoel:
            consoel_handler = logging.StreamHandler()
            self.addHandler(consoel_handler)
        if log_file:
            file_handler = logging.FileHandler("iai.log")
            self.addHandler(file_handler)

    @staticmethod
    def logfmt(message, **params):
        props = dict(message=message, **params)

        def fmt(key, val):
            # Handle case where val is a bytes or bytesarray
            if hasattr(val, "decode"):
                val = val.decode("utf-8")
            # Check if val is already a string to avoid re-encoding into ascii.
            if not isinstance(val, str):
                val = str(val)
            if re.search(r"\s", val):
                val = repr(val)
            # key should already be a string
            if re.search(r"\s", key):
                key = repr(key)
            return f"{key}={val}"

        return " ".join([fmt(key, val) for key, val in sorted(props.items())])


def rot(rot):
    """Rotate in 2d"""
    return np.array([[np.cos(rot), -np.sin(rot)],
                     [np.sin(rot),  np.cos(rot)]])


class ScenePlotter:
    def __init__(self, map_image, fov, xy_offset, static_actors):
        self.conditional_agents = None
        self.agent_attributes = None
        self.traffic_lights_history = None
        self.agent_states_history = None
        self.map_image = map_image
        self.fov = fov
        self.extent = (- self.fov / 2 + xy_offset[0], self.fov / 2 + xy_offset[0]) + \
            (- self.fov / 2 + xy_offset[1], self.fov / 2 + xy_offset[1])

        self.traffic_lights = {static_actor.actor_id: static_actor
                               for static_actor in static_actors
                               if static_actor.agent_type == 'traffic-light'}

        self.traffic_light_colors = {
            'red':    (1.0, 0.0, 0.0),
            'green':  (0.0, 1.0, 0.0),
            'yellow': (1.0, 0.8, 0.0)
        }

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

        ani = animation.FuncAnimation(fig, animate, np.arange(start_idx, end_idx), interval=100)
        if output_name is not None:
            ani.save(f'{output_name}', writer='pillow')
        return ani

    def _initialize_plot(self, ax=None, numbers=False, direction_vec=True, velocity_vec=False, plot_frame_number=False):
        if ax is None:
            plt.clf()
            ax = plt.gca()
        self.current_ax = ax
        ax.imshow(self.map_image, extent=self.extent)

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
                self.frame_label = self.current_ax.text(self.extent[0], self.extent[2], str(frame_idx), c='r', fontsize=18)
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
                self.dir_lines[agent_idx] = self.current_ax.plot(box[0:2,0], box[0:2,1], lw=2.0, c=self.dir_c)[0] # plot the direction vector
            else:
                self.dir_lines[agent_idx].set_xdata(box[0:2,0])
                self.dir_lines[agent_idx].set_ydata(box[0:2,1])

        if self.velocity_vec:
            if agent_idx not in self.v_lines:
                self.v_lines[agent_idx] = self.current_ax.plot(box[2:4,0], box[2:4,1], lw=1.5 , c=self.v_c)[0] # plot the speed
            else:
                self.v_lines[agent_idx].set_xdata(box[2:4,0])
                self.v_lines[agent_idx].set_ydata(box[2:4,1])
        if self.numbers:
            if agent_idx not in self.box_labels:
                self.box_labels[agent_idx] = self.current_ax.text(x, y, str(agent_idx), c='r', fontsize=18)
                self.box_labels[agent_idx].set_clip_on(True)
            else:
                self.box_labels[agent_idx].set_x(x)
                self.box_labels[agent_idx].set_y(y)

        if agent_idx in self.conditional_agents:
            c = self.cond_c
        else:
            c = self.agent_c

        rect = Rectangle((x - l / 2,y - w / 2), l, w, angle=psi * 180 / np.pi, rotation_point='center', fc=c, lw=0)
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

        rect = Rectangle((x - l / 2,y - w / 2), l, w, angle=psi * 180 / np.pi,
                         rotation_point='center',
                         fc=self.traffic_light_colors[light_state], lw=0)
        if light_id in self.traffic_light_boxes:
            self.traffic_light_boxes[light_id].remove()
        self.current_ax.add_patch(rect)
        self.traffic_light_boxes[light_id] = rect
