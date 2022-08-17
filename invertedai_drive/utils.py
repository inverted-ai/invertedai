import requests
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from requests.auth import AuthBase
import invertedai_drive
import os

TIMEOUT_SECS = 600
MAX_RETRIES = 10


class Client:
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

    def run(self, model_inputs: dict) -> dict:
        response = self._request(
            method="post",
            relative_path="/drive",
            json=model_inputs,
        )
        # TODO: Add high-level reponse parser, error handling
        return response.json()

    def initialize(
        self, location, agent_count=10, batch_size=1, min_speed=1, max_speed=3
    ):
        params = {
            "location": location,
            "num_agents_to_spawn": agent_count,
            "num_samples": batch_size,
            "spawn_min_speed": min_speed,
            "spawn_max_speed": max_speed,
        }

        response = self._request(
            method="get",
            relative_path="/initialize",
            params=params,
        )
        # TODO: Add high-level reponse parser, error handling
        return response.json()

    def _request(
        self,
        method,
        relative_path: str = "",
        params=None,
        headers=None,
        json=None,
        data=None,
    ) -> requests.Response:
        try:
            result = self.session.request(
                method=method,
                params=params,
                url=self.base_url + relative_path,
                headers=headers,
                data=data,
                json=json,
            )
        except requests.exceptions.RequestException as e:
            raise e
        # TODO: Add logger
        # TODO: Add low-level reponse parser, error handling
        return result

    def _get_base_url(self) -> str:
        """
        This function returns the endpoint for API calls, which includes the
        version and other endpoint specifications.
        The method path should be appended to the base_url
        """
        if not invertedai_drive.dev:
            base_url = "https://api.inverted.ai/drive"
        else:
            base_url = "http://localhost:8888"
        # TODO: Add endpoint option and versioning to base_url
        return base_url


class APITokenAuth(AuthBase):
    def __init__(self, api_token):
        self.api_token = api_token

    def __call__(self, r):
        r.headers["x-api-key"] = self.api_token
        r.headers["api-key"] = self.api_token
        return r


class Jupyter_Render(widgets.HBox):
    def __init__(self):
        super().__init__()
        output = widgets.Output()
        self.buffer = [np.zeros([128, 128, 3], dtype=np.uint8)]

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
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
