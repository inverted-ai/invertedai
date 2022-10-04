import requests
import json
import re
from typing import Dict, Optional
import numpy as np
from requests.auth import AuthBase
import invertedai as iai
from invertedai import error
import logging

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
        self.session.auth = APITokenAuth(api_token)

    def request(
        self, model: str, params: Optional[dict] = None, data: Optional[dict] = None
    ):
        method, relative_path = iai.model_resources[model]
        response = self._request(
            method=method,
            relative_path=relative_path,
            params=params,
            json=data,
        )

        return response

    def _request(
        self,
        method,
        relative_path: str = "",
        params=None,
        headers=None,
        json=None,
        data=None,
    ) -> Dict:
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
            raise error.APIConnectionError("Error communicating with IAI") from e
        iai.logger.info(
            iai.logger.logfmt(
                "IAI API response",
                path=self.base_url,
                response_code=result.status_code,
            )
        )
        data = self._interpret_response_line(result)
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
