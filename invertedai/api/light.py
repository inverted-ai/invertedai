import time
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validate_arguments
import invertedai as iai

from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import TryAgain

from invertedai.common import TrafficLightStatesDict, TrafficLightState


class LightResponse(BaseModel):
    """
    Response returned from an API call to :func:`iai.light`.
    """
    traffic_lights_states: TrafficLightStatesDict = Field(default=None,
                                                          description="Current traffic lights states, an object "
                                                          "where key is the traffic-light id and value is "
                                                          "the state, i.e., 'green', 'yellow', 'red', or None.")

    recurrent_states: str = Field(default=None,
                                  description="Recurrent states for traffic-lights, obtained from the previous call to "
                                  "`LIGHT`.",)


@validate_arguments
def light(
    location: str,
    recurrent_states: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> LightResponse:
    """
    Parameters
    ----------
    location:
        Location name in IAI format.
        If `recurrent_state` is provided which is obtained from previous calls to light, next state is returned.
        Otherwise, a random state is generated which can be reproduced by setting the `random_seed`.

    recurrent_states:
        Recurrent states for traffic lights, obtained from the previous call to
        :func:`light`.

    random_seed:
        Controls the stochastic aspects of agent behavior for reproducibility.

    See Also
    --------
    :func:`initialize`
    :func:`location_info`
    :func:`drive`
    """
    if should_use_mock_api():
        response = LightResponse(traffic_lights_states={123: TrafficLightState("green"),
                                                        124: TrafficLightState("red"),
                                                        126: TrafficLightState("yellow")},
                                 recurrent_states="ABC-DEF-GHI@12"
                                 )
        return response
    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "recurrent_states": recurrent_states, "random_seed": random_seed}
    while True:
        try:
            response = iai.session.request(model="light", params=params)
            return LightResponse(**response)
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))
