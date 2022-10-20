import time
from dataclasses import dataclass

from typing import Optional, List

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import TryAgain
from invertedai.api.mock import get_mock_birdview

from invertedai.common import Point, Image, LocationMap, StaticMapActor


@dataclass
class LocationResponse:
    """
    Response returned from an API call to :func:`iai.location_info`.
    """
    version: str  #: Map version. Matches the version in the input location string, if one was specified.
    max_agent_number: int  #: Maximum number of agents recommended in the location. Use more at your own risk.
    bounding_polygon: Optional[
        List[Point]
    ]  #: Convex polygon denoting the boundary of the supported area within the location.
    birdview_image: Image  #: Visualization of the location.
    osm_map: Optional[LocationMap]  #: Underlying map annotation, returned if `include_map_source` was set.
    static_actors: List[StaticMapActor]  #: Lists traffic lights with their IDs and locations.


def location_info(
    location: str,
    include_map_source: bool = False,
) -> LocationResponse:
    """
    Provides static information about a given location.

    Parameters
    ----------
    location:
        Location string in IAI format.

    include_map_source:
        Whether to return full map specification in Lanelet2 OSM format.
        This significantly increases the response size, consuming more network resources.

    Examples
    --------
    >>> import invertedai as iai
    >>> response = iai.location_info(location="")
    >>> if response.osm_map is not None:
    >>>     file_path = f"{file_name}.osm"
    >>>     with open(file_path, "w") as f:
    >>>         f.write(response.osm_map[0])
    >>> if response.birdview_image is not None:
    >>>     file_path = f"{file_name}.jpg"
    >>>     rendered_map = np.array(response.birdview_image, dtype=np.uint8)
    >>>     image = cv2.imdecode(rendered_map, cv2.IMREAD_COLOR)
    >>>     cv2.imwrite(file_path, image)
    """

    if should_use_mock_api():
        response = LocationResponse(
            version="v0.0.0",
            birdview_image=get_mock_birdview(),
            osm_map=None,
            static_actors=[],
            bounding_polygon=[],
            max_agent_number=10,
        )
        return response

    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "include_map_source": include_map_source}
    while True:
        try:
            response = iai.session.request(model="location_info", params=params)
            if response['bounding_polygon'] is not None:
                response['bounding_polygon'] = [Point(x=x, y=y) for (x, y) in response['bounding_polygon']]
            if response["static_actors"] is not None:
                response["static_actors"] = [
                    StaticMapActor.fromdict(actor) for actor in response["static_actors"]
                ]
            if response["osm_map"] is not None:
                response["osm_map"] = LocationMap(encoded_map=response["osm_map"], origin=response["map_origin"])
            del response["map_origin"]
            response['birdview_image'] = Image(response['birdview_image'])
            return LocationResponse(**response)
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))
