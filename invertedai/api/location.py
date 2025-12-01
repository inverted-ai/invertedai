import time
from pydantic import BaseModel, validate_call
from typing import Optional, List, Tuple
import tempfile

import invertedai as iai
from invertedai.api.config import TIMEOUT, should_use_mock_api
from invertedai.error import TryAgain
from invertedai.api.mock import get_mock_birdview

from invertedai.common import Point, Origin, Image, LocationMap, StaticMapActor


class LocationResponse(BaseModel):
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
    map_center: Point  #: The x,y coordinate of the center of the map.
    map_fov: float  #: The field of view in meters for the birdview image.
    static_actors: List[StaticMapActor]  #: Lists traffic lights with their IDs and locations.

    
    def get_lanelet_map(self):
        assert self.osm_map is not None and self.osm_map.encoded_map, "osm_map is empty, please ensure this response was obtained with `include_map_source` set to true."
        import lanelet2

        with tempfile.NamedTemporaryFile(suffix=".osm", delete=True) as tmp:
            self.osm_map.save_osm_file(tmp.name)
            tmp.flush()
            projector = lanelet2.projection.UtmProjector(
                lanelet2.io.Origin(self.osm_map.origin.x, self.osm_map.origin.y)
            )
            return lanelet2.io.load(tmp.name, projector)

@validate_call
def location_info(
    location: str,
    include_map_source: bool = False,
    rendering_fov: Optional[int] = None,
    rendering_center: Optional[Tuple[float, float]] = None,
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

    rendering_fov:
        Optional fov for both x and y axis for the rendered birdview in meters.

    rendering_center:
        Optional center x,y coordinates for the rendered birdview.
    See Also
    --------
    :func:`drive`
    :func:`initialize`
    :func:`light`
    :func:`blame`
    """

    if should_use_mock_api():
        response = LocationResponse(
            version="v0.0.0",
            birdview_image=get_mock_birdview(),
            osm_map=None,
            static_actors=[],
            bounding_polygon=[],
            max_agent_number=10,
            map_center=Point(x=0, y=0),
            map_fov=100,

        )
        return response

    start = time.time()
    timeout = TIMEOUT

    params = {"location": location, "include_map_source": include_map_source, "rendering_fov": rendering_fov,
              "rendering_center": ",".join([str(rendering_center[0]), str(rendering_center[1])]) if rendering_center else rendering_center}
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
                response["osm_map"] = LocationMap(
                    encoded_map=response["osm_map"],
                    origin=Origin.fromlist(
                        response["map_origin"]))
            del response["map_origin"]
            response["map_center"] = Point.fromlist(response["map_center"])
            response['birdview_image'] = Image.fromval(response['birdview_image'])
            return LocationResponse(**response)
        except TryAgain as e:
            if timeout is not None and time.time() > start + timeout:
                raise e
            iai.logger.info(iai.logger.logfmt("Waiting for model to warm up", error=e))
