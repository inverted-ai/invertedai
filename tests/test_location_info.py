import sys
import pytest

sys.path.insert(0, "../../")
import invertedai as iai
from invertedai.api.initialize import initialize
from invertedai.api.drive import drive, DriveResponse
from invertedai.api.location import location_info
from invertedai.api.light import light
from invertedai.common import Point
from invertedai.error import InvalidRequestError


def test_location_info():
    location = "carla:Town03"
    _ = iai.location_info(location=location)


def test_mock_location_info():
    iai.api.config.mock_api = True
    location = "carla:Town03"
    _ = iai.location_info(location=location, rendering_center=None, rendering_fov=800)
    iai.api.config.mock_api = False
