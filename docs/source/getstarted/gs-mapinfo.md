# Locations and Maps

Inverted AI provides an assortment of diverse road configurations and geometries, including real-world locations and maps from simulators (CARLA, Huawei SMARTS, ...).

To get information about a scene use **{ref}`LOCATION_INFO`**.
```python
iai.location_info("iai:ubc_roundabout")
```
The scene information include the map in [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) format, map in JPEG format, maximum number of allowed (driveable) vehicles, latitude longitude coordinates (for real-world locations), id and list of traffic light and signs (if any exist in the map), etc.


---
{ref}`LOCATION_INFO`: more information about the python SDK.\
{ref}`REST API`: more information about the REST API and other programming languages.



