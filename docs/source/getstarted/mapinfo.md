# Locations and Maps

Inverted AI provides an assortment of diverse road configurations and geometries, including real-world locations and maps from simulators (CARLA, Huawei SMARTS, ...).\

## AVAILABLE-LOCATIONS
To search the catalog for available maps to your account (API key) use **iai.available_locations** method by providing keywords
```python
iai.available_locations("roundabout", "carla")
```

## GET-MAP
To get information about a scene use **iai.get_map**.
```python
iai.get_map**("CARLA:Town03:Roundabout")
```
The scene information include the map in [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) format, map in JPEG format, maximum number of allowed (driveable) vehicles, latitude longitude coordinates (for real-world locations), id and list of traffic light and signs (if any exist in the map), etc.



