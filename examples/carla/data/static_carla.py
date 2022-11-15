from carla import Location, Rotation, Transform
from collections import namedtuple
from typing import Tuple
import carla
import numpy as np

MAP_CENTERS = {
    "carla:Town01:3way": [93.70275115966797, 200.7403106689453],
    "carla:Town01:Straight": [254.2249755859375, 300.88372802734375],
    "carla:Town02:3way": [41.620059967041016, 191.54580688476562],
    "carla:Town02:Straight": [110.15972900390625, 258.85760498046875],
    "carla:Town03:3way_Protected": [187.63462829589844, -2.8370285034179688],
    "carla:Town03:3way_Unprotected": [3.7141036987304688, -154.9966278076172],
    "carla:Town03:4way": [-65.73756408691406, -122.9695053100586],
    "carla:Town03:5way": [-64.09320068359375, 2.4671669006347656],
    "carla:Town03:Gas_Station": [-20.871902465820312, 141.3246307373047],
    "carla:Town03:Roundabout": [0.0, 0.0],
    "carla:Town04:3way_Large": [-378.30316162109375, -12.988147735595703],
    "carla:Town04:3way_Small": [127.83306884765625, -167.14236450195312],
    "carla:Town04:4way_Stop": [197.35690307617188, -192.64134216308594],
    "carla:Town04:Merging": [92.00399780273438, 5.6601104736328125],
    "carla:Town04:Parking": [288.290771484375, -212.30825805664062],
    "carla:Town06:4way_Large": [2.7990493774414062, -17.995826721191406],
    "carla:Town06:Merge_Double": [-142.56692504882812, 58.08183670043945],
    "carla:Town06:Merge_Single": [515.815185546875, 29.682167053222656],
    "carla:Town07:3way": [-95.72276306152344, 25.259380340576172],
    "carla:Town07:4way": [-101.99176025390625, 40.31178283691406],
    "carla:Town10HD:3way_Protected": [-32.242706298828125, 87.99059295654297],
    "carla:Town10HD:3way_Stop": [28.333023071289062, 62.63806915283203],
    "carla:Town10HD:4way": [-33.782386779785156, 26.179428100585938],
}

DEMO_LOCATIONS = {
    "carla:Town01:3way": dict(
        proximity_threshold=40,
        spawning_locations=[
            Transform(
                Location(x=184.2, y=194.3, z=0.5),
                Rotation(pitch=0.0, yaw=174.4, roll=0.0),
            )
        ],
    ),
    "carla:Town02:3way": dict(
        proximity_threshold=25,
        spawning_locations=[
            Transform(
                Location(x=-7.2, y=154.1, z=0.5), Rotation(pitch=0.0, yaw=92, roll=0.0)
            )
        ],
    ),
    "carla:Town03:Roundabout": dict(
        proximity_threshold=60,
        spawning_locations=[
            Transform(
                Location(x=-54.5, y=-0.1, z=0.5),
                Rotation(pitch=0.0, yaw=1.76, roll=0.0),
            ),
            Transform(
                Location(x=-1.6, y=-87.4, z=0.5),
                Rotation(pitch=0.0, yaw=91.0, roll=0.0),
            ),
            Transform(
                Location(x=1.5, y=78.6, z=0.5), Rotation(pitch=0.0, yaw=-83.5, roll=0.0)
            ),
            Transform(
                Location(x=68.1, y=-4.1, z=0.5),
                Rotation(pitch=0.0, yaw=178.7, roll=0.0),
            ),
        ],
    ),
    "carla:Town03:4way": dict(
        proximity_threshold=45,
        spawning_locations=[
            Transform(
                Location(x=-145.7, y=-18.7, z=0.5),
                Rotation(pitch=0.0, yaw=-89.2, roll=0.0),
            ),
            Transform(
                Location(x=0.56, y=-183.0, z=0.5),
                Rotation(pitch=0.0, yaw=93.2, roll=0.0),
            ),
            Transform(
                Location(x=-14.2, y=-141.8, z=0.5),
                Rotation(pitch=0.0, yaw=-175.6, roll=-0.0),
            ),
            Transform(
                Location(x=-77.8, y=-10.2, z=0.5),
                Rotation(pitch=-0.5, yaw=-88.6, roll=0.0),
            ),
        ],
    ),
    "carla:Town03:GasStation": dict(
        proximity_threshold=40,
        spawning_locations=[
            Transform(
                Location(x=-10.7, y=46.2, z=0.5),
                Rotation(pitch=0.0, yaw=90.4, roll=0.0),
            )
        ],
    ),
    "carla:Town04:Merging": dict(
        proximity_threshold=80,
        spawning_locations=[
            Transform(
                Location(x=-49.8, y=37.2, z=10.2),
                Rotation(pitch=0.1, yaw=1.5, roll=-0.1),
            ),
            Transform(
                Location(x=44.7, y=-99.3, z=0.5),
                Rotation(pitch=0.0, yaw=-22.0, roll=0.0),
            ),
        ],
    ),
    "carla:Town04:4way_Stop": dict(
        proximity_threshold=80,
        spawning_locations=[
            Transform(
                Location(x=150.8, y=-169.6, z=0.5),
                Rotation(pitch=0.0, yaw=1.0, roll=0.0),
            ),
            Transform(
                Location(x=223.3, y=-124.6, z=0.5),
                Rotation(pitch=0.0, yaw=-151.2, roll=0.0),
            ),
        ],
    ),
    "carla:Town10HD:4way": dict(
        proximity_threshold=70,
        spawning_locations=[
            Transform(
                Location(x=-103.6, y=47.1, z=0.5),
                Rotation(pitch=0.0, yaw=-85.8, roll=0.0),
            ),
            Transform(
                Location(x=-41.8, y=110.8, z=0.5),
                Rotation(pitch=0.0, yaw=-80.6, roll=0.0),
            ),
            Transform(
                Location(x=-41.8, y=110.8, z=0.5),
                Rotation(pitch=0.0, yaw=-80.6, roll=0.0),
            ),
        ],
    ),
}


NPC_BPS: Tuple[str] = (
    "vehicle.audi.a2",
    "vehicle.audi.etron",
    "vehicle.audi.tt",
    "vehicle.bmw.grandtourer",
    "vehicle.citroen.c3",
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020",
    "vehicle.ford.mustang",
    "vehicle.ford.crown",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.nissan.micra",
    "vehicle.nissan.patrol_2021",
    "vehicle.seat.leon",
    "vehicle.toyota.prius",
    "vehicle.volkswagen.t2_2021",
)
EGO_FLAG_COLOR = carla.Color(255, 0, 0, 0)
NPC_FLAG_COLOR = carla.Color(0, 0, 255, 0)
RS = np.zeros([2, 64]).tolist()
cord = namedtuple("cord", ["x", "y"])
