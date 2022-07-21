import requests
import os
import torch
from invertedai_drive.models import DriveResponse, AgentStatesWithSample
from dotenv import load_dotenv

from enum import Enum


class MapLocation(Enum):
    two_lane_circle: str = '2lane_circle'
    two_lane_intersection: str = '2lane_intersection'
    four_lane_t_intersection: str = '4lane_t_intersection'
    four_lane_intersection: str = '4lane_intersection'
    six_lane_intersection: str = '6lane_intersection'
    cloverleaf: str = 'cloverleaf'
    figure_eight: str = 'figure_eight'
    i80: str = 'i80'
    loop: str = 'loop'
    minicity_toronto: str = 'minicity_toronto'
    od_4lane: str = 'od_4lane'
    od_merge: str = 'od_merge'
    od_newmarket: str = 'od_newmarket'
    peachtree: str = 'peachtree'
    straight: str = 'straight'
    us101: str = 'us101'
    Town01_3way: str = 'Town01_3way'
    Town01_Straight: str = 'Town01_Straight'
    Town02_3way: str = 'Town02_3way'
    Town02_Straight: str = 'Town02_Straight'
    Town03_3way_Protected: str = 'Town03_3way_Protected'
    Town03_3way_Unprotected: str = 'Town03_3way_Unprotected'
    Town03_4way: str = 'Town03_4way'
    Town03_5way: str = 'Town03_5way'
    Town03_GasStation: str = 'Town03_GasStation'
    Town03_Roundabout: str = 'Town03_Roundabout'
    Town04_3way_Large: str = 'Town04_3way_Large'
    Town04_3way_Small: str = 'Town04_3way_Small'
    Town04_4way_Stop: str = 'Town04_4way_Stop'
    Town04_Merging: str = 'Town04_Merging'
    Town04_Parking: str = 'Town04_Parking'
    Town06_4way_large: str = 'Town06_4way_large'
    Town06_Merge_Double: str = 'Town06_Merge_Double'
    Town06_Merge_Single: str = 'Town06_Merge_Single'
    Town07_3way: str = 'Town07_3way'
    Town07_4way: str = 'Town07_4way'
    Town10HD_3way_Protected: str = 'Town10HD_3way_Protected'
    Town10HD_3way_Stop: str = 'Town10HD_3way_Stop'
    Town10HD_4way: str = 'Town10HD_4way'


class Client:
    def __init__(self):
        load_dotenv()
        self.dev = "DEV" in os.environ
        if not self.dev:
            self._endpoint = "https://gzjse7c92i.execute-api.us-west-2.amazonaws.com/dev"
        else:
            self._endpoint = "http://localhost:8000"

    def run(self, api_key: str, model_inputs: dict) -> DriveResponse:
        def _extract_model_outputs(dict_model_output):
            states = dict_model_output["states"]
            states = AgentStatesWithSample(x=states['x'], y=states['y'], psi=states['psi'], speed=states['speed'])
            recurrent_states = dict_model_output["recurrent_states"]
            bird_views = dict_model_output["bird_view"]
            model_outputs = DriveResponse(states, recurrent_states, bird_views)
            return model_outputs
        response = requests.post(f'{self._endpoint}/drive',
                                 json=model_inputs,
                                 headers={'Content-Type': 'application/json',
                                          'Accept-Encoding': 'gzip, deflate, br',
                                          'Connection': 'keep-alive',
                                          'x-api-key': api_key,
                                          'api-key': api_key})
        return _extract_model_outputs(response.json())

    def initialize(self, location, agents_counts=10, num_samples=1, min_speed=1, max_speed=3):
        response = requests.get(f'{self._endpoint}/initialize',
                                params={'location': location, 'num_agents_to_spawn': agents_counts,
                                        'num_samples': num_samples,
                                        'spawn_min_speed': min_speed, 'spawn_max_speed': max_speed},
                                headers={'Content-Type': 'application/json',
                                         'Accept-Encoding': 'gzip, deflate, br',
                                         'Connection': 'keep-alive'})

        return response.json()
