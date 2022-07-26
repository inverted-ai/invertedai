from invertedai_drive.drive import run, initialize
from invertedai_drive.utils import MapLocation

def test_mock():
    result = run(api_key='', location=MapLocation.Town03_Roundabout,
                 x=[
                         [
                                 [
                                         0
                                 ]
                         ]
                 ],
                 y=[
                         [
                                 [
                                         0
                                 ]
                         ]
                 ],
                 psi=[
                         [
                                 [
                                         0
                                 ]
                         ]
                 ],
                 speed=[
                         [
                                 [
                                         0
                                 ]
                         ]
                 ],
                 length=[
                         [
                                 0
                         ]
                 ],
                 width=[
                         [
                                 0
                         ]
                 ],
                 lr=[
                         [
                                 0
                         ]
                 ],
                 recurrent_states=None,
                 present_masks=[
                         [
                                 [
                                         True
                                 ]
                         ]
                 ],
                 batch_size=1,
                 agent_counts=1,
                 obs_length=1,
                 step_times=1,
                 num_predictions=1)
    print(result)


def test_initialize():
    result = initialize(MapLocation.Town03_Roundabout, 1, 1, 1, 1)
    print(result)
