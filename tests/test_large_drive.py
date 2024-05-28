import sys
import pytest

sys.path.insert(0, "../../")
from invertedai.large.initialize import large_initialize
from invertedai.large.drive import large_drive
from invertedai.large.common import Region

from invertedai.api.drive import DriveResponse
from invertedai.api.location import location_info
from invertedai.common import Point, AgentAttributes, AgentState, RecurrentState
from invertedai.error import InvalidRequestError


# negative_tests = [

# ]

# def run_direct_drive(location, agent_states, agent_attributes, recurrent_states, get_infractions):
#     drive_response = drive(
#         agent_attributes=agent_attributes,
#         agent_states=agent_states,
#         recurrent_states=recurrent_states,
#         traffic_lights_states=None,
#         get_birdview=False,
#         location=location,
#         get_infractions=get_infractions
#     )
#     assert isinstance(drive_response,DriveResponse) and drive_response.agent_states is not None and drive_response.recurrent_states is not None


# @pytest.mark.parametrize("location, regions, return_exact_agents, simulation_length", negative_tests)
# def test_negative(location, regions, return_exact_agents, simulation_length):
#     with pytest.raises(InvalidRequestError):
#         run_direct_drive(location, regions, return_exact_agents, simulation_length)

location_precalculated = "carla:Town03"
initial_response = large_initialize(
    location="carla:Town03",
    regions=[
        Region.create_square_region(center=Point.fromlist([79.8,-4.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
        Region.create_square_region(center=Point.fromlist([-82.7,-122.1]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
        Region.create_square_region(center=Point.fromlist([-82.7,136.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
        Region.create_square_region(center=Point.fromlist([-2.2,132.3]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)]),
        Region.create_square_region(center=Point.fromlist([-80.5,-3.2]),agent_attributes=[AgentAttributes.fromlist(["car"]) for _ in range(25)])
    ],
    return_exact_agents=True
)

single_agent_state = AgentState.fromlist([37.08,-31.78,1.52,0.15])
single_agent_attributes = AgentAttributes.fromlist([5,2,1.8,'car'])
single_recurrent_state = RecurrentState.fromval([
    0.03927750885486603, -0.02364601008594036, -0.1111457347869873, -0.05127168819308281, -0.03704476356506348, 0.04733209311962128, -0.005560159217566252, 0.0646270141005516, 
    0.014635450206696987, 0.5463542342185974, 0.09056184440851212, 0.0963616818189621, -0.04014160484075546, 0.08820615708827972, -0.12021058052778244, 0.043172966688871384, 
    -0.0981266051530838, 0.1660698801279068, 0.1142461821436882, -0.5930933952331543, -0.06753531098365784, -0.03540925309062004, 0.37989747524261475, -0.006204646546393633, 
    -0.07429599016904831, -0.386107474565506, -0.17112380266189575, 0.11227439343929291, 0.06273427605628967, -0.04169732704758644, 0.0640067532658577, -0.07385984063148499, 
    -0.02857285924255848, 0.01853279210627079, 0.10566180944442749, 0.13444045186042786, -0.10455009341239929, -0.2542324662208557, 0.03798912465572357, 0.3476734757423401, 
    0.023403123021125793, -0.06454779207706451, -0.17175643146038055, -0.016702326014637947, -0.1828240305185318, -0.22749686241149902, -0.2600526213645935, -0.1013202890753746, 
    0.14666825532913208, -0.029236560687422752, 0.331624299287796, -0.0321040116250515, 0.3058949410915375, 0.023920278996229172, -0.0908496081829071, -0.06013471260666847, 
    -0.6968921422958374, -0.1263137012720108, -0.048175569623708725, 0.15210536122322083, 0.2735196352005005, -0.08595801144838333, 0.0930427610874176, -0.09820298850536346, 
    0.058781299740076065, -0.07763578742742538, 0.09238878637552261, -0.17754335701465607, -0.18359598517417908, -0.2463427037000656, 0.13524113595485687, 0.18852432072162628, 
    -0.15690360963344574, 0.17655128240585327, 0.018684938549995422, 0.2505805194377899, -0.026350922882556915, -0.3250160813331604, 0.05824386700987816, 0.10591545701026917, 
    -0.022652864456176758, 0.1572694033384323, -0.2661135494709015, 0.40190088748931885, 0.05969567224383354, 0.14561055600643158, 0.08736822754144669, 0.20525376498699188, 
    -0.546593964099884, 0.054666101932525635, -0.02235374227166176, -0.2613176703453064, 0.1728011965751648, -0.0925864726305008, -0.036109961569309235, 0.02835044637322426, 
    0.11358940601348877, 0.14728863537311554, -0.16177217662334442, 0.05613938719034195, 0.21455825865268707, 0.12710559368133545, 0.04110559821128845, -0.012531726621091366, 
    0.10438267886638641, -0.010487884283065796, -0.08379699289798737, -0.09544740617275238, 0.12141408771276474, -0.5470792651176453, 0.4102642238140106, 0.004504964221268892, 
    0.012034604325890541, 0.2667303681373596, 0.09879391640424728, 0.10467187315225601, -0.41841819882392883, 0.13036806881427765, -0.14156298339366913, 0.47851330041885376, 
    -0.6324084997177124, -0.38299620151519775, -0.2614000737667084, 0.3241986930370331, -0.015558694489300251, -0.6518490314483643, 0.16429710388183594, -0.07537366449832916, 
    37.08000183105469, -31.780000686645508, 1.5199999809265137, 0.15000000596046448, 0.0033642947673797607, -0.01787346601486206, 0.0, 0.0, 0.02416781522333622, 3.7035329772568126e-16, 
    0.0, 0.0, 0.0, 0.024167809635400772, 0.0, 0.0, 0.0, 1.8407117822244516e-18, 0.00010101009684149176, -9.248569864543793e-44, 0.0, 0.0, -9.248569864543793e-44, 0.00010101009684149176
])
positive_tests = [
    ( #Drive fewer agents than typically allowed
        "canada:ubc_roundabout",
        [single_agent_state],
        [single_agent_attributes],
        [single_recurrent_state],
        None,
        10
    ),
    ( #Drive more agents than typically allowed
        location_precalculated,
        initial_response.agent_states,
        initial_response.agent_attributes,
        initial_response.recurrent_states,
        initial_response.light_recurrent_states,
        10
    ),
]

def run_initialize_drive_flow(location, regions, return_exact_agents, simulation_length):
    location_info_response = location_info(location=location)
    scene_has_lights = any(actor.agent_type == "traffic-light" for actor in location_info_response.static_actors)
    
    response = large_initialize(
        location=location,
        regions=regions,
        traffic_light_state_history=None,
        return_exact_agents=return_exact_agents
    )
    agent_attributes = response.agent_attributes
    for t in range(simulation_length):
        response = large_drive(
            location=location,
            agent_states=response.agent_states,
            agent_attributes=agent_attributes,
            recurrent_states=response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states if scene_has_lights else None,
        )
        assert isinstance(response,DriveResponse) and response.agent_states is not None and response.recurrent_states is not None
        if scene_has_lights:
            assert response.traffic_lights_states is not None
            assert response.light_recurrent_states is not None

def run_drive_direct(location, agent_states, agent_attributes, recurrent_states, light_recurrent_states, simulation_length):
    location_info_response = location_info(location=location)
    scene_has_lights = any(actor.agent_type == "traffic-light" for actor in location_info_response.static_actors)

    for t in range(simulation_length):
        response = large_drive(
            location=location,
            agent_states=agent_states,
            agent_attributes=agent_attributes,
            recurrent_states=recurrent_states,
            light_recurrent_states=light_recurrent_states if scene_has_lights else None,
        )
        assert isinstance(response,DriveResponse) and response.agent_states is not None and response.recurrent_states is not None
        if scene_has_lights:
            assert response.traffic_lights_states is not None
            assert response.light_recurrent_states is not None

# @pytest.mark.parametrize("location, agent_states, agent_attributes, recurrent_states, light_recurrent_states, simulation_length", positive_tests)
# def test_postivie(location, agent_states, agent_attributes, recurrent_states, light_recurrent_states, simulation_length):
#     run_drive_direct(location, agent_states, agent_attributes, recurrent_states, light_recurrent_states, simulation_length)



