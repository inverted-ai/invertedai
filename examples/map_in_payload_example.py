import os                                                                                                                                                                                   
import invertedai as iai
from invertedai.utils import get_default_agent_properties
from invertedai.common import (
    AgentState,
    AgentType,
    Point,
    StaticMapActor
)
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import time as time

import numpy as np
import imageio
import argparse


import torch
from torchdrivesim.map import traffic_controls_from_map_config
from torchdrivesim.mesh import BirdviewMesh, BirdviewRGBMeshGenerator
from torchdrivesim.rendering import  renderer_from_config, RendererConfig
from torchdrivesim.map import load_map_config, MapConfig
from torchdrivesim.utils import Resolution


@dataclass
class MapVisualizationConfig:
    res: Resolution = Resolution(1024, 1024)
    fov: float = 250
    center: Optional[Tuple[float, float]] = None
    map_origin: Tuple[float, float] = (0, 0)
    orientation: float = 0
    save_path: str = './map_visualization.png'
    is_test: bool = False


def save_gif(imgs, filename, batch_index=0, fps=10):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimwrite(
        filename, [img[batch_index].cpu().numpy().astype(np.uint8).transpose(1, 2, 0) for img in imgs],
        format="GIF", fps=fps, loop=0
    )

def visualize_trajectory(
    agent_state: List[List[AgentState]],
    road_mesh_per_timestep: List[BirdviewMesh],
    map_vis_cfg: MapVisualizationConfig,
    map_cfg_list: List[MapConfig],
    device = 'cuda'
):
    T = len(agent_state)
    if T == 0:
        raise ValueError("agent_state is empty")

    A = len(agent_state[0])
    print(f"Number of timesteps: {T}, Number of agents: {A}")

    states = torch.zeros((1, 1, A, T, 4), device=device)

    for t in range(T):
        if len(agent_state[t]) != A:
            raise ValueError("Number of agents must be consistent across timesteps")

        for a in range(A):
            s = agent_state[t][a]

            states[0, 0, a, t, 0] = s.center.x
            states[0, 0, a, t, 1] = s.center.y
            states[0, 0, a, t, 2] = s.orientation
            states[0, 0, a, t, 3] = s.speed

    present_mask = torch.ones((1, 1, A, T), dtype=torch.bool, device=device)

    batch_size, sample_size, num_agents, timesteps, _ = states.shape

    renderer_cfg = RendererConfig(left_handed_coordinates=map_cfg_list[0].left_handed_coordinates)
    renderer = renderer_from_config(renderer_cfg)

    birdview_mesh_generator = BirdviewRGBMeshGenerator(
        background_mesh=road_mesh_per_timestep[0],
        color_map=renderer.color_map,
        rendering_levels=renderer.rendering_levels,
        traffic_controls = traffic_controls_from_map_config(map_cfg_list[0])
    )

    lenwid = torch.full(
        (1, A, 2),
        2.0,
        device=device,
    )
    lenwid[:, :, 0] = 3.0

    agent_type = torch.zeros(
        (1, A),
        dtype=torch.long,
        device=device,
    )

    agent_type_names = ["vehicle"]
    birdview_mesh_generator.initialize_actors_mesh(
        lenwid,
        agent_type,
        agent_type_names,
    )

    birdview_mesh_generator.to(device)
    
    xy = states[..., :2]  # (1,1,A,T,2)
    camera_xy = (xy.max(dim=2).values + xy.min(dim=2).values) / 2  # (1,1,T,2)
    camera_psi = torch.full_like(camera_xy[..., :1], torch.pi / 2)

    camera_sc = torch.cat(
        [torch.sin(camera_psi), torch.cos(camera_psi)], dim=-1
    )

    n_cameras = 1
    imgs = []

    for t in range(timesteps):
        current_state = states[..., t, :] # (1,1,A,4)
        current_mask = present_mask[..., t]
        birdview_mesh_generator.initialize_background_mesh(
            BirdviewMesh.collate(road_mesh_per_timestep[t])
        )

        rbg_mesh = birdview_mesh_generator.generate(
            n_cameras,
            agent_state=current_state[:, 0:1].expand(-1, n_cameras, -1, -1),
            present_mask=current_mask[:, 0:1].expand(-1, n_cameras, -1),
        )

        bv = renderer.render_frame(
            rbg_mesh,
            camera_xy[..., t, :],
            camera_sc[..., t, :],
            res=map_vis_cfg.res,
            fov=map_vis_cfg.fov,
        )

        imgs.append(bv)

    save_gif(imgs, map_vis_cfg.save_path, batch_index=0, fps=10)
    return imgs

def parse_folder_to_files(
    timestamp_dir: str, 
    sim_time: int
):
    with open(os.path.join(timestamp_dir, f"{sim_time}", "metadata.json"), 'r') as file:
        metadata = json.load(file)
        map_center = metadata["center"]

    files = {
        "lanelet_osm_file": metadata.get("lanelet_path", None),
        "stoplines_file": metadata.get("stoplines_path", None),
        "traffic_light_controller_file": metadata.get("traffic_light_controller_path", None),
    }
    files = { fname : open(os.path.join(timestamp_dir, f"{sim_time}", fpath), "rb") 
        for fname, fpath in files.items() if fpath is not None
    }
    
    map_cfg_metadata_path = os.path.join(timestamp_dir, f'{sim_time}/metadata.json')
    map_cfg = load_map_config(map_cfg_metadata_path)

    return files, map_center, map_cfg


def main(args):

    location = ""
    local_map_dir = args.data_dir
    simulation_length = args.sim_length 
    seed = 1
    num_agents = args.num_agents

    iai.add_apikey(args.api_key)  # specify your key here or through the IAI_API_KEY variable

    print("Begin initialization.")
    road_mesh_per_frame = []
    agent_states_over_time = []
    map_cfg_list = []

    files, map_center, map_cfg = parse_folder_to_files(local_map_dir, 0)
    map_vis_cfg = MapVisualizationConfig(
        center = map_center,
        save_path = f'./test_gifs/{int(time.time())}_map_visualization.png',
        is_test = True
    )
    road_mesh_per_frame.append(map_cfg.road_mesh.to("cuda"))
    map_cfg_list.append(map_cfg)

    response = iai.initialize(
        location=location, 
        agent_properties=get_default_agent_properties({AgentType.car:num_agents}),  # number of NPCs to spawn
        random_seed=seed,
        get_birdview=True,
        map_center=map_center,
        files=files
    )
    agent_properties = response.agent_properties
    agent_states_over_time.append(response.agent_states)

    print("Begin stepping through simulation.")
    for i in range(simulation_length): 
        print(i)
        files, map_center, map_cfg = parse_folder_to_files(local_map_dir, i)
        road_mesh_per_frame.append(map_cfg.road_mesh.to("cuda"))
        map_cfg_list.append(map_cfg)

        response = iai.drive(
            location=location,
            agent_properties=agent_properties,
            agent_states=response.agent_states,
            recurrent_states=response.recurrent_states,
            light_recurrent_states=response.light_recurrent_states,
            random_seed=seed,
            map_center=map_center,
            files=files
        )
        agent_states_over_time.append(response.agent_states)

    print("Simulation finished, saving visualization.")

    visualize_trajectory(
        agent_state = agent_states_over_time,
        road_mesh_per_timestep = road_mesh_per_frame,
        map_vis_cfg = map_vis_cfg,
        map_cfg_list = map_cfg_list,
    )
    print("Done")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-N',
        '--num_agents',
        metavar='D',
        default=1,
        type=int,
        help='Number of vehicles to spawn per 100x100m grid (default: 10)'
    )
    argparser.add_argument(
        '--sim_length',
        type=int,
        help="Length of the simulation in timesteps (default: 100)",
        default=10
    )
    argparser.add_argument(
        '--api_key',
        type=str,
        help=f"API Key to make calls to the IAI API",
        default='None'
    )
    argparser.add_argument(
        '--data_dir',
        type=str,
        help=f"Directory of formatted data.",
        default=''
    )

    args = argparser.parse_args()

    main(args)