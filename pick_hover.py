from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@dataclass
class LegoMinifigureDomainRandomizationConfig:
    initial_qpos_noise_scale: float = 0.02
    robot_color: Optional[Union[str, Sequence[float]]] = None
    minifigure_color: Optional[Union[str, Sequence[float]]] = "random"
    randomize_lighting: bool = True
    max_camera_offset: Sequence[float] = (0.025, 0.025, 0.025)
    camera_target_noise: float = 1e-3
    camera_view_rot_noise: float = 5e-3
    camera_fov_noise: float = np.deg2rad(2)

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@register_env("LegoMinifigurePickPlace-v2", max_episode_steps=100)
class LegoMinifigurePickPlaceEnv(BaseDigitalTwinEnv):
    """
    **Task Description:**
    A placement task where the objective is to place a LEGO minifigure inside a fighter ship using the SO100 arm.
    The robot must grasp the minifigure and then place it inside the fighter.

    **Randomizations:**
    - the minifigure's xy position is randomized on top of a table
    - the minifigure's z-axis rotation is randomized
    - the fighter's xy position is randomized on top of a table
    - the fighter's z-axis rotation is randomized

    **Success Conditions:**
    - the minifigure is lifted and placed inside the fighter
    """

    SUPPORTED_ROBOTS = ["so100"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb+segmentation"]
    agent: SO100

    def __init__(
        self,
        *args,
        robot_uids="so100",
        control_mode="pd_joint_target_delta_pos",
        greenscreen_overlay_path=None,
        domain_randomization_config: Union[
            LegoMinifigureDomainRandomizationConfig, dict
        ] = LegoMinifigureDomainRandomizationConfig(),
        domain_randomization=True,
        base_camera_settings=dict(
            fov=52 * np.pi / 180,
            pos=[0.5, 0.3, 0.35],
            target=[0.3, 0.0, 0.1],
        ),
        spawn_box_pos=[0.3, 0.05],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        self.domain_randomization_config = LegoMinifigureDomainRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(
                merged_domain_randomization_config, domain_randomization_config
            )
            self.domain_randomization_config = dacite.from_dict(
                data_class=LegoMinifigureDomainRandomizationConfig,
                data=domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        self.base_camera_settings = base_camera_settings

        if greenscreen_overlay_path is None:
            logger.warning(
                "No greenscreen overlay path provided, no greenscreen will be used"
            )
            self.rgb_overlay_mode = "none"
        else:
            self.rgb_overlay_paths = dict(base_camera=greenscreen_overlay_path)

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size
        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode, **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        if self.domain_randomization:
            camera_fov_noise = self.domain_randomization_config.camera_fov_noise * (
                2 * self._batched_episode_rng.rand() - 1
            )
        else:
            camera_fov_noise = 0
        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=camera_fov_noise + self.base_camera_settings["fov"],
                near=0.01,
                far=100,
                mount=self.camera_mount,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_lighting(self, options: dict):
        if self.domain_randomization:
            if self.domain_randomization_config.randomize_lighting:
                ambient_colors = self._batched_episode_rng.uniform(0.2, 0.5, size=(3,))
                for i, scene in enumerate(self.scene.sub_scenes):
                    scene.render_system.ambient_light = ambient_colors[i]
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Build LEGO minifigures (using the working approach from create_actors.py)
        minifigures = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            
            # Use the same approach as the working create_actors.py
            builder.add_convex_collision_from_file(
                filename="./assets/Boba_meters.obj"
            )
            builder.add_visual_from_file(
                filename="./assets/Boba_meters_v2.glb"
            )
            
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi/2, 0, 0))  # 90deg X rotation to stand upright
            builder.set_scene_idxs([i])
            minifig = builder.build(name=f"lego_minifigure-{i}")
            minifigures.append(minifig)
            self.remove_from_state_dict_registry(minifig)

        # Merge minifigures into a single actor
        self.minifigure = Actor.merge(minifigures, name="lego_minifigure")
        self.add_to_state_dict_registry(self.minifigure)

        # Build fighter objects
        fighters = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            
            # Load fighter.obj for both collision and visual
            builder.add_convex_collision_from_file(
                filename="./assets/fighter_mm.obj"
            )
            builder.add_visual_from_file(
                filename="./assets/fighter_mm.glb"
            )
            
            builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0))  # Adjust height as needed
            builder.set_scene_idxs([i])
            fighter = builder.build(name=f"fighter-{i}")
            fighters.append(fighter)
            self.remove_from_state_dict_registry(fighter)

        # Merge fighters into a single actor
        self.fighter = Actor.merge(fighters, name="fighter")
        self.add_to_state_dict_registry(self.fighter)

        # Remove from greenscreen
        self.remove_object_from_greenscreen(self.agent.robot)
        self.remove_object_from_greenscreen(self.minifigure)
        self.remove_object_from_greenscreen(self.fighter)


        # Rest configuration
        self.rest_qpos = torch.tensor(
            [0, 0, 0, np.pi / 2, np.pi / 2, 0],
            device=self.device,
        )
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # Camera mount
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # Randomize or set a fixed robot color (from original environment)
        if self.domain_randomization_config.robot_color is not None:
            for link in self.agent.robot.links:
                for i, obj in enumerate(link._objs):
                    # modify the i-th object which is in parallel environment i
                    render_body_component: RenderBodyComponent = (
                        obj.entity.find_component_by_type(RenderBodyComponent)
                    )
                    if render_body_component is not None:
                        for render_shape in render_body_component.render_shapes:
                            for part in render_shape.parts:
                                if (
                                    self.domain_randomization
                                    and self.domain_randomization_config.robot_color
                                    == "random"
                                ):
                                    part.material.set_base_color(
                                        self._batched_episode_rng[i]
                                        .uniform(low=0.0, high=1.0, size=(3,))
                                        .tolist()
                                        + [1]
                                    )
                                else:
                                    part.material.set_base_color(
                                        list(
                                            self.domain_randomization_config.robot_color
                                        )
                                        + [1]
                                    )



    def sample_camera_poses(self, n: int):
        if self.domain_randomization:
            self.base_camera_settings["pos"] = common.to_tensor(
                self.base_camera_settings["pos"], device=self.device
            )
            self.base_camera_settings["target"] = common.to_tensor(
                self.base_camera_settings["target"], device=self.device
            )
            self.domain_randomization_config.max_camera_offset = common.to_tensor(
                self.domain_randomization_config.max_camera_offset, device=self.device
            )

            eyes = randomization.camera.make_camera_rectangular_prism(
                n,
                scale=self.domain_randomization_config.max_camera_offset,
                center=self.base_camera_settings["pos"],
                theta=0,
                device=self.device,
            )
            return randomization.camera.noised_look_at(
                eyes,
                target=self.base_camera_settings["target"],
                look_at_noise=self.domain_randomization_config.camera_target_noise,
                view_axis_rot_noise=self.domain_randomization_config.camera_view_rot_noise,
                device=self.device,
            )
        else:
            return sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Set robot initial pose
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )

            # Initialize minifigure at random position
            spawn_box_pos = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = 0  # Height for real-sized minifigure
                
            # Keep upright orientation but add random Z rotation (spinning)
            random_z_angle = torch.rand(b) * 2 * np.pi  # Random angle between 0 and 2π
            # Convert tensor to numpy array and handle each environment separately
            random_z_angles = random_z_angle.cpu().numpy()
            qs = []
            for i in range(b):
                q = euler2quat(np.pi/2, 0, random_z_angles[i])  # Upright + random spin
                qs.append(q)
            qs = np.array(qs)
            self.minifigure.set_pose(Pose.create_from_pq(xyz, qs))

            # Initialize fighter at random position (different area from minifigure)
            fighter_spawn_box_pos = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0] + 0.1, self.spawn_box_pos[1], 0]  # Offset from minifigure spawn area
            )
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += fighter_spawn_box_pos[env_idx, :2]
            xyz[:, 2] = 0  # Height for fighter
                
            # Keep upright orientation but add random Z rotation (spinning)
            random_z_angle = torch.rand(b) * 2 * np.pi  # Random angle between 0 and 2π
            # Convert tensor to numpy array and handle each environment separately
            random_z_angles = random_z_angle.cpu().numpy()
            qs = []
            for i in range(b):
                q = euler2quat(np.pi/2, 0, random_z_angles[i])  # 90deg X rotation + random spin
                qs.append(q)
            qs = np.array(qs)
            self.fighter.set_pose(Pose.create_from_pq(xyz, qs))

            # Randomize camera poses
            self.camera_mount.set_pose(self.sample_camera_poses(n=b))

    def _before_control_step(self):
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

    def _get_obs_agent(self):
        obs = dict(qpos=self.agent.robot.get_qpos())
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1]
            - self.rest_qpos[:-1],
        )
        if self.obs_mode_struct.state:
            obs.update(
                is_grasped=info["is_grasped"],
                minifigure_pose=self.minifigure.pose.raw_pose,
                fighter_pose=self.fighter.pose.raw_pose,
                tcp_pos=self.agent.tcp_pos,
                tcp_to_minifigure_pos=self.minifigure.pose.p - self.agent.tcp_pos,
                tcp_to_fighter_pos=self.fighter.pose.p - self.agent.tcp_pos,
            )
        return obs

    def evaluate(self):
        tcp_to_minifigure_dist = torch.linalg.norm(
            self.minifigure.pose.p - self.agent.tcp_pos,
            axis=-1,
        )
        tcp_to_fighter_dist = torch.linalg.norm(
            self.fighter.pose.p - self.agent.tcp_pos,
            axis=-1,
        )
        reached_minifigure = tcp_to_minifigure_dist < 0.03
        reached_fighter = tcp_to_fighter_dist < 0.03
        is_grasped = self.agent.is_grasping(self.minifigure)

        target_qpos = self.agent.controller._target_qpos.clone()
        distance_to_rest_qpos = torch.linalg.norm(
            target_qpos[:, :-1] - self.rest_qpos[:-1], axis=-1
        )
        reached_rest_qpos = distance_to_rest_qpos < 0.2

        # Minifigure is lifted if it's above the table
        minifigure_lifted = self.minifigure.pose.p[..., -1] >= 0.0215
        # Fighter is lifted if it's above the table
        fighter_lifted = self.fighter.pose.p[..., -1] >= 0.05

        # Check if minifigure is inside the fighter (close to fighter position)
        minifigure_to_fighter_dist = torch.linalg.norm(
            self.minifigure.pose.p - self.fighter.pose.p,
            axis=-1,
        )
        minifigure_inside_fighter = minifigure_to_fighter_dist < 0.05  # Adjust threshold as needed

        # Success: minifigure is inside the fighter
        success = minifigure_inside_fighter & minifigure_lifted

        # Check if robot is touching table
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.table_scene.table
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.table_scene.table
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        touching_table = torch.logical_or(
            lforce >= 1e-2,
            rforce >= 1e-2,
        )
        
        return {
            "is_grasped": is_grasped,
            "reached_minifigure": reached_minifigure,
            "reached_fighter": reached_fighter,
            "distance_to_rest_qpos": distance_to_rest_qpos,
            "touching_table": touching_table,
            "minifigure_lifted": minifigure_lifted,
            "fighter_lifted": fighter_lifted,
            "minifigure_inside_fighter": minifigure_inside_fighter,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_minifigure_dist = torch.linalg.norm(
            self.minifigure.pose.p - self.agent.tcp_pose.p, axis=1
        )
        tcp_to_fighter_dist = torch.linalg.norm(
            self.fighter.pose.p - self.agent.tcp_pose.p, axis=1
        )
        minifigure_to_fighter_dist = torch.linalg.norm(
            self.minifigure.pose.p - self.fighter.pose.p, axis=1
        )
        
        # Reward for being close to minifigure when not grasped
        reaching_reward = 1 - torch.tanh(5 * tcp_to_minifigure_dist)
        
        # Reward for placing minifigure inside fighter
        placement_reward = 1 - torch.tanh(5 * minifigure_to_fighter_dist)
        
        # Combine rewards: prioritize grasping, then placement
        reward = reaching_reward + info["is_grasped"] * 2 + placement_reward * info["is_grasped"]
        reward -= 2 * info["touching_table"].float()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3 