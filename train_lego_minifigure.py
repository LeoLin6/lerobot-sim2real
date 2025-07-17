#!/usr/bin/env python3
"""
Training script for LEGO Minifigure Pick and Place environment
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the current directory to Python path so we can import our custom environment
sys.path.append(str(Path(__file__).parent))

from lego_minifigure_env import LegoMinifigurePickPlaceEnv

def main():
    # Environment configuration
    env_kwargs = {
        "num_envs": 4,  # Start with a small number for testing
        "obs_mode": "state",
        "control_mode": "pd_joint_target_delta_pos",
        "render_mode": "rgb_array",
        "domain_randomization": True,
        "domain_randomization_config": {
            "randomize_minifigure_color": True,
            "minifigure_scale": 1.0,
            "minifigure_friction_mean": 0.3,
            "minifigure_friction_std": 0.05,
        },
        "base_camera_settings": {
            "fov": 52 * np.pi / 180,
            "pos": [0.5, 0.3, 0.35],
            "target": [0.3, 0.0, 0.1],
        },
        "spawn_box_pos": [0.3, 0.05],
        "spawn_box_half_size": 0.2 / 2,
        "target_pos": [0.3, -0.15, 0.1],
        "use_mesh_model": True,  # Use your Boba Fett LEGO model
    }

    # Create environment
    print("Creating LEGO Minifigure environment...")
    env = LegoMinifigurePickPlaceEnv(**env_kwargs)
    
    # Test the environment
    print("Testing environment...")
    obs, info = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Environment info: {info}")
    
    # Test a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward}, Success = {info.get('success', False)}")
        
        if terminated.any() or truncated.any():
            obs, info = env.reset()
    
    print("Environment test completed successfully!")
    print("\nTo train with PPO, use:")
    print("python lerobot_sim2real/scripts/train_ppo_rgb.py --env_id LegoMinifigurePickPlace-v1")
    print("\n✅ Using Boba Fett LEGO model from assets/boba_lego.obj")
    print("\nTo use a different 3D mesh model:")
    print("1. Replace assets/boba_lego.obj with your model")
    print("2. Update the mesh_path in lego_minifigure_env.py")
    print("3. Or set use_mesh_model=False to use basic shapes")

if __name__ == "__main__":
    main() 