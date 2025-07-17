#!/usr/bin/env python3

import numpy as np
from mani_skill.envs.utils import gym_vec_env
from mani_skill.utils.wrappers import RecordEpisode

def test_fighter_environment():
    """Test that the fighter object is properly loaded and placed."""
    
    # Create the environment
    env = gym_vec_env(
        "LegoMinifigurePickPlace-v1",
        num_envs=2,
        obs_mode="state",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization=True,
    )
    
    print("Environment created successfully!")
    
    # Reset the environment
    obs = env.reset()
    print(f"Environment reset. Observation keys: {obs.keys()}")
    
    # Check if fighter-related observations are present
    if "fighter_pose" in obs:
        print("✓ Fighter pose observation found")
        print(f"Fighter pose shape: {obs['fighter_pose'].shape}")
    else:
        print("✗ Fighter pose observation not found")
    
    if "tcp_to_fighter_pos" in obs:
        print("✓ TCP to fighter position observation found")
        print(f"TCP to fighter pos shape: {obs['tcp_to_fighter_pos'].shape}")
    else:
        print("✗ TCP to fighter position observation not found")
    
    # Run a few steps to see if everything works
    print("\nRunning a few steps...")
    for step in range(5):
        action = np.random.uniform(-0.1, 0.1, size=(2, 7))  # Random actions
        obs, reward, done, info = env.step(action)
        
        if step == 0:
            print(f"Step {step}: Reward shape: {reward.shape}")
            print(f"Step {step}: Info keys: {list(info.keys())}")
            
            # Check if fighter-related info is present
            if "reached_fighter" in info:
                print("✓ Fighter reach info found")
            if "fighter_lifted" in info:
                print("✓ Fighter lifted info found")
    
    print("\nTest completed successfully!")
    env.close()

if __name__ == "__main__":
    test_fighter_environment() 