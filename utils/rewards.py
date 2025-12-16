"""
Reward functions for legged locomotion training.
Based on the paper's reward structure (Section B).
"""
import torch
import numpy as np


def compute_rewards(
    joint_torques,
    joint_velocities,
    base_velocity,
    base_angular_velocity,
    commanded_velocity,
    commanded_angular_velocity,
    foot_forces,
    foot_velocities,
    contact_states,
    collision_forces,
    reward_scales,
    dt=0.02
):
    """
    Compute reward components for locomotion.
    
    Args:
        joint_torques: (num_envs, 12) joint torques
        joint_velocities: (num_envs, 12) joint velocities
        base_velocity: (num_envs, 3) base linear velocity
        base_angular_velocity: (num_envs, 3) base angular velocity
        commanded_velocity: (num_envs,) commanded forward velocity
        commanded_angular_velocity: (num_envs,) commanded yaw angular velocity
        foot_forces: (num_envs, 4, 3) forces on each foot
        foot_velocities: (num_envs, 4, 3) velocities of each foot
        contact_states: (num_envs, 4) boolean contact states
        collision_forces: (num_envs, num_collision_bodies) collision forces
        reward_scales: dict with reward scales
        dt: time step
    
    Returns:
        rewards: (num_envs,) total reward per environment
        reward_dict: dict with individual reward components
    """
    num_envs = base_velocity.shape[0]
    device = base_velocity.device
    
    # Extract forward velocity and yaw angular velocity
    v_x = base_velocity[:, 0]
    omega_z = base_angular_velocity[:, 2]
    
    # 1. Command tracking reward
    tracking_reward = (
        reward_scales['tracking'] * (
            commanded_velocity - torch.abs(commanded_velocity - v_x) -
            torch.abs(commanded_angular_velocity - omega_z)
        )
    )
    
    # 2. Absolute work penalty
    work = torch.abs(joint_torques * joint_velocities).sum(dim=1)
    work_penalty = reward_scales['work'] * work
    
    # 3. Foot jerk penalty (change in foot force)
    foot_jerk = torch.zeros(num_envs, device=device)
    if foot_forces.shape[1] > 0:  # Check if we have previous forces
        # This would need previous step's forces - simplified here
        foot_jerk = torch.norm(foot_forces.view(num_envs, -1), dim=1)
    foot_jerk_penalty = reward_scales['foot_jerk'] * foot_jerk
    
    # 4. Feet drag penalty
    feet_drag = torch.zeros(num_envs, device=device)
    for foot_idx in range(4):
        in_contact = contact_states[:, foot_idx] > 0.5
        foot_force_z = foot_forces[:, foot_idx, 2]
        foot_vel_xy = torch.norm(foot_velocities[:, foot_idx, :2], dim=1)
        
        dragging = (foot_force_z >= 1.0) & in_contact
        feet_drag += dragging.float() * foot_vel_xy
    
    feet_drag_penalty = reward_scales['feet_drag'] * feet_drag
    
    # 5. Collision penalty (calf and thigh contacts)
    collision_penalty = reward_scales['collision'] * (
        (collision_forces >= 0.1).any(dim=1).float()
    )
    
    # 6. Survival bonus
    survival_bonus = torch.ones(num_envs, device=device) * reward_scales['survival']
    
    # Total reward
    total_reward = (
        tracking_reward +
        work_penalty +
        foot_jerk_penalty +
        feet_drag_penalty +
        collision_penalty +
        survival_bonus
    )
    
    reward_dict = {
        'tracking': tracking_reward,
        'work': work_penalty,
        'foot_jerk': foot_jerk_penalty,
        'feet_drag': feet_drag_penalty,
        'collision': collision_penalty,
        'survival': survival_bonus,
        'total': total_reward
    }
    
    return total_reward, reward_dict
