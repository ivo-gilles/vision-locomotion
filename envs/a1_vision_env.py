"""
Custom A1 environment for vision-based locomotion.
Updated for Isaac Sim (Omniverse) compatibility on Windows.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import os

# Isaac Sim imports (Omniverse)
try:
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.sensor import Camera
    from pxr import UsdGeom, Gf
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    # Fallback for Isaac Gym Preview (legacy)
    try:
        from isaacgym import gymapi, gymutil
        ISAAC_SIM_AVAILABLE = False
        ISAAC_GYM_AVAILABLE = True
    except ImportError:
        ISAAC_SIM_AVAILABLE = False
        ISAAC_GYM_AVAILABLE = False
        print("Warning: Neither Isaac Sim nor Isaac Gym Preview found!")

# Try to import legged_gym, but make it optional
try:
    from legged_gym.envs import A1RoughCfg, A1RoughEnv
    LEGGED_GYM_AVAILABLE = True
except ImportError:
    LEGGED_GYM_AVAILABLE = False
    print("Warning: legged_gym not found. Some features may not work.")
    # Create dummy classes for compatibility
    class A1RoughCfg:
        pass
    class A1RoughEnv:
        def __init__(self, *args, **kwargs):
            pass

from utils.randomization import DomainRandomizer
from utils.terrain import TerrainGenerator


class A1VisionEnv(A1RoughEnv if LEGGED_GYM_AVAILABLE else object):
    """
    Extended A1 environment with scandots (Phase 1) or depth (Phase 2) support.
    Updated for Isaac Sim compatibility on Windows.
    """
    
    def __init__(
        self,
        cfg: A1RoughCfg,
        phase: int = 1,
        num_scandots: int = 16,
        depth_resolution: Tuple[int, int] = (58, 87),
        sim_device: str = "cuda:0",
        headless: bool = False
    ):
        """
        Initialize vision-based A1 environment.
        
        Args:
            cfg: A1RoughCfg configuration
            phase: Training phase (1 for scandots, 2 for depth)
            num_scandots: Number of scandot points (Phase 1)
            depth_resolution: Depth image resolution (Phase 2)
            sim_device: Simulation device
            headless: Whether to run headless
        """
        self.phase = phase
        self.num_scandots = num_scandots
        self.depth_resolution = depth_resolution
        self.device = torch.device(sim_device)
        self.headless = headless
        self.use_isaac_sim = ISAAC_SIM_AVAILABLE
        self.cfg = cfg  # Store cfg for later access
        
        # Modify cfg for vision
        if hasattr(cfg, 'terrain'):
            if not hasattr(cfg.terrain, 'num_rows'):
                cfg.terrain.num_rows = getattr(cfg.terrain, 'num_rows', 20)
            if not hasattr(cfg.terrain, 'num_cols'):
                cfg.terrain.num_cols = getattr(cfg.terrain, 'num_cols', 10)
        
        # Initialize parent if legged_gym is available
        if LEGGED_GYM_AVAILABLE:
            super().__init__(cfg, sim_device, headless)
            self.num_envs = getattr(self, 'num_envs', 1)
        else:
            # Standalone initialization for Isaac Sim
            self.num_envs = getattr(cfg, 'num_envs', 1) if hasattr(cfg, 'num_envs') else 1
            self.world = None
            if self.use_isaac_sim:
                self._init_isaac_sim_world()
        
        # Initialize randomizer and terrain generator
        domain_rand_config = getattr(cfg, 'domain_randomization', {}) if hasattr(cfg, 'domain_randomization') else {}
        # Convert domain_rand_config to dict if it's an object
        if not isinstance(domain_rand_config, dict):
            domain_rand_config = {k: getattr(domain_rand_config, k) for k in dir(domain_rand_config) 
                                 if not k.startswith('_') and not callable(getattr(domain_rand_config, k, None))}
        
        terrain_config = getattr(cfg, 'terrain', {}) if hasattr(cfg, 'terrain') else {}
        # Convert terrain_config to dict if it's an object
        if not isinstance(terrain_config, dict):
            terrain_dict = {}
            # Extract all attributes from terrain object
            for attr in ['num_rows', 'num_cols', 'terrain_size', 'fractal_height_range', 'types']:
                if hasattr(terrain_config, attr):
                    value = getattr(terrain_config, attr)
                    # Convert attribute name from snake_case to match dict keys
                    terrain_dict[attr] = value
            # Set defaults if missing
            terrain_dict.setdefault('num_rows', 20)
            terrain_dict.setdefault('num_cols', 10)
            terrain_dict.setdefault('terrain_size', 8.0)
            terrain_dict.setdefault('fractal_height_range', [0.02, 0.04])
            terrain_dict.setdefault('types', ['flat'])
            terrain_config = terrain_dict
        else:
            # Ensure it's a proper dict with defaults
            terrain_config = dict(terrain_config)  # Make a copy
            terrain_config.setdefault('num_rows', 20)
            terrain_config.setdefault('num_cols', 10)
            terrain_config.setdefault('terrain_size', 8.0)
            terrain_config.setdefault('fractal_height_range', [0.02, 0.04])
            terrain_config.setdefault('types', ['flat'])
        
        self.randomizer = DomainRandomizer(domain_rand_config)
        self.terrain_generator = TerrainGenerator(terrain_config)
        
        # Setup scandots (Phase 1)
        if phase == 1:
            self._setup_scandots()
        
        # Setup depth camera (Phase 2)
        if phase == 2:
            self._setup_depth_camera()
        
        # Curriculum tracking
        self.curriculum_tracker = {}
        for env_id in range(self.num_envs):
            self.curriculum_tracker[env_id] = {
                'current_row': 0,
                'current_col': 0,
                'displacement': 0.0
            }
    
    def _init_isaac_sim_world(self):
        """Initialize Isaac Sim world (Omniverse)."""
        if not self.use_isaac_sim:
            return
        # World initialization will be handled by the simulation setup
        # This is a placeholder for future implementation
        pass
    
    def _setup_scandots(self):
        """Setup scandot sampling points in robot frame."""
        # Create grid of points under and in front of robot
        # Points are in robot's local frame (x forward, y left, z up)
        x_range = np.linspace(0.1, 0.8, 4)  # 0.1m to 0.8m in front
        y_range = np.linspace(-0.3, 0.3, 4)  # -0.3m to 0.3m left/right
        
        self.scandot_positions = []
        for x in x_range:
            for y in y_range:
                self.scandot_positions.append([x, y, 0.0])  # z will be queried
        
        # Truncate to num_scandots
        self.scandot_positions = np.array(self.scandot_positions[:self.num_scandots])
    
    def _setup_depth_camera(self):
        """Setup depth camera sensor."""
        self.camera_handles = []
        
        if self.use_isaac_sim:
            # Isaac Sim (Omniverse) camera setup
            try:
                from omni.isaac.sensor import Camera
                from pxr import Gf
                
                for env_id in range(self.num_envs):
                    # Create camera prim path
                    camera_path = f"/World/envs/env_{env_id}/robot/camera"
                    
                    # Create camera
                    camera = Camera(
                        prim_path=camera_path,
                        name=f"depth_camera_{env_id}",
                        position=Gf.Vec3d(0.1, 0.0, 0.15),  # Slightly forward and up
                        resolution=(640, 480),
                        horizontal_fov=87.0
                    )
                    self.camera_handles.append(camera)
            except Exception as e:
                print(f"Warning: Could not setup Isaac Sim cameras: {e}")
                self.camera_handles = [None] * self.num_envs
        else:
            # Legacy Isaac Gym Preview camera setup
            if ISAAC_GYM_AVAILABLE:
                camera_props = gymapi.CameraProperties()
                camera_props.width = 480
                camera_props.height = 640
                camera_props.horizontal_fov = 87.0  # degrees
                
                # Camera pose (in robot frame, front-facing)
                camera_pose = gymapi.Transform()
                camera_pose.p = gymapi.Vec3(0.1, 0.0, 0.15)  # Slightly forward and up
                camera_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)  # Looking forward
                
                for env_id in range(self.num_envs):
                    if hasattr(self, 'envs') and hasattr(self, 'actor_handles'):
                        env_handle = self.envs[env_id]
                        robot_handle = self.actor_handles[env_id]
                        
                        camera_handle = self.gym.create_camera_sensor(
                            env_handle, camera_props
                        )
                        self.gym.attach_camera_to_body(
                            camera_handle, env_handle, robot_handle,
                            camera_pose, gymapi.FOLLOW_TRANSFORM
                        )
                        self.camera_handles.append(camera_handle)
                    else:
                        self.camera_handles.append(None)
            else:
                self.camera_handles = [None] * self.num_envs
    
    def _get_scandots(self) -> torch.Tensor:
        """
        Query terrain heights at scandot positions.
        
        Returns:
            scandots: (num_envs, num_scandots, 3) [x, y, z] positions
        """
        scandots = torch.zeros(
            self.num_envs, self.num_scandots, 3,
            device=self.device
        )
        
        for env_id in range(self.num_envs):
            # Transform scandot positions to world frame
            for i, local_pos in enumerate(self.scandot_positions):
                if self.use_isaac_sim:
                    # Isaac Sim: Use raycasting to get terrain height
                    try:
                        from omni.isaac.core.utils.prims import get_prim_at_path
                        from omni.isaac.core.utils.rotations import quat_to_euler_angles
                        
                        # Get robot pose (simplified - would get from actual robot state)
                        # In practice, query robot's current pose and transform local positions
                        world_pos = local_pos.copy()  # Simplified
                        
                        # Raycast to get terrain height
                        # This is a placeholder - actual implementation would use Isaac Sim's raycasting
                        height = 0.0  # Would query from terrain using raycast
                        
                        scandots[env_id, i] = torch.tensor(
                            [world_pos[0], world_pos[1], height],
                            device=self.device
                        )
                    except Exception as e:
                        # Fallback: use zero height
                        scandots[env_id, i] = torch.tensor(
                            [local_pos[0], local_pos[1], 0.0],
                            device=self.device
                        )
                else:
                    # Legacy Isaac Gym Preview
                    if ISAAC_GYM_AVAILABLE and hasattr(self, 'gym') and hasattr(self, 'envs'):
                        try:
                            env_handle = self.envs[env_id]
                            robot_handle = self.actor_handles[env_id]
                            
                            # Get robot pose
                            robot_pose = self.gym.get_actor_rigid_body_states(
                                env_handle, robot_handle, gymapi.STATE_POS
                            )['pose']
                            
                            # Transform scandot positions to world frame
                            world_pos = local_pos  # Simplified
                            
                            # Query terrain height (simplified - would use raycast)
                            height = 0.0  # Would query from terrain
                            
                            scandots[env_id, i] = torch.tensor(
                                [world_pos[0], world_pos[1], height],
                                device=self.device
                            )
                        except Exception as e:
                            # Fallback
                            scandots[env_id, i] = torch.tensor(
                                [local_pos[0], local_pos[1], 0.0],
                                device=self.device
                            )
                    else:
                        # No simulation available - use zero height
                        scandots[env_id, i] = torch.tensor(
                            [local_pos[0], local_pos[1], 0.0],
                            device=self.device
                        )
        
        return scandots
    
    def _get_depth_images(self) -> torch.Tensor:
        """
        Get depth images from camera.
        
        Returns:
            depth_images: (num_envs, height, width) depth values
        """
        depth_images = torch.zeros(
            self.num_envs, *self.depth_resolution,
            device=self.device
        )
        
        for env_id in range(self.num_envs):
            camera_handle = self.camera_handles[env_id]
            
            if camera_handle is None:
                continue
            
            if self.use_isaac_sim:
                # Isaac Sim (Omniverse) depth image retrieval
                try:
                    # Get depth data from camera
                    depth_data = camera_handle.get_current_frame()["distance_to_image_plane"]
                    if depth_data is not None:
                        depth_array = np.array(depth_data)
                        depth_processed = self._preprocess_depth(depth_array)
                        depth_images[env_id] = torch.tensor(
                            depth_processed, device=self.device
                        )
                except Exception as e:
                    print(f"Warning: Could not get depth image for env {env_id}: {e}")
            else:
                # Legacy Isaac Gym Preview depth image retrieval
                if ISAAC_GYM_AVAILABLE and hasattr(self, 'gym') and hasattr(self, 'envs'):
                    # Render camera
                    self.gym.render_camera_sensor(
                        self.envs[env_id], camera_handle
                    )
                    
                    # Get depth image
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[env_id], camera_handle,
                        gymapi.IMAGE_DEPTH
                    )
                    
                    # Convert to numpy and process
                    try:
                        import gymtorch
                        depth_array = gymtorch.wrap_tensor(depth_tensor).cpu().numpy()
                    except ImportError:
                        # Fallback if gymtorch not available
                        depth_array = np.zeros((480, 640))
                    
                    # Crop and downsample
                    depth_processed = self._preprocess_depth(depth_array)
                    
                    depth_images[env_id] = torch.tensor(
                        depth_processed, device=self.device
                    )
        
        return depth_images
    
    def _preprocess_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """
        Preprocess depth image: crop, hole-fill, downsample.
        
        Args:
            depth_array: Raw depth image
        
        Returns:
            processed: Processed depth image
        """
        # Crop 200 pixels from left
        if depth_array.shape[1] > 200:
            depth_array = depth_array[:, 200:]
        
        # Nearest neighbor hole-filling (simplified)
        # In practice, use proper hole-filling algorithm
        
        # Downsample to target resolution
        from scipy.ndimage import zoom
        zoom_factors = (
            self.depth_resolution[0] / depth_array.shape[0],
            self.depth_resolution[1] / depth_array.shape[1]
        )
        depth_processed = zoom(depth_array, zoom_factors, order=0)
        
        return depth_processed
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Get observations including vision input.
        
        Returns:
            observations: Dictionary with proprioception, vision, commands
        """
        # Get base observations from parent class (if available)
        if LEGGED_GYM_AVAILABLE:
            try:
                base_obs = super()._get_observations()
            except:
                base_obs = {}
        else:
            base_obs = {}
        
        # Get proprioception (from parent or create dummy)
        if hasattr(self, 'dof_pos') and hasattr(self, 'default_dof_pos'):
            dof_pos_diff = self.dof_pos - self.default_dof_pos
        else:
            dof_pos_diff = torch.zeros(self.num_envs, 12, device=self.device)
        
        if hasattr(self, 'dof_vel'):
            dof_vel = self.dof_vel
        else:
            dof_vel = torch.zeros(self.num_envs, 12, device=self.device)
        
        if hasattr(self, 'base_lin_vel'):
            base_lin_vel = self.base_lin_vel
        else:
            base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        if hasattr(self, 'base_ang_vel'):
            base_ang_vel = self.base_ang_vel
        else:
            base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        if hasattr(self, 'projected_gravity'):
            projected_gravity = self.projected_gravity
        else:
            projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        
        if hasattr(self, 'last_actions'):
            last_actions = self.last_actions
        else:
            last_actions = torch.zeros(self.num_envs, 12, device=self.device)
        
        proprioception = torch.cat([
            dof_pos_diff,  # Joint angles
            dof_vel,  # Joint velocities
            base_lin_vel,  # Base linear velocity
            base_ang_vel,  # Base angular velocity
            projected_gravity,  # Roll/pitch
            last_actions,  # Previous actions
        ], dim=-1)
        
        # Get vision input
        if self.phase == 1:
            vision_input = self._get_scandots()
        else:
            vision_input = self._get_depth_images()
        
        # Get commands
        if hasattr(self, 'commands') and self.commands is not None:
            cmd_vx = self.commands[:, 0:1] if self.commands.shape[1] > 0 else torch.zeros(self.num_envs, 1, device=self.device)
            cmd_omega = self.commands[:, 2:3] if self.commands.shape[1] > 2 else torch.zeros(self.num_envs, 1, device=self.device)
        else:
            cmd_vx = torch.zeros(self.num_envs, 1, device=self.device)
            cmd_omega = torch.zeros(self.num_envs, 1, device=self.device)
        
        commands = torch.cat([
            cmd_vx,  # v_x
            torch.zeros(self.num_envs, 1, device=self.device),  # v_y (0)
            cmd_omega,  # omega_z
        ], dim=-1)
        
        observations = {
            'proprioception': proprioception,
            'scandots' if self.phase == 1 else 'depth': vision_input,
            'commands': commands
        }
        
        # Add privileged info for RMA Phase 1
        if hasattr(self, 'use_privileged') and self.use_privileged:
            privileged = self._get_privileged_info()
            observations['privileged'] = privileged
        
        # Apply observation noise
        observation_noise_config = {}
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'observation_noise'):
            observation_noise_config = self.cfg.observation_noise
        elif hasattr(self, 'cfg') and isinstance(self.cfg, dict) and 'observation_noise' in self.cfg:
            observation_noise_config = self.cfg['observation_noise']
        
        if observation_noise_config:
            observations = self.randomizer.apply_observation_noise(
                observations, observation_noise_config
            )
        
        return observations
    
    def _get_privileged_info(self) -> torch.Tensor:
        """
        Get privileged information (Phase 1 RMA only).
        
        Returns:
            privileged: (num_envs, num_privileged) privileged info
        """
        # COM shift, friction, motor strength, etc.
        privileged = torch.zeros(
            self.num_envs, 20, device=self.device
        )
        
        # This would be populated from simulation state
        # Simplified here
        
        return privileged
    
    def reset(self):
        """Reset all environments and return initial observations."""
        if LEGGED_GYM_AVAILABLE and hasattr(super(), 'reset'):
            # Use parent's reset if available
            try:
                obs = super().reset()
                # Convert to dict format if needed
                if not isinstance(obs, dict):
                    obs = self._get_observations()
            except:
                # Fallback to standalone reset
                env_ids = list(range(self.num_envs))
                self.reset_idx(env_ids)
                obs = self._get_observations()
        else:
            # Standalone reset - reset all environments
            env_ids = list(range(self.num_envs))
            self.reset_idx(env_ids)
            # Get initial observations
            obs = self._get_observations()
        return obs
    
    def reset_idx(self, env_ids):
        """Reset specific environments with curriculum."""
        # Apply domain randomization
        if hasattr(self, 'randomizer'):
            random_params = self.randomizer.sample_parameters(
                len(env_ids), self.device
            )
        
        # Reset parent if available
        if LEGGED_GYM_AVAILABLE and hasattr(super(), 'reset_idx'):
            super().reset_idx(env_ids)
        else:
            # Standalone reset - would reset simulation state here
            # For now, just mark as reset
            pass
        
        # Update curriculum tracker if available
        if hasattr(self, 'curriculum_tracker'):
            for env_id in env_ids:
                self.curriculum_tracker[env_id]['displacement'] = 0.0
    
    def step(self, actions: torch.Tensor) -> Tuple:
        """
        Step environment.
        
        Args:
            actions: (num_envs, 12) joint target angles
        
        Returns:
            observations, rewards, dones, info
        """
        # Step parent if available
        if LEGGED_GYM_AVAILABLE and hasattr(super(), 'step'):
            try:
                obs, rewards, dones, info = super().step(actions)
                # Ensure obs is in dict format
                if not isinstance(obs, dict):
                    obs = self._get_observations()
            except:
                # Fallback: create dummy step
                obs = self._get_observations()
                rewards = torch.zeros(self.num_envs, device=self.device)
                dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                info = {}
        else:
            # Standalone step - would step simulation here
            # For now, just get observations
            obs = self._get_observations()
            rewards = torch.zeros(self.num_envs, device=self.device)
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            info = {}
        
        # Update curriculum
        if hasattr(self, '_update_curriculum'):
            self._update_curriculum()
        
        # Apply random pushes
        if hasattr(self, '_apply_random_pushes'):
            self._apply_random_pushes()
        
        return obs, rewards, dones, info
    
    def _update_curriculum(self):
        """Update curriculum based on robot performance."""
        # Track displacement
        if hasattr(self, 'root_states'):
            for env_id in range(self.num_envs):
                if env_id < self.root_states.shape[0]:
                    base_pos = self.root_states[env_id, :3]
                    self.curriculum_tracker[env_id]['displacement'] = base_pos[0].item() if torch.is_tensor(base_pos[0]) else base_pos[0]
        
        # Promote/demote based on performance
        # Simplified - would implement full curriculum logic
    
    def _apply_random_pushes(self):
        """Apply random pushes to robots."""
        if self.use_isaac_sim:
            # Isaac Sim: Get simulation time
            try:
                if self.world is not None:
                    current_time = self.world.current_time
                else:
                    current_time = 0.0
            except:
                current_time = 0.0
        else:
            # Legacy Isaac Gym Preview
            if ISAAC_GYM_AVAILABLE and hasattr(self, 'gym') and hasattr(self, 'sim'):
                current_time = self.gym.get_sim_time(self.sim)
            else:
                current_time = 0.0
        
        for env_id in range(self.num_envs):
            if self.randomizer.should_apply_push(current_time):
                push_force = self.randomizer.sample_push(1, self.device)
                # Apply push to robot (simplified)
                # In practice, apply force to base link
                # This would need to be implemented based on the specific simulation API
