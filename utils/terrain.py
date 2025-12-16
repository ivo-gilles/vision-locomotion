"""
Terrain generation utilities for curriculum learning.
"""
import numpy as np
from typing import List, Tuple


class TerrainGenerator:
    """Generates various terrain types for training."""
    
    def __init__(self, config: dict):
        """
        Initialize terrain generator.
        
        Args:
            config: Terrain configuration
        """
        self.config = config
        self.terrain_types = config.get('types', ['flat'])
        self.terrain_size = config.get('terrain_size', 8.0)
        self.fractal_range = config.get('fractal_height_range', [0.02, 0.04])
        self.num_rows = config.get('num_rows', 20)
        self.num_cols = config.get('num_cols', 10)
    
    def generate_terrain_grid(self) -> List[List[dict]]:
        """
        Generate a grid of terrains with increasing difficulty.
        
        Returns:
            terrain_grid: List of lists, each containing terrain configs
        """
        terrain_grid = []
        
        for row_idx, terrain_type in enumerate(self.terrain_types):
            row = []
            for col_idx in range(self.num_cols):
                # Difficulty increases from left to right
                difficulty = col_idx / (self.num_cols - 1)
                
                terrain_config = {
                    'type': terrain_type,
                    'difficulty': difficulty,
                    'size': self.terrain_size,
                    'position': (row_idx * self.terrain_size, col_idx * self.terrain_size),
                    'fractal_height': (
                        self.fractal_range[0] +
                        difficulty * (self.fractal_range[1] - self.fractal_range[0])
                    )
                }
                
                row.append(terrain_config)
            
            terrain_grid.append(row)
        
        return terrain_grid
    
    def generate_stairs(
        self,
        num_steps: int,
        step_height: float,
        step_width: float,
        difficulty: float = 1.0
    ) -> np.ndarray:
        """
        Generate stair terrain height map.
        
        Args:
            num_steps: Number of steps
            step_height: Height of each step
            step_width: Width of each step
            difficulty: Difficulty multiplier
        
        Returns:
            height_map: 2D array of heights
        """
        # This would integrate with Isaac Gym's terrain system
        # Simplified version here
        height_map = np.zeros((100, 100))
        
        for i in range(num_steps):
            start_x = int(i * step_width * 10)
            end_x = int((i + 1) * step_width * 10)
            height = (i + 1) * step_height * difficulty
            height_map[:, start_x:end_x] = height
        
        return height_map
    
    def generate_stepping_stones(
        self,
        num_stones: int,
        stone_size: float,
        gap_size: float,
        difficulty: float = 1.0
    ) -> np.ndarray:
        """
        Generate stepping stones terrain.
        
        Args:
            num_stones: Number of stones
            stone_size: Size of each stone
            gap_size: Gap between stones
            difficulty: Difficulty multiplier
        
        Returns:
            height_map: 2D array of heights
        """
        height_map = np.zeros((100, 100))
        
        for i in range(num_stones):
            x_pos = int(i * (stone_size + gap_size) * 10)
            stone_pixels = int(stone_size * 10)
            height_map[40:60, x_pos:x_pos+stone_pixels] = 0.1 * difficulty
        
        return height_map
    
    def add_fractal_noise(self, height_map: np.ndarray, amplitude: float) -> np.ndarray:
        """
        Add Perlin-like fractal noise to terrain.
        
        Args:
            height_map: Base height map
            amplitude: Noise amplitude
        
        Returns:
            noisy_map: Height map with noise added
        """
        # Simplified Perlin noise - in practice use proper implementation
        noise = np.random.randn(*height_map.shape) * amplitude
        return height_map + noise
