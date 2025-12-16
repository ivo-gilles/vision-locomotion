"""
Curriculum learning utilities.
"""
import numpy as np
from typing import Dict, List, Tuple


class CurriculumManager:
    """Manages curriculum progression for terrain difficulty."""
    
    def __init__(self, terrain_grid: List[List[dict]]):
        """
        Initialize curriculum manager.
        
        Args:
            terrain_grid: Grid of terrain configurations
        """
        self.terrain_grid = terrain_grid
        self.num_rows = len(terrain_grid)
        self.num_cols = len(terrain_grid[0]) if terrain_grid else 0
        
        # Track performance per terrain
        self.terrain_stats = {}
        for row_idx in range(self.num_rows):
            for col_idx in range(self.num_cols):
                self.terrain_stats[(row_idx, col_idx)] = {
                    'successes': 0,
                    'attempts': 0,
                    'avg_displacement': 0.0
                }
    
    def get_terrain_for_env(self, env_id: int, current_row: int, current_col: int) -> Tuple[int, int]:
        """
        Get terrain assignment for environment based on curriculum.
        
        Args:
            env_id: Environment ID
            current_row: Current terrain row
            current_col: Current terrain column
        
        Returns:
            new_row, new_col: New terrain assignment
        """
        stats = self.terrain_stats.get((current_row, current_col), {})
        success_rate = (
            stats['successes'] / stats['attempts']
            if stats['attempts'] > 0 else 0.0
        )
        
        # Promote if success rate > 0.5 and displacement > 50% of terrain length
        if success_rate > 0.5 and stats.get('avg_displacement', 0) > 0.5:
            # Move to next column (harder terrain)
            new_col = min(current_col + 1, self.num_cols - 1)
            new_row = current_row
        # Demote if success rate < 0.3
        elif success_rate < 0.3 and stats['attempts'] > 10:
            new_col = max(current_col - 1, 0)
            new_row = current_row
        else:
            new_row, new_col = current_row, current_col
        
        return new_row, new_col
    
    def update_stats(
        self,
        row: int,
        col: int,
        success: bool,
        displacement: float
    ):
        """
        Update statistics for a terrain.
        
        Args:
            row: Terrain row
            col: Terrain column
            success: Whether episode was successful
            displacement: Distance traveled
        """
        key = (row, col)
        if key not in self.terrain_stats:
            self.terrain_stats[key] = {
                'successes': 0,
                'attempts': 0,
                'avg_displacement': 0.0
            }
        
        stats = self.terrain_stats[key]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
        
        # Update average displacement (exponential moving average)
        alpha = 0.1
        stats['avg_displacement'] = (
            alpha * displacement + (1 - alpha) * stats['avg_displacement']
        )
    
    def get_difficulty_distribution(self) -> Dict[Tuple[int, int], float]:
        """
        Get distribution of environments across terrain difficulties.
        
        Returns:
            distribution: Dictionary mapping (row, col) to fraction of envs
        """
        total = sum(stats['attempts'] for stats in self.terrain_stats.values())
        if total == 0:
            return {}
        
        distribution = {}
        for key, stats in self.terrain_stats.items():
            distribution[key] = stats['attempts'] / total
        
        return distribution
