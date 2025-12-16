"""
DAgger trainer for Phase 2 distillation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb


class DAggerDataset(Dataset):
    """Dataset for DAgger training."""
    
    def __init__(self, observations: List[Dict], actions: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            observations: List of observation dictionaries
            actions: (N, 12) teacher actions
        """
        self.observations = observations
        self.actions = actions
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class DAggerTrainer:
    """DAgger trainer for Phase 2."""
    
    def __init__(
        self,
        env,
        teacher_policy,
        student_policy,
        config: Dict,
        device: str = 'cuda:0'
    ):
        """
        Initialize DAgger trainer.
        
        Args:
            env: Environment
            teacher_policy: Phase 1 trained policy
            student_policy: Phase 2 policy to train
            config: Training configuration
            device: Device
        """
        self.env = env
        self.teacher_policy = teacher_policy
        self.student_policy = student_policy
        self.config = config
        self.device = device
        
        # Optimizer
        # Convert learning rate to float (YAML may parse scientific notation as string)
        learning_rate = float(config['training']['learning_rate'])
        self.optimizer = optim.Adam(
            self.student_policy.parameters(),
            lr=learning_rate
        )
        
        # Dataset
        self.dataset = []
        
        # Initialize wandb
        self.use_wandb = config.get('logging', {}).get('use_wandb', False)
        if self.use_wandb:
            try:
                wandb.init(
                    project=config['logging'].get('project_name', 'vision-locomotion-phase2'),
                    config=config,
                    mode='offline'  # Use offline mode to avoid connection issues
                )
                # Give wandb a moment to stabilize
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
    
    def collect_rollout(self, use_student: bool = True) -> Tuple[List[Dict], torch.Tensor]:
        """
        Collect rollout from environment.
        
        Args:
            use_student: Whether to use student policy (True) or teacher (False)
        
        Returns:
            observations: List of observation dicts
            teacher_actions: Teacher actions for each state
        """
        policy = self.student_policy if use_student else self.teacher_policy
        policy.eval()
        
        observations = []
        teacher_actions = []
        
        obs = self.env.reset()
        hidden_state = None
        
        for step in range(self.config['env']['episode_length']):
            # Get action from student
            with torch.no_grad():
                if use_student:
                    action, _, hidden_state = policy(obs, hidden_state)
                else:
                    action, _, hidden_state = policy(obs, hidden_state)
            
            # Get teacher action for this state
            with torch.no_grad():
                teacher_action, _, _ = self.teacher_policy(obs, None)
            
            observations.append(obs)
            teacher_actions.append(teacher_action)
            
            # Step environment
            obs, _, done, _ = self.env.step(action)
            
            if done.any():
                break
        
        teacher_actions = torch.stack(teacher_actions, dim=0)
        
        return observations, teacher_actions
    
    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of (observations, teacher_actions)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        observations, teacher_actions = batch
        
        self.student_policy.train()
        self.optimizer.zero_grad()
        
        # Forward pass with BPTT
        bptt_length = self.config['training']['bptt_length']
        total_loss = 0.0
        hidden_state = None
        
        for t in range(min(bptt_length, len(observations))):
            obs = observations[t]
            teacher_action = teacher_actions[t]
            
            # Student forward
            student_action, _, hidden_state = self.student_policy(obs, hidden_state)
            
            # MSE loss
            loss = nn.MSELoss()(student_action, teacher_action)
            total_loss += loss
        
        total_loss = total_loss / len(observations)
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student_policy.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'mse': total_loss.item()
        }
    
    def train(self):
        """Train using DAgger algorithm."""
        config = self.config['training']
        
        print("Starting Phase 2 training with DAgger...")
        print(f"Total timesteps: {config['total_timesteps']}")
        print(f"DAgger iterations: {config['dagger_iterations']}")
        
        # Initial behavioral cloning
        print("Collecting initial BC dataset...")
        for _ in tqdm(range(config['bc_initial_samples'] // self.env.num_envs)):
            obs, teacher_actions = self.collect_rollout(use_student=False)
            for i in range(len(obs)):
                self.dataset.append((obs[i], teacher_actions[i]))
        
        print(f"Initial dataset size: {len(self.dataset)}")
        
        # DAgger iterations
        for iteration in range(config['dagger_iterations']):
            print(f"\nDAgger iteration {iteration + 1}/{config['dagger_iterations']}")
            
            # Collect rollout with student
            print("Collecting student rollout...")
            obs, teacher_actions = self.collect_rollout(use_student=True)
            
            # Aggregate dataset
            for i in range(len(obs)):
                self.dataset.append((obs[i], teacher_actions[i]))
            
            print(f"Dataset size: {len(self.dataset)}")
            
            # Train on aggregated dataset
            print("Training student...")
            dataloader = DataLoader(
                self.dataset,
                batch_size=config['batch_size'],
                shuffle=True
            )
            
            for epoch in range(config['n_epochs']):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                    metrics = self.train_step(batch)
                    epoch_loss += metrics['loss']
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                
                if wandb.run is not None:
                    try:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/iteration': iteration,
                            'train/epoch': epoch,
                            'train/dataset_size': len(self.dataset)
                        })
                    except (ConnectionResetError, ConnectionError, RuntimeError) as e:
                        # Silently handle connection errors - training can continue without logging
                        pass
                
                print(f"Epoch {epoch + 1} loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (iteration + 1) % (config['checkpoint_freq'] // config['dagger_iterations']) == 0:
                checkpoint_path = f"checkpoints/phase2_iter_{iteration + 1}.pth"
                self.save_checkpoint(checkpoint_path)
        
        print("Phase 2 training complete!")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'student_state_dict': self.student_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student_policy.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
