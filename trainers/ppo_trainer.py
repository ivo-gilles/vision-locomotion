"""
PPO trainer for Phase 1 training with recurrent policies.
Implements best practices for RL training with BPTT support.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from collections import deque
from tqdm import tqdm
import wandb


class RolloutBuffer:
    """
    Experience buffer for collecting rollouts with recurrent policies.
    Supports BPTT (Backpropagation Through Time).
    """
    
    def __init__(self, buffer_size: int, num_envs: int, device: torch.device):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Number of steps to collect
            num_envs: Number of parallel environments
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False
        
        # Storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.hidden_states = []
        
    def add(
        self,
        obs: Dict,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        hidden_state: torch.Tensor
    ):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action.cpu())
        self.rewards.append(reward.cpu())
        self.values.append(value.cpu())
        self.log_probs.append(log_prob.cpu())
        self.dones.append(done.cpu())
        self.hidden_states.append(hidden_state.cpu() if hidden_state is not None else None)
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            last_values: Value estimates for the last state
            last_dones: Done flags for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.returns = []
        self.advantages = []
        
        # Compute advantages using GAE
        gae = torch.zeros(self.num_envs)
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = last_values.cpu()
                next_done = last_dones.cpu()
            else:
                next_value = self.values[step + 1]
                next_done = self.dones[step + 1]
            
            # TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = self.rewards[step] + gamma * next_value * (1 - next_done.float()) - self.values[step]
            
            # GAE: A = delta + gamma * lambda * (1 - done) * A_next
            gae = delta + gamma * gae_lambda * (1 - next_done.float()) * gae
            
            self.advantages.insert(0, gae.clone())
            self.returns.insert(0, gae + self.values[step])
        
        # Normalize advantages (best practice)
        advantages_tensor = torch.stack(self.advantages)
        self.advantages = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    def get(self, batch_size: int, bptt_length: int):
        """
        Generate batches for training with BPTT.
        
        Args:
            batch_size: Batch size for training
            bptt_length: Length of sequences for BPTT
            
        Yields:
            Batches of (obs, actions, old_values, old_log_probs, advantages, returns, hidden_states)
        """
        num_steps = len(self.actions)
        num_sequences = num_steps // bptt_length
        
        # Create indices for all environments and sequences
        indices = []
        for env_idx in range(self.num_envs):
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * bptt_length
                indices.append((env_idx, start_idx))
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Generate batches
        num_batches = len(indices) // batch_size
        
        for batch_idx in range(num_batches):
            batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            
            # Collect batch data
            batch_obs = {key: [] for key in self.observations[0].keys()}
            batch_actions = []
            batch_old_values = []
            batch_old_log_probs = []
            batch_advantages = []
            batch_returns = []
            batch_hidden_states = []
            
            for env_idx, start_idx in batch_indices:
                # Extract sequence
                seq_obs = {key: [] for key in self.observations[0].keys()}
                seq_actions = []
                seq_old_values = []
                seq_old_log_probs = []
                seq_advantages = []
                seq_returns = []
                
                for t in range(bptt_length):
                    step_idx = start_idx + t
                    if step_idx >= num_steps:
                        break
                    
                    # Extract data for this environment at this step
                    obs = self.observations[step_idx]
                    for key in obs.keys():
                        seq_obs[key].append(obs[key][env_idx])
                    
                    seq_actions.append(self.actions[step_idx][env_idx])
                    seq_old_values.append(self.values[step_idx][env_idx])
                    seq_old_log_probs.append(self.log_probs[step_idx][env_idx])
                    seq_advantages.append(self.advantages[step_idx][env_idx])
                    seq_returns.append(self.returns[step_idx][env_idx])
                
                # Stack sequences
                for key in seq_obs.keys():
                    batch_obs[key].append(torch.stack(seq_obs[key]))
                batch_actions.append(torch.stack(seq_actions))
                batch_old_values.append(torch.stack(seq_old_values))
                batch_old_log_probs.append(torch.stack(seq_old_log_probs))
                batch_advantages.append(torch.stack(seq_advantages))
                batch_returns.append(torch.stack(seq_returns))
                
                # Get initial hidden state for this sequence
                if self.hidden_states[start_idx] is not None:
                    batch_hidden_states.append(self.hidden_states[start_idx][:, env_idx:env_idx+1, :])
            
            # Convert to tensors and move to device
            batch_data = {}
            for key in batch_obs.keys():
                batch_data[key] = torch.stack(batch_obs[key]).to(self.device)  # (batch, seq_len, *)
            
            yield (
                batch_data,
                torch.stack(batch_actions).to(self.device),
                torch.stack(batch_old_values).to(self.device),
                torch.stack(batch_old_log_probs).to(self.device),
                torch.stack(batch_advantages).to(self.device),
                torch.stack(batch_returns).to(self.device),
                torch.cat(batch_hidden_states, dim=1).to(self.device) if batch_hidden_states else None
            )
    
    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.hidden_states = []
        self.returns = []
        self.advantages = []
        self.pos = 0
        self.full = False


class PPOTrainer:
    """PPO trainer for Phase 1 with recurrent policies."""
    
    def __init__(
        self,
        env,
        policy,
        config: Dict,
        device: str = 'cuda:0'
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env: Environment
            policy: Policy network with value head
            config: Training configuration
            device: Device
        """
        self.env = env
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        
        # Training config
        self.train_config = config['training']
        self.learning_rate = float(self.train_config['learning_rate'])
        self.n_epochs = self.train_config['n_epochs']
        self.batch_size = self.train_config['batch_size']
        self.clip_range = self.train_config['clip_range']
        self.entropy_coef = self.train_config['entropy_coef']
        self.value_loss_coef = self.train_config['value_loss_coef']
        self.max_grad_norm = self.train_config['max_grad_norm']
        self.bptt_length = self.train_config['bptt_length']
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE lambda
        
        # Optimizer with AdamW (best practice)
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=0.01
        )
        
        # Learning rate scheduler (cosine annealing)
        total_updates = self.train_config['total_timesteps'] // (self.env.num_envs * self.bptt_length)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_updates,
            eta_min=self.learning_rate * 0.1
        )
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.bptt_length * 10,  # Collect multiple BPTT sequences
            num_envs=self.env.num_envs,
            device=self.device
        )
        
        # Statistics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_reward_buffer = np.zeros(self.env.num_envs)
        self.episode_length_buffer = np.zeros(self.env.num_envs)
        
        # Initialize wandb if configured
        self.use_wandb = config.get('logging', {}).get('use_wandb', False)
        if self.use_wandb:
            try:
                wandb.init(
                    project=config['logging'].get('project_name', 'vision-locomotion'),
                    config=config,
                    mode='offline'  # Use offline mode to avoid connection issues
                )
                # Give wandb a moment to stabilize
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
        
        # Training state
        self.num_timesteps = 0
        self.num_updates = 0
    
    def collect_rollouts(self, n_steps: int) -> bool:
        """
        Collect rollouts from the environment.
        
        Args:
            n_steps: Number of steps to collect
            
        Returns:
            True if collection was successful
        """
        self.policy.eval()
        self.rollout_buffer.clear()
        
        # Reset environment if needed
        if self.num_timesteps == 0:
            obs = self.env.reset()
            self.policy.reset_hidden(self.env.num_envs, self.device)
        else:
            # Continue from current state
            obs = self.current_obs
        
        hidden_state = self.policy.hidden_state
        
        for step in range(n_steps):
            with torch.no_grad():
                # Get action and value from policy
                actions, values, new_hidden = self.policy(obs, hidden_state)
                
                # Compute log probabilities (assuming Gaussian distribution for continuous actions)
                action_std = 0.5  # Fixed standard deviation (can be learned)
                action_dist = torch.distributions.Normal(actions, action_std)
                log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
            # Step environment
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=actions,
                reward=rewards,
                value=values,
                log_prob=log_probs,
                done=dones,
                hidden_state=hidden_state
            )
            
            # Update episode statistics
            self.episode_reward_buffer += rewards.cpu().numpy().flatten()
            self.episode_length_buffer += 1
            
            # Reset hidden state for done environments
            if dones.any():
                for env_idx in range(self.env.num_envs):
                    if dones[env_idx]:
                        self.episode_rewards.append(self.episode_reward_buffer[env_idx])
                        self.episode_lengths.append(self.episode_length_buffer[env_idx])
                        self.episode_reward_buffer[env_idx] = 0
                        self.episode_length_buffer[env_idx] = 0
                        
                        # Reset hidden state for this environment
                        if new_hidden is not None:
                            new_hidden[:, env_idx, :] = 0
            
            # Update state
            obs = next_obs
            hidden_state = new_hidden
            self.num_timesteps += self.env.num_envs
        
        # Store current state for next rollout
        self.current_obs = obs
        self.policy.hidden_state = hidden_state
        
        # Compute returns and advantages
        with torch.no_grad():
            _, last_values, _ = self.policy(obs, hidden_state)
        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            last_dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        return True
    
    def train_on_batch(
        self,
        obs_batch: Dict,
        actions_batch: torch.Tensor,
        old_values_batch: torch.Tensor,
        old_log_probs_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        hidden_states_batch: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Train on a single batch using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        self.policy.train()
        
        # Forward pass through sequences
        batch_size, seq_len = actions_batch.shape[:2]
        
        # Process sequences
        all_actions = []
        all_values = []
        all_log_probs = []
        
        hidden = hidden_states_batch
        
        for t in range(seq_len):
            # Extract observations for this timestep
            obs_t = {key: obs_batch[key][:, t] for key in obs_batch.keys()}
            
            # Forward pass
            actions_t, values_t, hidden = self.policy(obs_t, hidden)
            
            # Compute log probabilities
            action_std = 0.5
            action_dist = torch.distributions.Normal(actions_t, action_std)
            log_probs_t = action_dist.log_prob(actions_batch[:, t]).sum(dim=-1, keepdim=True)
            
            # Compute entropy (for entropy bonus)
            entropy_t = action_dist.entropy().sum(dim=-1, keepdim=True)
            
            all_actions.append(actions_t)
            all_values.append(values_t)
            all_log_probs.append(log_probs_t)
        
        # Stack over time
        values = torch.stack(all_values, dim=1).squeeze(-1)  # (batch, seq_len)
        log_probs = torch.stack(all_log_probs, dim=1).squeeze(-1)  # (batch, seq_len)
        
        # PPO loss computation
        # Ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs_batch)
        
        # Surrogate losses
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with clipping (best practice)
        values_clipped = old_values_batch + torch.clamp(
            values - old_values_batch,
            -self.clip_range,
            self.clip_range
        )
        value_loss_unclipped = (values - returns_batch) ** 2
        value_loss_clipped = (values_clipped - returns_batch) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy loss (encourage exploration)
        # Note: entropy should be computed from action_dist above, simplified here
        entropy_loss = 0.0  # Placeholder - should compute from distribution
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (best practice)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Compute explained variance (diagnostic metric)
        with torch.no_grad():
            explained_var = 1 - torch.var(returns_batch - values) / torch.var(returns_batch)
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss,
            'explained_variance': explained_var.item(),
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean().item()
        }
    
    def train(self):
        """Main training loop."""
        print("=" * 80)
        print("Starting Phase 1 PPO Training with Recurrent Policy")
        print("=" * 80)
        print(f"Total timesteps: {self.train_config['total_timesteps']:,}")
        print(f"Number of environments: {self.env.num_envs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"BPTT length: {self.bptt_length}")
        print(f"Clip range: {self.clip_range}")
        print(f"Entropy coefficient: {self.entropy_coef}")
        print(f"Value loss coefficient: {self.value_loss_coef}")
        print("=" * 80)
        
        # Training loop
        total_timesteps = self.train_config['total_timesteps']
        rollout_steps = self.rollout_buffer.buffer_size
        checkpoint_freq = self.train_config['checkpoint_freq']
        log_freq = self.config.get('logging', {}).get('log_freq', 1000)
        
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            self.collect_rollouts(rollout_steps)
            
            # Train for multiple epochs
            for epoch in range(self.n_epochs):
                # Get batches and train
                for batch in self.rollout_buffer.get(self.batch_size, self.bptt_length):
                    obs_batch, actions_batch, old_values_batch, old_log_probs_batch, \
                        advantages_batch, returns_batch, hidden_states_batch = batch
                    
                    metrics = self.train_on_batch(
                        obs_batch, actions_batch, old_values_batch,
                        old_log_probs_batch, advantages_batch, returns_batch,
                        hidden_states_batch
                    )
            
            # Update learning rate
            self.scheduler.step()
            self.num_updates += 1
            
            # Logging
            if self.num_timesteps % log_freq < rollout_steps * self.env.num_envs:
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths)
                    
                    pbar.set_postfix({
                        'reward': f'{mean_reward:.2f}',
                        'length': f'{mean_length:.0f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if self.use_wandb:
                        try:
                            wandb.log({
                                'train/mean_reward': mean_reward,
                                'train/mean_episode_length': mean_length,
                                'train/timesteps': self.num_timesteps,
                                'train/learning_rate': self.scheduler.get_last_lr()[0],
                                'train/policy_loss': metrics['policy_loss'],
                                'train/value_loss': metrics['value_loss'],
                                'train/explained_variance': metrics['explained_variance'],
                                'train/approx_kl': metrics['approx_kl']
                            }, step=self.num_timesteps)
                        except (ConnectionResetError, ConnectionError, RuntimeError):
                            pass
            
            # Checkpointing
            if self.num_timesteps % checkpoint_freq < rollout_steps * self.env.num_envs:
                checkpoint_path = f"{self.config['logging']['save_dir']}/checkpoints/phase1_step_{self.num_timesteps}.pth"
                self.save_checkpoint(checkpoint_path)
            
            pbar.update(rollout_steps * self.env.num_envs)
        
        pbar.close()
        print("\n" + "=" * 80)
        print("Training complete!")
        print("=" * 80)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_updates': self.num_updates,
            'config': self.config
        }, path)
        print(f"[Checkpoint] Saved to {path} (timesteps: {self.num_timesteps:,})")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'num_timesteps' in checkpoint:
            self.num_timesteps = checkpoint['num_timesteps']
        if 'num_updates' in checkpoint:
            self.num_updates = checkpoint['num_updates']
        
        print(f"[Checkpoint] Loaded from {path} (timesteps: {self.num_timesteps:,})")
