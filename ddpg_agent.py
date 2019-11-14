import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(2e5)  # replay buffer size                     - default int(1e5), try to int(1e6)
BATCH_SIZE = 1024       # minibatch size                         - default 128 
GAMMA = 0.99            # discount factor                        - default 0.99
TAU = 2e-3              # for soft update of target parameters   - default 1e-3
LR_ACTOR = 1e-3 # 1e-3  # learning rate of the actor             - default 1e-4 , try to 1e-3
LR_CRITIC = 1e-3 #1e-3  # learning rate of the critic            - default 1e-3, try 2e-3, 2e-4, 3e-3, 3e-4
OPTIM = 'Adam'          # optimizer to use                       - default is Adam, experimented with AdamW
WEIGHT_DECAY = 0        # L2 weight decay                        - default for Adam = 0
AWEIGHT_DECAY = 0.01    # L2 weight decay for AdamW              - default for AdamW = 0.01
AMSGRAD = False         # AMSGrad variant of optimizer           - default False
LEAKINESS = 0.01        # leakiness, leaky_relu used if > 0      - default for leaky_relu is 0.01
USEKAIMING = False      # kaiming normal weight initialization   - default False

PRIORITIZED = False     # Use Prioritized Replay Buffer          - default False
PROB_ALPHA =0.6         # Used with Prioritized Replay Buffer    - default 0.6
BETA = 0.4              # Used with Prioritized Replay Buffer    - default 0.4
BETAFRAMES = 1000       # Used with Prioritized Replay Buffer    - default 1000
PRIOS_FACTOR = 1e-5     # Used with Prioritized Replay Buffer    - default 1e-5

# Suggested on slack:
LEARN_EVERY = 1         # learning timestep interval (20 for the Continuous Reacher task, 1 for Tennis)
LEARN_NUM   = 1         # number of learning passes  (10 for the Continuous Reacher task, 1 for Tennis)
GRAD_CLIPPING = 1.0     # Gradient Clipping                      - default 1

# Ornstein-Uhlenbeck noise parameters
OU_SIGMA  = 0.01        # 0.1 # default 0.2
OU_THETA  = 0.15        # default 0.15
# 
EPSILON       = 1.0     # for epsilon in the noise process (act step)
EPSILON_DECAY = 1e-6    # 1e-6    # decay rate (learn step), default 1e-6, 0 for no decay
# End - suggested on slack

USE_GPU = True

device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

beta_start = BETA
beta_frames = BETAFRAMES 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):  dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.epsilon = EPSILON

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, leakiness=LEAKINESS, kaiming=USEKAIMING).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, leakiness=LEAKINESS, kaiming=USEKAIMING).to(device)
        if OPTIM == "AdamW" and hasattr(optim,'AdamW'):
            self.actor_optimizer = optim.AdamW(self.actor_local.parameters(), lr=LR_ACTOR)
        else:    
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, leakiness=LEAKINESS, kaiming=USEKAIMING).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, leakiness=LEAKINESS, kaiming=USEKAIMING).to(device)
        if OPTIM == "AdamW" and hasattr(optim,'AdamW'):
            self.critic_optimizer = optim.AdamW(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=AWEIGHT_DECAY)
        else:    
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        if not PRIORITIZED:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        else:
            self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)    
    
    def step(self, states, actions, rewards, next_states, dones, timestep, num_agents):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        for i in range(num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        beta = beta_by_frame(timestep)

        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample(beta)
                self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.epsilon * self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, indices, weights) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        if not PRIORITIZED:
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        else:
            # Multiply loss by weights from Prioritized Replay Buffer
            loss = (Q_expected - Q_targets) ** 2
            loss = loss * weights.view(-1, 1)
            prios = loss + PRIOS_FACTOR
            critic_loss = torch.mean(loss)
             
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if PRIORITIZED:
            self.memory.update_priorities(indices, prios.data.cpu().numpy())

        # gradient clipping for critic
        if GRAD_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIPPING)
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)    

        # --------------------- and update epsilon decay ----------------------- # 
        if EPSILON_DECAY > 0:                
            self.epsilon -= EPSILON_DECAY
            self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        # dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory.
           beta is not used in standard ReplayBuffer
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # return 2 extra values as dummy replacements for the indices and weights returned by NaivePrioritizedBuffer
        indices = np.ones(len(actions))
        weights = torch.from_numpy(np.ones(len(actions))).float().to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        """ dummy function in standard buffer"""
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class NaivePrioritizedBuffer(object):
    """ 
    Prioritized Replay buffer to store experience tuples

    Adpated from https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
    """
    def __init__(self, buffer_size, batch_size, seed, prob_alpha=PROB_ALPHA):
        self.prob_alpha = prob_alpha
        self.batch_size = batch_size
        self.buffer_size   = buffer_size
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self, beta):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(np.array(actions)).float().to(device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(np.array(dones)).float().to(device)
        weights = torch.from_numpy(np.array(weights)).float().to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, batch_indices, batch_priorities):
        #import pdb; pdb.set_trace() 
        for idx, prio in zip(batch_indices, batch_priorities):
            # using np.mean(prio) experimentally, as otherwise we are trying
            # to push batch-size array into a single position in priorities
            # and are getting "ValueError: setting an array element with a sequence"
            self.priorities[idx] = np.mean(prio)

            # try max prio, as mean prio gets nothing (zero score after over 1,000 episodes)
            # self.priorities[idx] = np.max(prio)

    def __len__(self):
        return len(self.buffer)



