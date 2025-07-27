import sys
import os
sys.path.append(os.path.abspath('..'))

import torch as T
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from recurrent_network import RecurrentNetwork
from torch.distributions import Categorical
import gymnasium as gym

class SSMTrainer():
    def __init__(self, env, embedding_size, state_space_size, reward_threshold, reward_target, lr, gamma, entropy_coefficient, validation_length, dir, max_episode_time, device):

        '''
        env: 
        The name of the gymnasium environment. Should the same name as when building it manually
        
        embedding_size: 
        State space model embedding size determined by the user
        
        state_space_size: 
        Internal state space size of the hidden state
        
        reward_threshold: 
        The threshold to trigger model validation. mostly to save compute and skip verification of poor training performance.
        Lowering may allow the model to converge with less training steps at the cost of compute.

        reward_target:
        The target all validation rewards must achieve in order for the model to be considered 'passing'. Used as stopping criteria and to prevent flukes

        lr:
        learning rate

        gamma:
        controls how the model prioritizes long / short term rewards

        entropy_coefficient:
        helps exploration and helps model move out of local minima

        validation_length:
        controls how many additional episodes to play for validation

        dir:
        save directory. Currently unused, but will be used in the future to export model weights

        max_episode_time:
        controls how much to cap the live replay environment

        device:
        device to train on cpu or gpu.
        '''
        # creating gymnasium environments
        self.env = gym.make(env)
        self.render_env = gym.make(env, render_mode="human")
        self.render_env._max_episode_steps = max_episode_time

        self.device = device

        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        
        # vectorizing envs is out of my pay grade (which is nothing)
        self.batch_size = 1

        # initializing ssm
        self.agent = RecurrentNetwork(
            embedding_size=embedding_size,
            state_space_size=state_space_size,
            input_size=self.input_size,
            output_size=self.output_size,
            batch_size=self.batch_size,
            device=device,
            network_name=dir
        ).to(device)

        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr, weight_decay=1e-3)

        self.reward_threshold = reward_threshold
        self.reward_target = reward_target
        self.running_baseline = 0
        self.total_training_steps = 0
        self.episode = 0
        # initialize at -inf for envs that can have negative rewards
        #self.validation_reward = -float('inf')
        

        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.validation_length = validation_length

        # initializes dictionaries to store gradient and training information
        self.gradient_data = {name:[] for name, _ in self.agent.named_parameters()}
        self.training_data = {'training_steps':[], 'training_reward':[], 'validation_reward':[], 'loss':[], 'validation_episodes':[]}


    def play_episode(self, training=True, render_mode=False):
        # configs env for rendering if set
        if render_mode:
            env = self.render_env
        else:
            env = self.env

        obs, _ = env.reset()

        h_t = self.agent.get_inital_hidden_state()
        h_t = h_t.unsqueeze(0).expand(self.batch_size, -1, -1)

        done=False
        total_reward = 0
        training_steps = 0
        log_probs = []
        rewards = []
        entropies = []
        
        # looping through entire loop
        while not done:
            # changes the shape of the observation into (B, L, F), which is the allowable shape of the ssm model
            obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            h_t, y_t = self.agent(obs_tensor, h_t, embeddings_only=False)
            
            # calculating stochasitc actions for training
            if training:
                action, log_prob, entropy = self.__select_action_from_logits(y_t)
                log_probs.append(log_prob)
                entropies.append(entropy)
                training_steps += 1

            # deterministic actions for validation / replay
            else:
                logits = y_t.squeeze(0).squeeze(0)
                action = T.argmax(logits).item()

            obs, reward, terminated, truncated, _ = env.step(action)

            # appends time step information
            done = terminated or truncated
            rewards.append(reward)
            total_reward += reward

            if render_mode:
                env.render()
            
        if training:
            self.episode += 1
            self.total_training_steps += training_steps
            loss_vars = (log_probs, rewards, entropies)
            return loss_vars, total_reward
        
        else:
            return total_reward

    def train(self, verbose=True):
        '''
        verbose:
        controls if training loop will save gradients for debugging and model analysis
        '''       
        passed = False

        # main training loop
        while not passed:

            loss_vars, training_reward = self.play_episode(training=True)

            # calculates loss
            loss = self.__calculate_loss(*loss_vars)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            T.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

            # checks if the total reward is worthy of wasting compute to validate
            if training_reward >= self.reward_threshold:
                validation_reward, episodes_lasted, passed = self.__evaluate_policy()
                print(f"Episode {self.episode}: Reward = {training_reward} | Validation Avg = {validation_reward:.2f} from {episodes_lasted} episodes ")
            else:
                validation_reward = training_reward
                print(f"Episode {self.episode}: Reward = {training_reward}")

            # saves data to dictionaries
            if verbose:
                for name, param in self.agent.named_parameters():
                    self.gradient_data[name].append(param.grad.norm().item())

                self.training_data['training_steps'].append(self.total_training_steps)
                self.training_data['training_reward'].append(training_reward)
                self.training_data['validation_reward'].append(validation_reward)
                self.training_data['loss'].append(loss.item())
                self.training_data['validation_episodes'].append(episodes_lasted)

    def __select_action_from_logits(self, logits_3d):
        '''
        logits_3d (B, L, D) -> logits (D,)
        '''
        # stochastic for exploration
        logits = logits_3d.squeeze(0).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def __calculate_loss(self, log_probs, rewards, entropies):
        # standard policy gradient loss function

        # calculating returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = T.tensor(returns, dtype=T.float32, device=self.device)

        # action advantage calculation
        batch_mean = returns.mean().item()
        self.running_baseline = 0.95 * self.running_baseline + 0.05 * batch_mean
        advantages = returns - self.running_baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # calculating policy loss
        policy_loss = T.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
        entropy_bonus = T.stack(entropies).sum()
        loss = policy_loss - self.entropy_coefficient * entropy_bonus

        return loss

    def __evaluate_policy(self):
        # just runs an episode n # of times as specified during initialization. Uses argmax to get deterministic actions
        total_reward = 0
        for episode in range(self.validation_length):
            with T.no_grad():
                episode_reward = self.play_episode(training=False)
                total_reward += episode_reward

        # ends validation early if any of the episodes drop below the target reward
            if episode_reward < self.reward_target:
                return total_reward / (episode + 1), episode+1, False
            
        return total_reward / self.validation_length, episode+1, True

    def replay_live(self):
        # runs an episode live to demo training results
        self.agent.eval()

        with T.no_grad():
            total_reward = self.play_episode(training=False, render_mode=True)
        
        self.agent.train()
        self.render_env.close()
        print(f"Live Replay Episode Reward: {total_reward}")

    def compile_data(self):
        # converts the dictionaries made during the training into a dataframe for plotting and data analysis
        gradient_data = pd.DataFrame(self.gradient_data)
        training_data = pd.DataFrame(self.training_data)
        compiled_data = pd.concat([gradient_data, training_data], axis=1)
        return compiled_data


class ICM(nn.Module):
    def __init__(self, state_space_size, embedding_size, lr):
        super(ICM, self).__init__()

        # accepts input h_t and y_t to predict the next hidden state h_t+1'
        self.forward_curiosity = nn.Sequential(
            nn.Linear(state_space_size + embedding_size, state_space_size, dtype=T.complex64),
            nn.SiLU(),
            nn.Linear(state_space_size, state_space_size, dtype=T.complex64)
        )

        # accepts input h_t and h_t+1 to predict the output y_t' that was used to get here
        self.inverse_curiosity = nn.Sequential(
            nn.Linear(state_space_size*2, state_space_size, dtype=T.complex64),
            nn.SiLU(),
            # state_space_size -> state_space_size, since we will use the ssm's internal feature compressor
            nn.Linear(state_space_size, state_space_size, dtype=T.complex64)
        )

        self.optimizer = optim.AdamW(list(self.forward_curiosity.parameters()) + list(self.inverse_curiosity.parameters()), lr=lr, weight_decay=1e-3)
        self.mse_loss = nn.MSELoss()
    
    # compression_fn is an input variable since the internals change with gradient updates
    def forward(self, h_prev, h_t, y_prev, compression_fn):       
        # detaches inputs to prevent curiosity module from updating ssm parameters
        h_prev_d = h_prev.detach()
        h_t_d = h_t.detach()
        y_complex_d = T.complex(y_prev, T.zeros_like(y_prev)).detach()

        # computes the next hidden state
        h_next = self.forward_curiosity(T.concat(tensors=(h_prev_d, y_complex_d), dim=-1))

        # computes interpolates the action taken to current hidden state
        y_embed = self.inverse_curiosity(T.concat(tensors=(h_prev_d, h_t_d), dim=-1))
        y_pred = compression_fn(y_embed.detach())

        # computes losses
        exploration_signal = self.mse_loss(h_next, h_t_d)
        inverse_loss = self.mse_loss(y_pred, y_prev.detach())
        total_curiosity_loss = exploration_signal + inverse_loss

        return exploration_signal, total_curiosity_loss


