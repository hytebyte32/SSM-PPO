import sys
import os
sys.path.append(os.path.abspath('..'))

import torch as T
import torch.optim as optim
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
        
        # initializing ssm
        self.agent = RecurrentNetwork(
            embedding_size=embedding_size,
            state_space_size=state_space_size,
            input_size=self.env.observation_space.shape[0],
            output_size=self.env.action_space.n,
            batch_size=1,
            device=device,
            network_name=dir
        ).to(device)

        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr, weight_decay=1e-3)

        self.reward_threshold = reward_threshold
        self.reward_target = reward_target
        self.running_baseline = 0
        self.training_steps = 0
        

        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.validation_length = validation_length    

        
               
    
    def train(self, verbose=True):
        '''
        verbose:
        controls if training loop will save gradients for debugging and model analysis
        '''

        # initializes dictionaries to store gradient and training information
        self.gradient_data = {name:[] for name, _ in self.agent.named_parameters()}
        self.training_data = {'training_steps':[], 'training_reward':[], 'validation_reward':[], 'loss':[]}
        # reset + init incase training is run again without resarting
        self.running_baseline = 0
        training_steps = 0
        episode = 0
        avg_reward = 0

        # main training loop
        while avg_reward < self.reward_target:

            # reset environment for next episode
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            self.avg_reward = 0.0
            h_t = None

            episode += 1

            log_probs = []
            rewards = []
            entropies = []

            # looping through entire loop
            while not done:
                # changes the shape of the observation into (B, L, F), which is the allowable shape of the ssm model
                obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                h_t, y_t = self.agent(obs_tensor, h_t, embeddings_only=False)

                # calculating actions and loss variables
                action, log_prob, entropy = self.__select_action_from_logits(y_t)
                log_probs.append(log_prob)
                entropies.append(entropy)

                obs, reward, terminated, truncated, _ = self.env.step(action)

                # appends time step information
                done = terminated or truncated
                rewards.append(reward)
                total_reward += reward
                training_steps += 1

            # calculates loss
            loss = self.__learn_from_episode(log_probs, rewards, entropies)

            # checks if the total reward is worthy of wasting compute to validate
            if total_reward >= self.reward_threshold:
                avg_reward, episodes_lasted = self.__evaluate_policy()
                print(f"Episode {episode}: Reward = {total_reward} | Validation Avg = {avg_reward:.2f} from {episodes_lasted} episodes ")
            else:
                print(f"Episode {episode}: Reward = {total_reward}")

            
            # saves data to dictionaries
            if verbose:
                for name, param in self.agent.named_parameters():
                    self.gradient_data[name].append(param.grad.norm().item())

                self.training_data['training_steps'].append(training_steps)
                self.training_data['training_reward'].append(total_reward)
                self.training_data['validation_reward'].append(avg_reward)
                self.training_data['loss'].append(loss.item())

    def __select_action_from_logits(self, logits_3d):
        '''
        logits_3d (B, L, D) -> logits (D,)
        '''
        # stochastic for exploration
        logits = logits_3d.squeeze(0).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def __learn_from_episode(self, log_probs, rewards, entropies):
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

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        T.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()

        # loss is returned for data collection
        return loss

    def __evaluate_policy(self):
        # just runs an episode n # of times as specified during initialization. Uses argmax to get deterministic actions
        total_reward = 0
        for episode in range(self.validation_length):
            obs, _ = self.env.reset()
            done = False
            h_t = None
            while not done:
                # disables gradients since we dont want to train off the validation data
                with T.no_grad():
                    obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                    h_t, y_t = self.agent(obs_tensor, h_t, embeddings_only=False)
                    action = T.argmax(y_t.squeeze(0).squeeze(0)).item()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

        # ends validation early if any of the episodes drop below the target reward
            if total_reward / (episode + 1) < self.reward_target:
                return total_reward / (episode + 1), episode+1
        return total_reward / self.validation_length, episode+1

    def replay_live(self):
        # runs an episode live to demo training results
        obs, _ = self.render_env.reset()
        done = False
        h_t = None
        total_reward = 0
        step = 0

        self.agent.eval()

        while not done:
            obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            # disables gradients since this is a demo
            with T.no_grad():
                h_t, y_t = self.agent(obs_tensor, h_t, embeddings_only=False)
                logits = y_t.squeeze(0).squeeze(0)
                action = T.argmax(logits).item()

            obs, reward, terminated, truncated, _ = self.render_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            self.render_env.render()
        
        self.agent.train()
        self.render_env.close()
        print(f"Live Replay Episode Reward: {total_reward}")

    def compile_data(self):
        # converts the dictionaries made during the training into a dataframe for plotting and data analysis
        gradient_data = pd.DataFrame(self.gradient_data)
        training_data = pd.DataFrame(self.training_data)
        compiled_data = pd.concat([gradient_data, training_data], axis=1)
        return compiled_data

        