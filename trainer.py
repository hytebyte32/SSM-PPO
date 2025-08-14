import torch as T
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from recurrent_network import RecurrentNetwork
from torch.distributions import Categorical
import gymnasium as gym
from copy import deepcopy

class SSMTrainer():
    def __init__(self, env, embedding_size, state_space_size, reward_threshold, reward_target, lr, cf, gamma, entropy_coefficient, validation_length, dir, max_episode_time, strict_mode, device):

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

        cf:
        curiosity factor. controls how important it is to explore the environment

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

        self.strict_mode = strict_mode

        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

        self.lr = lr
        self.modulated_lr = -np.inf
        self.patience = 0
        self.cf = cf
        
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
        
        # initializing curiosity module
        self.curiosity_module = ICM(
            state_space_size=state_space_size,
            embedding_size=embedding_size,
            lr=lr,
            device=device
        )

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
        self.ssm_gradient_data = {name:[] for name, _ in self.agent.named_parameters()}
        self.curiosity_gradient_data = {name:[] for name, _ in self.curiosity_module.named_parameters()}
        self.training_data = {'training_steps':[], 'training_reward':[], 'validation_reward':[], 'loss':[], 'validation_episodes':[], 'curiosity_loss':[], 'exploration_signal_bonus':[]}


    def play_episode(self, training=True, render_mode=False, verbose=True):
        # configs env for rendering if set
        if render_mode:
            env = self.render_env
        else:
            env = self.env

        done=False
        total_reward = 0
        training_steps = 0
        log_probs = []
        rewards = []
        entropies = []     

        obs, _ = env.reset()

        h_t = self.agent.get_inital_hidden_state()
        h_t = h_t.unsqueeze(0).expand(self.batch_size, -1, -1)
        h_prev = []
        y_prev = []
        y_embedded = None
        
        # looping through entire loop
        while not done:
            # changes the shape of the observation into (B, L, F), which is the allowable shape of the ssm model
            obs_tensor = T.tensor(obs, dtype=T.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            y_embedded, h_t = self.agent(obs_tensor, h_t)
            y_t = self.agent.pred(y_embedded)
            
            # calculating stochasitc actions for training
            if training:
                
                h_prev.append(h_t)
                y_prev.append(y_embedded)
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
            
            # calculate curiosity parameters at t=1 since y_prev does not exist at t=0
            output_fns = (self.agent.recurrent_block.post_process_output, deepcopy(self.agent.recurrent_block.output_compression))
            total_exploration_signal, total_curiosity_loss = self.curiosity_module(h_prev, y_prev, output_fns)                    

            loss_vars = (log_probs, rewards, entropies, total_exploration_signal, total_curiosity_loss)
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
            total_exploration_signal = loss_vars[3]

            # calculates loss
            model_loss, total_curiosity_loss = self.__calculate_loss(*loss_vars)

            # backpropagation
            self.__learn(model_loss, total_curiosity_loss)

            # checks if the total reward is worthy of wasting compute to validate
            if training_reward >= self.reward_threshold:
                validation_reward, episodes_lasted, passed = self.__evaluate_policy()
                print(f"Episode {self.episode}: Reward = {training_reward} | Validation Avg = {validation_reward:.2f} from {episodes_lasted} episodes ")
                self.modulate_lr(episodes_lasted)
            else:
                validation_reward = training_reward
                episodes_lasted = 0
                print(f"Episode {self.episode}: Reward = {training_reward}")

            # saves data to dictionaries
            if verbose:
                for name, param in self.agent.named_parameters():
                    self.ssm_gradient_data[name].append(param.grad.norm().item())
                    
                for name, param in self.curiosity_module.named_parameters():
                    if param.grad is not None: self.curiosity_gradient_data[name].append(param.grad.norm().item())
                    else: self.curiosity_gradient_data[name].append(None)

                self.training_data['training_steps'].append(self.total_training_steps)
                self.training_data['training_reward'].append(training_reward)
                self.training_data['validation_reward'].append(validation_reward)
                self.training_data['validation_episodes'].append(episodes_lasted)
                self.training_data['loss'].append(model_loss.item())
                self.training_data['curiosity_loss'].append(total_curiosity_loss.item())
                self.training_data['exploration_signal_bonus'].append(T.mean(total_exploration_signal).item())
    def __select_action_from_logits(self, logits_3d):
        '''
        logits_3d (B, L, D) -> logits (D,)
        '''
        # stochastic for exploration
        logits = logits_3d.squeeze(0).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def __calculate_loss(self, log_probs, rewards, entropies, exploration_signals, total_curiosity_loss):
        # standard policy gradient loss function

        # calculating returns
        returns = []
        G = 0
        for reward, exploration_signal in zip(reversed(rewards), reversed(self.normalize_exploration_bonus(exploration_signals))):
            G = reward + exploration_signal * self.cf + self.gamma * G
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
        model_loss = policy_loss - self.entropy_coefficient * entropy_bonus
        curiosity_loss = total_curiosity_loss.sum()
        
        return model_loss, curiosity_loss

    def __evaluate_policy(self):
        # just runs an episode n # of times as specified during initialization. Uses argmax to get deterministic actions
        total_reward = 0
        for episode in range(self.validation_length):
            with T.no_grad():
                episode_reward = self.play_episode(training=False)
                total_reward += episode_reward

            avg_reward = total_reward / (episode + 1)

            # ends validation early if any of the episodes drop below the target reward (strict)
            if episode_reward < self.reward_target and self.strict_mode:
                return avg_reward, episode+1, False
            '''
            I cant think of a clean way to calculate if the avg will be above the target given the remaining episodes
            without hardcoding max reward. I want to use the same class for other envs aswell, so going to run with avg cutoff for now.
            '''
            # less strict validation ends early if the avg is below the target reward
            if avg_reward < self.reward_target and not self.strict_mode:
                return avg_reward, episode+1, False

        return avg_reward, self.validation_length, True
    
    def __learn(self, loss, curiosity_loss):
        self.optimizer.zero_grad()
        self.curiosity_module.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        curiosity_loss.backward()
        T.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.step()

    # makes learning rate dependent on validation episode. Useful for fine-tuning when close to solving
    def modulate_lr(self, validation_episode):
        
        prev_lr = self.modulated_lr

        # will put in constructor later
        patience_threshold = 3

        ### peicewise fn controlling lr decay
        m1 = 0.5
        m2 = 3
        b = 105
        
        offset_episode = validation_episode - m2

        if offset_episode < 0:
            modulator = (m1 / m2) * validation_episode + m1 + 2
        else:
            modulator = -(m1/b) * (offset_episode - b)

        clipped_modulator = np.clip(modulator, min=0.1, max=1)

        self.modulated_lr = clipped_modulator * self.lr

        if prev_lr < self.modulated_lr and self.modulated_lr != 1:
            self.patience += 1
            return

        if self.patience == patience_threshold:
            self.patience = 0
            for group in self.optimizer.param_groups:
                group['lr'] = self.modulated_lr

    def normalize_exploration_bonus(self, exploration_bonus):
        norm = nn.LayerNorm(len(exploration_bonus), elementwise_affine=False)
        return norm(exploration_bonus)
    
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
        ssm_gradient_data = pd.DataFrame(self.ssm_gradient_data)
        curiosity_gradient_data = pd.DataFrame(self.curiosity_gradient_data)
        training_data = pd.DataFrame(self.training_data)
        compiled_data = pd.concat([ssm_gradient_data, curiosity_gradient_data, training_data], axis=1)
        return compiled_data


class ICM(nn.Module):
    '''
    ICM (Intrinsic Curiosity Module) is responsible for punishing the model if the next state is very predictable,
    which likely means that the model is stuck in some sort of local minima. By punishing predictable next states, the model
    will be forced to explore its environment and hopefully come up with a more optimal solution. 
    '''
    def __init__(self, state_space_size, embedding_size, lr, device):
        super(ICM, self).__init__()

        '''
        rather than taking raw state inputs/outputs, chose to use and predict ssm hidden states and embeddings since temporal information is intrinsically modeled,
        potentially improving representation
        '''
        # accepts input h_t and y_t to predict the next hidden state h_t+1'
        # since h_t is 3d and y_t is 2d, we need to project y_t into 3rd dimension, hence the +1
        self.forward_curiosity = nn.Sequential(
            nn.Linear(state_space_size + 1, state_space_size, dtype=T.complex64),
            nn.Tanh(),
            nn.Linear(state_space_size, state_space_size, dtype=T.complex64)
        )

        # accepts input h_t and h_t+1 to interpolate the output y_t' that was used to get here
        # since output is a prediction of y_t (2d) given h_t and h_t+1 (3d), we need to squeeze back down to a 2d structure later, hence the output size of 1
        self.inverse_curiosity = nn.Sequential(
            nn.Linear(state_space_size*2, embedding_size, dtype=T.complex64),
            nn.Tanh(),
            nn.Linear(embedding_size, 1, dtype=T.complex64)
        )

        self.optimizer = optim.AdamW(list(self.forward_curiosity.parameters()) + list(self.inverse_curiosity.parameters()), lr=lr, weight_decay=1e-3)
        self.mse_loss = nn.MSELoss()

        self.device = device
        self.to(device)
    
    # compression_fn is an input variable since the internal weights change with gradient updates
    def forward(self, h_prev, y_prev, output_fns):
        # detaches inputs to prevent curiosity module from updating ssm parameters

        post_processing_fn, compression_fn = output_fns

        h_true = h_prev[1:]
        y_true = y_prev[1:]

        h_prev = h_prev[:-1]
        y_prev = y_prev[:-1]

        

        total_exploration_signal = []
        total_curiosity_loss = []
        for y_prev_t, y_true_t, h_prev_t, h_true_t in zip(y_prev, y_true, h_prev, h_true):
            y_prev_t = T.complex(y_prev_t, T.zeros_like(y_prev_t)).squeeze(1).unsqueeze(-1)

            # computes the exploration signal
            h_pred_t = self.forward_curiosity(T.concat(tensors=(h_prev_t, y_prev_t), dim=-1))
            real_exploration_signal = self.mse_loss(T.real(h_pred_t), T.real(h_true_t))
            img_exploration_signal = self.mse_loss(T.imag(h_pred_t), T.imag(h_true_t))
            exploration_signal = real_exploration_signal + img_exploration_signal
            total_exploration_signal.append(exploration_signal)

            # computes the curiosity loss
            y_embed_t = self.inverse_curiosity(T.concat(tensors=(h_prev_t, h_true_t), dim=-1)).squeeze(-1)
            y_pred_t = compression_fn(*post_processing_fn(y_embed_t))
            inverse_loss = self.mse_loss(y_pred_t, y_true_t)
            total_curiosity_loss.append(exploration_signal + inverse_loss)

        total_exploration_signal = T.stack(tensors=total_exploration_signal)
        total_curiosity_loss = T.stack(tensors=total_curiosity_loss)

        '''
        exploration signal helps trains ssm model
        curiosity loss trains the ICM network
        '''
        return total_exploration_signal, total_curiosity_loss

