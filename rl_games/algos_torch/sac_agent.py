from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from rl_games.common.a2c_common import print_statistics

from rl_games.interfaces.base_algorithm import  BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from rl_games.algos_torch import  model_builder
from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os


class SACAgent(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=float(self.config["critic_lr"]), betas=self.config.get("critic_betas", [0.9, 0.999]))
            for critic in self.model.sac_network.critics
        ]

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
        self.env_info['action_space'].shape,
        self.replay_buffer_size,
        self._device)
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

        self.algo_observer = config['features']['observer']

        self.update_step = 0  # Initialize the update counter
        self.actor_loss_info = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        print("[DEBUG] Loaded network from ModelBuilder.")

        # Load REDQ-specific parameters
        self.num_critics = self.config.get('num_critics', 10)
        self.m = self.config.get('critic_subsample_size', 2)
        self.use_layer_norm = self.config.get('use_layer_norm', True)
        self.use_dropout = self.config.get('use_dropout', True)
        self.dropout_prob = self.config.get('dropout_prob', 0.01)
        self.policy_delay = self.config.get("policy_delay", 1)
        self.gradient_steps = self.config.get('gradient_steps', 1)  # Default of 1 gradient step per env step
        self.policy_delay_offset = self.config.get('policy_delay_offset', 0)
        self.q_target_mode = self.config.get('q_target_mode', 'ave')  # Add this line


        print(f"[DEBUG] REDQ Config - num_critics: {self.num_critics}, "
            f"critic_subsample_size: {self.m}, "
            f"use_layer_norm: {self.use_layer_norm}, "
            f"use_dropout: {self.use_dropout}, "
            f"dropout_prob: {self.dropout_prob}")

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

        self.log_freq = config.get('log_frequency', 1)

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'actor': self.model.sac_network.actor.state_dict(),
         'critic': self.model.sac_network.critic.state_dict(), 
         'critic_target': self.model.sac_network.critic_target.state_dict()}
        if self.normalize_input:
            state['running_mean_std'] = self.model.running_mean_std.state_dict()
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        # Store all individual critic optimizer states
        state['critic_optimizers'] = [opt.state_dict() for opt in self.critic_optimizers]
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()        

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        # Load each critic optimizer state individually
        for opt, opt_state in zip(self.critic_optimizers, weights['critic_optimizers']):
            opt.load_state_dict(opt_state)
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("SAC restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()
        # # Ensure Dropout is active during evaluation
        # def apply_dropout(m):
        #     if isinstance(m, nn.Dropout):
        #         m.train()
        # self.model.apply(apply_dropout)

    def set_train(self):
        self.model.train()


    def update_critic(self, obs, action, reward, next_obs, not_done):
        # print("[DEBUG] Starting critic update.")
        
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            
            # Print next action and log probability
            # print(f"[DEBUG] Next Action: {next_action}, Log Probability: {log_prob}")

            # Get target Q-values from the ensemble of target critics
            target_Q_values = self.model.sac_network.critic_target(next_obs, next_action)
            # print(f"[DEBUG] Target Q Values from all critics: {target_Q_values}")

            if self.q_target_mode == 'min':
                # Randomly sample 'm' critics and take the minimum
                idxs = np.random.choice(self.num_critics, self.m, replace=False)
                target_Q_sampled = target_Q_values[idxs, :]
                target_Q_min = target_Q_sampled.min(dim=0)[0].unsqueeze(-1)
                target_V = target_Q_min - self.alpha * log_prob

            elif self.q_target_mode == 'ave':
                # Average over all critics
                target_Q_mean = target_Q_values.mean(dim=0).unsqueeze(-1)
                target_V = target_Q_mean - self.alpha * log_prob

            elif self.q_target_mode == 'rem':
                # Random Ensemble Mixture
                weights = torch.rand(self.num_critics, 1, device=self.device)
                weights = weights / weights.sum(0)
                target_Q_weighted = (weights * target_Q_values).sum(dim=0, keepdim=True)
                target_V = target_Q_weighted - self.alpha * log_prob

            else:
                raise ValueError(f"Unknown q_target_mode: {self.q_target_mode}")

            target_Q = reward + (not_done * self.gamma * target_V)
            target_Q = target_Q.detach()
            # print(f"[DEBUG] Target Q: {target_Q}")

        # Get current Q estimates from all critics
        current_Q_values = self.model.sac_network.critic(obs, action)
        # print(f"[DEBUG] Current Q Values from all critics: {current_Q_values}")

        # Compute critic loss over all critics
        critic_losses = []
        critic_loss = 0
        for i in range(self.num_critics):
            current_Q = current_Q_values[i, :].unsqueeze(-1)
            loss = self.c_loss(current_Q, target_Q)
            critic_losses.append(loss.detach())
            critic_loss += loss

            # Update the critic using its specific optimizer
            self.critic_optimizers[i].zero_grad(set_to_none=True)
            loss.backward()
            self.critic_optimizers[i].step()

        critic_loss = critic_loss / self.num_critics

        # print(f"[DEBUG] Critic Loss: {critic_loss}")

        return critic_loss.detach(), critic_losses


    def update_actor_and_alpha(self, obs):
        # print("[DEBUG] Starting actor and alpha update.")
        
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        dist = self.model.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = -log_prob.mean()
        
        # Print sampled action and log probability
        # print(f"[DEBUG] Action: {action}, Log Probability: {log_prob}, Entropy: {entropy}")

        # Get Q-values from all critics
        actor_Qs = self.model.sac_network.critic(obs, action)
        # print(f"[DEBUG] Actor Qs from all critics: {actor_Qs}")

        # Log Q-values from all critics
        for idx in range(self.num_critics):
            q_values = actor_Qs[idx, :].mean().item()
            self.writer.add_scalar(f'values/actor_Q_critic_{idx}', q_values, self.frame)

        actor_Q = torch.min(actor_Qs, dim=0)[0].unsqueeze(-1)
        # print(f"[DEBUG] Actor Q (min over critics): {actor_Q}")

        actor_loss = (torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q)
        actor_loss = actor_loss.mean()
        # print(f"[DEBUG] Actor Loss: {actor_loss}")

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            # print(f"[DEBUG] Alpha Loss: {alpha_loss}")

            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        # print(f"[DEBUG] Alpha: {self.alpha.detach()}")
        
        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss


    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self):
        total_batch_size = min(self.batch_size * self.gradient_steps, len(self.replay_buffer))
        obs, action, reward, next_obs, done = self.replay_buffer.sample(total_batch_size)

        # Ensure total_batch_size is divisible by gradient_steps
        if total_batch_size % self.gradient_steps != 0:
            print(f"Total batch size {total_batch_size} is not divisible by gradient_steps {self.gradient_steps}.")

        obs_batches = obs.view(self.gradient_steps, self.batch_size, *obs.shape[1:])
        action_batches = action.view(self.gradient_steps, self.batch_size, *action.shape[1:])
        reward_batches = reward.view(self.gradient_steps, self.batch_size, *reward.shape[1:])
        next_obs_batches = next_obs.view(self.gradient_steps, self.batch_size, *next_obs.shape[1:])
        done_batches = done.view(self.gradient_steps, self.batch_size, *done.shape[1:])

        for i in range(self.gradient_steps):
            obs_i = obs_batches[i]
            action_i = action_batches[i]
            reward_i = reward_batches[i]
            next_obs_i = next_obs_batches[i]
            done_i = done_batches[i]
            not_done_i = ~done_i

            obs_i = self.preproc_obs(obs_i)
            next_obs_i = self.preproc_obs(next_obs_i)

            # Update critic
            critic_loss, critic_losses = self.update_critic(obs_i, action_i, reward_i, next_obs_i, not_done_i)

            # Update actor and alpha if needed
            if (self.update_step + self.policy_delay_offset)% self.policy_delay == 0:
                actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs_i)
                self.actor_loss_info = actor_loss, entropy, alpha, alpha_loss

            # Soft-update target networks
            self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target, self.critic_tau)

            self.update_step += 1  # Increment the update counter

        return self.actor_loss_info, critic_loss, critic_losses


    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = self.model.norm_obs(obs)

        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device), torch.from_numpy(rewards).to(self._device), torch.from_numpy(dones).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)

        actions = dist.sample() if sample else dist.mean
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2

        return actions

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_losses is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration = False):
        total_time_start = time.perf_counter()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic_main_losses = []
        critic_losses = [[] for _ in range(self.num_critics)]

        obs = self.obs
        if isinstance(obs, dict):
            obs = self.obs['obs']

        next_obs_processed = obs.clone()

        for s in range(self.num_steps_per_episode):
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.perf_counter()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            step_end = time.perf_counter()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(next_obs, dict):    
                next_obs_processed = next_obs['obs']
                self.obs = next_obs_processed.clone()
            else:
                self.obs = next_obs.clone()

            rewards = self.rewards_shaper(rewards)
            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs_processed, torch.unsqueeze(dones, 1))

            if isinstance(obs, dict):
                obs = self.obs['obs']

            if not random_exploration:
                self.set_train()

                update_time_start = time.perf_counter()
                actor_loss_info, critic_loss, individual_critic_losses = self.update()
                update_time_end = time.perf_counter()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic_main_losses.append(critic_loss)

                # Collect individual critic losses
                for i in range(self.num_critics):
                    critic_losses[i].append(individual_critic_losses[i])
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.perf_counter()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_losses, entropies, alphas, alpha_losses, critic_main_losses, critic_losses

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        total_time = 0
        # rep_count = 0

        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, critic_loss, critic_losses = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                self.epoch_num, self.max_epochs, self.frame, self.max_frames)
            
            if self.epoch_num % self.log_freq == 0:
                self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
                self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
                self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
                self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
                self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
                self.writer.add_scalar('performance/step_time', step_time, self.frame)

                if self.epoch_num >= self.num_warmup_steps:
                    self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(actor_losses).item(), self.frame)
                    self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(critic_loss).item(), self.frame)

                    # Log individual critic losses
                    for i in range(self.num_critics):
                        if critic_losses[i]:
                            mean_loss = torch_ext.mean_list(critic_losses[i]).item()
                            self.writer.add_scalar(f'losses/critic_{i}_loss', mean_loss, self.frame)
                    self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), self.frame)

                    if alpha_losses[0] is not None:
                        self.writer.add_scalar('losses/alpha_loss', torch_ext.mean_list(alpha_losses).item(), self.frame)
                    self.writer.add_scalar('info/alpha', torch_ext.mean_list(alphas).item(), self.frame)

                self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()

                self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

                should_exit = False

                if self.save_freq > 0:
                    if self.epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, self.config['name']))
                    if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

                if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

                if should_exit:
                    return self.last_mean_rewards, self.epoch_num
