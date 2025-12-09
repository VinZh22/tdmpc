import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()

	episode_rewards = []
	episode_env_steps = []

	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		episode_rewards.append(episode.cumulative_reward)
		episode_env_steps.append(env_step)
	


		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')

		# plot loses from agent with log y-axis
	plt.figure(figsize=(12, 8))
	for key, values in agent.loss_history.items():
		plt.plot(values, label=key)
	plt.xlabel('Training Steps')
	plt.yscale('log')
	plt.ylabel('Loss')
	plt.title('Training Losses Over Time')
	plt.legend()
	# save the plot
	plt_path = work_dir / 'training_losses.png'
	plt.savefig(plt_path)
	print(f'Training loss plot saved to {plt_path}')

	# plot training returns over time
	plt.figure(figsize=(8, 5))
	plt.plot(episode_env_steps, episode_rewards)
	plt.xlabel('Environment steps')
	plt.ylabel('Episode return')
	plt.title('Training Return Over Time')
	plt.grid(True)
	ret_path = work_dir / 'training_returns.png'
	plt.savefig(ret_path)
	print(f'Training return plot saved to {ret_path}')


if __name__ == '__main__':
	cfg = parse_cfg(Path().cwd() / __CONFIG__)
	train(cfg)
