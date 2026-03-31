"""
Reinforcement Learning algorithms from stable_baseline3 to mitigate the consecutive outages.
To train the algorithm need to type in the command line: python main_sb3_v2.py DDPG
To test the algorithm need to type in the command line: python main_sb3_v2.py DDPG --test
"""

import os.path
import pickle
import json
import argparse
import numpy as np
import scipy.io
from environment_v1 import PowerControlEnv

import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Load the interference signal (actual and estimated values) and channel gains from a matlab file
def import_data_set(file_name):
    loaded_object = scipy.io.loadmat(file_name)
    desiredChannel = loaded_object['desiredChannel']
    sum_interference = loaded_object['sum_interference']
    est_interference = loaded_object['est_interference']
    return desiredChannel, sum_interference, est_interference

# Import data set
desiredChannel, sum_interference, est_interference = import_data_set('matlabfile.mat')

# Where to store trained models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():
    # Create the power control environment
    env = PowerControlEnv(
                max_blocklength=int(1e3),
                max_snr_dB=20.0,
                tgtOutageProb=1e-5,
                packetLength=50,
                consErrorThr=5,
                errorRateThr=0.01,
                channel_gains=desiredChannel,
                sumInterference=sum_interference,
                estimatedInterf=est_interference,
                w1=0.7, w2=0.3,
                )
    env = Monitor(env)

    # Build a model (Use MlpPolicy for observation space 1D vector and MultiInputPolicy for Dict observations)
    model = func_algorithm('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    # Stop training when mean reward reaches reward_threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)

    """
    Stop training when model shows no improvement after max_no_improvement_evals evaluations, 
    but do not start counting towards max_no_improvement_evals until after min_evals evaluations.
    Number of timesteps before possibly stopping training = min_evals * eval_freq (below)
    """
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=100, verbose=1)

    eval_callback = EvalCallback(
        env, 
        # eval_freq=1000, # How often to perform evaluation i.e. every eval_freq timesteps.
        n_eval_episodes=5, # The number of episodes to test the agent
        callback_on_new_best=callback_on_best, 
        callback_after_eval=stop_train_callback, 
        verbose=1, 
        best_model_save_path=os.path.join(model_dir, f"{args.sb3_algo}"),
    )
    
    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    tb_log_name: create log files with the name [sb3 algorithm]
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e10), tb_log_name=f"{args.sb3_algo}", callback=eval_callback)

def test():
    # Create the power control environment
    env = PowerControlEnv(
                max_blocklength=int(1e3),
                max_snr_dB=20.0,
                tgtOutageProb=1e-5,
                packetLength=50,
                consErrorThr=5,
                errorRateThr=0.01,
                channel_gains=desiredChannel[int(1e7):],
                sumInterference=sum_interference[int(1e7):],
                estimatedInterf=est_interference[int(1e7):],
                w1=0.7, w2=0.3,
                )
    # env = Monitor(env)
    
    model = func_algorithm.load(os.path.join(model_dir, f"{args.sb3_algo}", "best_model"), env=env)

    mean_rewards_per_episode = []
    mean_energy_per_episode = []
    error_density_per_episode = []
    consError_per_episode = []
    outageProb_per_episode = []

    # Evaluate the model by computing actions
    for episodes in range(int(1e2)):
        obs, info = env.reset()

        total_reward = 0
        total_energy = 0
        error = 0
        error_vector = []
        consError_vector = []
        outageProb_vector = []
        
        terminated = False    # True when runs > test dataset's length
        truncated = False     # True when actions > number of steps per episode (max_time=1000)
        while (not terminated and not truncated):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            # Accumulate the consumed energy
            consumed_energy = info["consumed_energy"].item()
            total_energy += consumed_energy

            # Extract the error vector including {0,1}
            outageProb = info["outageProb"].item()
            outageProb_vector.append(outageProb)
            error_vector.append(int(outageProb > env.tgtOutageProb))

            # Weights of reward elements
            reward_weights = info["reward_weights"]

            # Construct the consecutive error vector
            if outageProb > env.tgtOutageProb:
                error += 1
            else:
                consError_vector.append(error)
                error = 0

        mean_rewards_per_episode.append(total_reward/1000)
        mean_energy_per_episode.append(total_energy/1000)

        # Calculating the error rate
        error_density = np.mean(error_vector)
        error_density_per_episode.append(error_density)
        outageProb_per_episode.append(outageProb_vector)
        consError_per_episode.append(consError_vector)

    # Writing Pickle data
    data = {
        "reward_weights":reward_weights,
        "mean_rewards_per_episode":mean_rewards_per_episode,
        "mean_energy_per_episode":mean_energy_per_episode,
        "error_density_per_episode":error_density_per_episode,
        "consError_per_episode":consError_per_episode,
        "outageProb_per_episode":outageProb_per_episode,
        }
    with open(f'{args.sb3_algo}.pkl', 'wb') as file:
        pickle.dump(data, file)

    # Close the environment
    env.close()

if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. DDPG, TD3, A2C, SAC, PPO, DQN')
    parser.add_argument('--test', help='Test mode', action='store_true')
    args = parser.parse_args()

    func_algorithm = getattr(stable_baselines3, args.sb3_algo)

    if args.test:
        test()
    else:
        train()
