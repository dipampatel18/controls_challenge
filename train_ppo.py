from collections import namedtuple

import numpy as np
import pandas as pd
import gymnasium as gym
import onnxruntime as ort

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback

EPOCHS = 1000
ACC_G = 9.81
FPS = 10
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
# TODO: NEEDS TUNING!!
FUTURE_PLAN_STEPS = FPS * 2  # 2 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class TensorboardLoggingCallback(BaseCallback):
  """
  Custom callback for logging individual reward components to TensorBoard.
  """
  def __init__(self, verbose=0):
    super().__init__(verbose)

  def _on_step(self) -> bool:
    # self.locals['infos'] contains the info dicts from all 16 parallel environments
    infos = self.locals.get("infos", [])
    
    # Extract the custom metrics from the info dicts
    lataccel_costs = [info.get('lataccel_cost', 0) for info in infos]
    jerk_costs = [info.get('jerk_cost', 0) for info in infos]
    tanh_rewards = [info.get('tanh_reward', 0) for info in infos]
    
    # Log the mean of these metrics to TensorBoard
    # They will appear under a new "custom_metrics" tab
    self.logger.record('custom_metrics/lataccel_cost', np.mean(lataccel_costs))
    self.logger.record('custom_metrics/jerk_cost', np.mean(jerk_costs))
    self.logger.record('custom_metrics/tanh_reward', np.mean(tanh_rewards))
    
    return True
    

class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value):
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token):
    return self.bins[token]

  def clip(self, value):
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self):
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    provider = 'CPUExecutionProvider'
    with open("./models/tinyphysics.onnx", "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.):
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample

  def get_current_lataccel(self, sim_states, actions, past_preds):
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, temperature=0.8))


class TinyPhysicsEnv(gym.Env):
  """
  PPO controller
  """
  def __init__(self):
    super().__init__()
    # 'roll_lataccel', 'v_ego', 'a_ego', 'target_lataccel', 'current_lataccel', 'prev_action' (6)
    # 'roll_lataccel_history', 'v_ego_history', 'a_ego_history', 'target_lataccel_history' (FUTURE_PLAN_STEPS*4)
    # TODO: Determine the low and high
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6+(FUTURE_PLAN_STEPS*4),), dtype=np.float32)
    self.action_space = gym.spaces.Box(low=STEER_RANGE[0], high=STEER_RANGE[1], shape=(1,), dtype=np.float32)
    self.tinyphysicsmodel = TinyPhysicsModel()

    self.step_idx = CONTEXT_LENGTH
    self.prev_action = 0.0
    self.target_lataccel = 0.0
    self.current_lataccel = 0.0
    self.prev_lataccel = 0.0
    self.current_lataccel_history = None

    self.N = 20000
    self.indices = np.random.permutation(self.N)
    self.ptr = 0

  def get_data(self, data_path: str):
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
    })
    return processed_df

  def compute_rewards(self):
    accel_error = self.target_lataccel - self.current_lataccel
    abs_error = abs(accel_error)
    
    lat_accel_cost = ((accel_error)**2) * 100
    
    jerk_cost = (((self.current_lataccel - self.prev_lataccel) / DEL_T)**2) * 100
    self.prev_lataccel = self.current_lataccel

    tanh_reward = (1 - np.tanh(abs_error / 0.3))

    return {
            'lataccel_cost': lat_accel_cost,
            'jerk_cost': jerk_cost,
            'tanh_reward': tanh_reward
            }

  def get_state_target_futureplan(self, step_idx):
    state = self.data.iloc[step_idx]
    start_idx = self.step_idx + 1
    end_idx = start_idx + FUTURE_PLAN_STEPS
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=self.data['target_lataccel'].values[start_idx:end_idx].tolist(),
        roll_lataccel=self.data['roll_lataccel'].values[start_idx:end_idx].tolist(),
        v_ego=self.data['v_ego'].values[start_idx:end_idx].tolist(),
        a_ego=self.data['a_ego'].values[start_idx:end_idx].tolist()
      )
    )

  def get_observation(self):
    state = self.data.iloc[self.step_idx]

    current_state = [
      state['roll_lataccel'],
      state['v_ego'],
      state['a_ego'],
      state['target_lataccel'],
      self.current_lataccel,
      self.prev_action  # TODO: Determine if just last prev_action or history of prev_action should be used
    ]

    start_idx = self.step_idx + 1
    end_idx = start_idx + FUTURE_PLAN_STEPS
    future_states = [
      self.data['roll_lataccel'].values[start_idx:end_idx],
      self.data['v_ego'].values[start_idx:end_idx],
      self.data['a_ego'].values[start_idx:end_idx],
      self.data['target_lataccel'].values[start_idx:end_idx]
    ]
    future_states = np.concatenate(future_states)

    return np.concatenate([current_state, future_states]).astype(np.float32)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    # Reset the TinyPhysicsSimulator to a random CSV file from data_path
    if self.ptr >= self.N:
      self.indices = np.random.permutation(self.N)
      self.ptr = 0

    idx = self.indices[self.ptr]
    self.ptr += 1
    # data_path = f"data/{idx:05d}.csv"
    data_path = f"data/00000.csv"
    print(f"[INFO] Getting Data File: {data_path}")
    self.data = self.get_data(data_path)

    # Fast-forward the simulator to CONTEXT_LENGTH
    self.step_idx = CONTEXT_LENGTH
    state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_history = [x[0] for x in state_target_futureplans]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [x[1] for x in state_target_futureplans]
    
    self.current_lataccel = self.current_lataccel_history[-1]
    self.prev_lataccel = self.current_lataccel
    self.prev_action = 0.0

    observation = self.get_observation()
    info = {}

    return observation, info

  def step(self, action):
    # TODO: Pass the action to the simulator
    # TAKE ACTION!!
    action = float(np.clip(action[0], STEER_RANGE[0], STEER_RANGE[1]))

    state, _, _ = self.get_state_target_futureplan(self.step_idx)
    self.state_history.append(state)
    self.action_history.append(action)
    self.prev_action = action

    # TODO: Advance the simulator by one step
    pred = self.tinyphysicsmodel.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )

    self.current_lataccel = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    self.current_lataccel_history.append(self.current_lataccel)

    state = self.data.iloc[self.step_idx]
    self.target_lataccel = state['target_lataccel']

    # Calculate the step reward (negative cost)
    rewards_info = self.compute_rewards()
    accel_cost = (rewards_info['lataccel_cost'] * LAT_ACCEL_COST_MULTIPLIER) + rewards_info['jerk_cost']
    reward = float((rewards_info['tanh_reward'] * LAT_ACCEL_COST_MULTIPLIER) - accel_cost)
    
    self.step_idx += 1

    # Check if we reached COST_END_IDX (done flag)
    terminated = False
    truncated = False
    if self.step_idx == COST_END_IDX - FUTURE_PLAN_STEPS:
      truncated = True

    observation = self.get_observation()
    info = rewards_info

    return observation, reward, terminated, truncated, info


def make_env():
  """Utility function for multiprocessed env."""
  def _init():
    env = TinyPhysicsEnv()
    return env
  return _init


if __name__ == "__main__":
  # 1. Vectorize the Environment 
  # This spins up 16 parallel environments. Perfect for Slurm node parallelization.
  num_cpu = 4
  # 1. Training Environment
  vec_env = SubprocVecEnv([make_env() for i in range(num_cpu)])
  vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  # 2. Evaluation Environment (Runs completely independently)
  # We use fewer CPUs here, and importantly, we DO NOT normalize the rewards 
  # during evaluation so we get the true leaderboard-style score.
  eval_env = SubprocVecEnv([make_env() for i in range(4)])
  eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

  # 3. Define the PPO Architecture
  policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
  model = PPO(
    "MlpPolicy", vec_env, learning_rate=3e-4, n_steps=1024, batch_size=256,
    policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_tinyphysics_tensorboard/"
  )

  # 4. Setup Callbacks
  checkpoint_callback = CheckpointCallback(
    save_freq=50000, save_path='./models/ppo_checkpoints/', name_prefix='ppo_steer'
  )
  logging_callback = TensorboardLoggingCallback()
  
  # The EvalCallback will automatically sync the normalization statistics 
  # from the training env to the eval env before testing.
  eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./models/ppo_best/',
    log_path='./ppo_tinyphysics_tensorboard/eval/',
    eval_freq=10000, # Evaluate every 10,000 steps
    deterministic=True, 
    render=False
  )

  # 5. Train the Model (Now with 3 callbacks!)
  print("Starting Training...")
  try:
    model.learn(
      total_timesteps=5_000_000, 
      callback=[checkpoint_callback, logging_callback, eval_callback] 
    )
  except KeyboardInterrupt:
    print("\n[INFO] Training interrupted by user. Saving current state...")
  finally:
    # 6. Save BOTH the model and the normalization statistics
    # This will now run even if you hit Ctrl+C!
    model.save("ppo_tinyphysics_final")
    vec_env.save("vec_normalize.pkl") 
    print("Model and Normalization Stats Saved Successfully!")

# Observations:
# - roll_lataccel,
# - v_ego,
# - a_ego,
# - target_lataccel
# - current_lataccel

# Actions:
# - steer_command