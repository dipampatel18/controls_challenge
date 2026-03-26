import os
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from . import BaseController

# --- GLOBAL CACHE ---
# Prevents the multiprocessing workers from locking up your disk 
# by unzipping the model file 100 times concurrently.
_CACHED_MODEL = None
_CACHED_NORMS = None

class Controller(BaseController):
    def __init__(self):
        global _CACHED_MODEL, _CACHED_NORMS
        
        # Prevent PyTorch from thread-locking during process_map
        torch.set_num_threads(1)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Only load from disk if this specific worker hasn't cached it yet
        if _CACHED_MODEL is None:
            model_path = os.path.join(base_dir, "models/ppo_best/best_model.zip")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "ppo_tinyphysics_final.zip")
            _CACHED_MODEL = PPO.load(model_path, device="cpu")
        
        self.model = _CACHED_MODEL
        
        if _CACHED_NORMS is None:
            norm_path = os.path.join(base_dir, "vec_normalize.pkl")
            with open(norm_path, "rb") as f:
                _CACHED_NORMS = pickle.load(f)

        self.obs_rms = _CACHED_NORMS.obs_rms
        self.epsilon = _CACHED_NORMS.epsilon
        self.clip_obs = _CACHED_NORMS.clip_obs

        self.prev_action = 0.0
        self.FUTURE_PLAN_STEPS = 20

    def pad_sequence(self, seq, target_len):
        seq = np.array(seq)
        if len(seq) >= target_len:
            return seq[:target_len]
        pad_val = seq[-1] if len(seq) > 0 else 0.0
        return np.pad(seq, (0, target_len - len(seq)), mode='constant', constant_values=pad_val)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        current_state = [
            state.roll_lataccel,
            state.v_ego,
            state.a_ego,
            target_lataccel,
            current_lataccel,
            self.prev_action
        ]

        future_states = np.concatenate([
            self.pad_sequence(future_plan.roll_lataccel, self.FUTURE_PLAN_STEPS),
            self.pad_sequence(future_plan.v_ego, self.FUTURE_PLAN_STEPS),
            self.pad_sequence(future_plan.a_ego, self.FUTURE_PLAN_STEPS),
            self.pad_sequence(future_plan.lataccel, self.FUTURE_PLAN_STEPS)
        ])

        obs = np.concatenate([current_state, future_states]).astype(np.float32)

        norm_obs = np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs
        )

        action, _ = self.model.predict(norm_obs, deterministic=True)
        
        action_val = float(action[0])
        self.prev_action = action_val
        
        return action_val