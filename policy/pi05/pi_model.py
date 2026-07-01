#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.models_pytorch import trace_utils
import os

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        asset_candidates = [
            entry
            for entry in os.listdir(specified_path)
            if os.path.isfile(os.path.join(specified_path, entry, "norm_stats.json"))
        ]
        if not asset_candidates:
            raise FileNotFoundError(f"No norm_stats.json found under: {specified_path}")
        assets_id = asset_candidates[0]

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        trace_utils.start_infer(self.observation_window)
        actions = self.policy.infer(self.observation_window)["actions"]
        trace_utils.save_action(actions, actions[:self.pi0_step])
        return actions

    def clear_mlp_reuse_cache(self):
        torch_model = getattr(self.policy, "_model", None)
        if torch_model is None or not hasattr(torch_model, "modules"):
            return
        cache_attrs = (
            "_pi0_mlp_prev_x",
            "_pi0_mlp_prev_y",
            "_pi0_mlp_out_buffer",
            "_pi0_mlp_reuse_last",
        )
        for module in torch_model.modules():
            for attr in cache_attrs:
                if hasattr(module, attr):
                    delattr(module, attr)

    def clear_denoise_kv_cache(self):
        torch_model = getattr(self.policy, "_model", None)
        if torch_model is None:
            return
        for attr in ("_last_prefix_pad_masks", "_last_past_key_values", "_last_kv_mode_stats"):
            if hasattr(torch_model, attr):
                setattr(torch_model, attr, None)

    def reset_obsrvationwindows(self):
        self.clear_mlp_reuse_cache()
        self.clear_denoise_kv_cache()
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
