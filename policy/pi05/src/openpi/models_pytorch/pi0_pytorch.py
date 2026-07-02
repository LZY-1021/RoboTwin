import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch import trace_utils


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _optional_int_env(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _maybe_seed_sample_noise(device) -> int | None:
    value = os.environ.get("PI05_SAMPLE_SEED")
    if value is None or value.strip() == "":
        return None
    seed = int(value)
    torch.manual_seed(seed)
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self._denoise_kv_mode = os.environ.get("PI05_DENOISE_KV_MODE", "fresh").lower()
        self._last_prefix_pad_masks = None
        self._last_past_key_values = None
        self._last_kv_mode_stats = None
        self._trace_image_token_meta = []
        if self._denoise_kv_mode in {"sparse_attention", "sparse", "row_static"}:
            self.enable_denoise_sparse_attention(
                image_top_k=int(os.environ.get("PI05_DENOISE_SPARSE_IMAGE_TOP_K", "32")),
                prompt_top_k=_optional_int_env("PI05_DENOISE_SPARSE_PROMPT_TOP_K", 32),
            )
        if self._denoise_kv_mode == "fresh" and os.environ.get("PI05_TORCH_COMPILE", "1") == "1":
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def enable_denoise_sparse_attention(
        self,
        image_top_k: int = 32,
        prompt_top_k: int | None = 32,
        image_ranges: tuple[tuple[int, int], ...] = ((0, 256), (256, 512), (512, 768)),
    ):
        """Enable row-static step-0 sensitive-K/V attention for denoise layers."""
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layer.self_attn.enable_row_static_step0_sensitive_kv(
                image_ranges=image_ranges,
                image_top_k=image_top_k,
                prompt_top_k=prompt_top_k,
            )

    def disable_denoise_sparse_attention(self):
        """Disable denoise sparse attention and return to dense attention."""
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layer.self_attn.disable_row_static_step0_sensitive_kv()

    def reset_denoise_sparse_attention_cache(self):
        """Clear per-layer sensitive-K/V caches before a new denoise rollout."""
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layer.self_attn.reset_row_static_step0_sensitive_kv_cache()

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        image_token_meta = []
        image_embeds_for_trace = {}
        token_cursor = 0
        obs_key_map = {
            "base_0_rgb": "cam_high",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }

        # Process images
        for image_key, img, img_mask in zip(_preprocessing.IMAGE_KEYS, images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]
            grid_h = int(math.sqrt(num_img_embs))
            grid_w = num_img_embs // grid_h if grid_h > 0 and num_img_embs % grid_h == 0 else num_img_embs
            if grid_h * grid_w != num_img_embs:
                grid_h = 1
            token_start = token_cursor
            token_end = token_start + num_img_embs
            image_token_meta.append(
                {
                    "image_key": image_key,
                    "obs_key": obs_key_map.get(image_key, image_key),
                    "token_start": token_start,
                    "token_end": token_end,
                    "num_tokens": num_img_embs,
                    "grid": [grid_h, grid_w],
                    "embedding_shape": list(img_emb.shape),
                }
            )
            image_embeds_for_trace[image_key] = img_emb
            token_cursor = token_end

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        language_start = token_cursor
        language_end = language_start + num_lang_embs
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        trace_utils.save_prefix_tokens(
            {
                "image_tokens": image_token_meta,
                "language": {
                    "token_start": language_start,
                    "token_end": language_end,
                    "num_tokens": num_lang_embs,
                    "embedding_shape": list(lang_emb.shape),
                },
                "prefix_seq_len": int(embs.shape[1]),
            },
            image_embeds_for_trace,
        )
        self._trace_image_token_meta = image_token_meta

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            self._last_sample_seed = _maybe_seed_sample_noise(device)
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            self._last_sample_seed = None
        if self._denoise_kv_mode in {"sparse_attention", "sparse", "row_static"}:
            self.reset_denoise_sparse_attention_cache()

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        self._maybe_inject_vla_cache_donor(prefix_pad_masks, device)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        if self._denoise_kv_mode in {"layer_accumulate", "layerwise", "layer_accum"} and self._has_previous_prefix(
            prefix_pad_masks
        ):
            return self._sample_actions_layer_accumulate(
                device,
                bsize,
                state,
                noise,
                num_steps,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_2d_masks_4d,
                prefix_position_ids,
            )

        if self._denoise_kv_mode in {"step_cutoff", "cutoff"} and self._has_previous_prefix(prefix_pad_masks):
            return self._sample_actions_step_cutoff(
                device,
                bsize,
                state,
                noise,
                num_steps,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_2d_masks_4d,
                prefix_position_ids,
            )

        save_prefix_attn = os.environ.get("PI05_TRACE_SAVE_PREFIX_ATTN", "0") == "1"
        prefix_forward_output = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            output_attentions=save_prefix_attn,
        )
        if save_prefix_attn:
            _, past_key_values, prefix_attentions = prefix_forward_output
            self._save_prefix_attention(prefix_attentions)
        else:
            _, past_key_values = prefix_forward_output
        past_key_values = self._past_key_values_to_tuple(past_key_values)
        trace_utils.save_kv("prefix_current", past_key_values)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        if self._denoise_kv_mode in {"fresh", "current", "new", "sparse_attention", "sparse", "row_static"}:
            x_t = self._run_denoise_step_range(
                0, num_steps, x_t, state, prefix_pad_masks, past_key_values, num_steps, device, bsize
            )
            if self._denoise_kv_mode in {"sparse_attention", "sparse", "row_static"}:
                self._last_kv_mode_stats = {
                    "mode": "sparse_attention",
                    "image_top_k": int(os.environ.get("PI05_DENOISE_SPARSE_IMAGE_TOP_K", "32")),
                    "prompt_top_k": os.environ.get("PI05_DENOISE_SPARSE_PROMPT_TOP_K", "32"),
                }
            return x_t

        step_idx = 0
        kv_mode = self._denoise_kv_mode
        old_cache_ok = self._can_reuse_previous_kv(prefix_pad_masks, past_key_values)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            step_past_key_values, step_prefix_pad_masks = self._select_denoise_kv(
                kv_mode,
                step_idx,
                num_steps,
                prefix_pad_masks,
                past_key_values,
                old_cache_ok,
            )
            if os.environ.get("PI05_TRACE_SAVE_KV_STEPS", "0") == "1":
                trace_utils.save_kv(f"denoise_step_{step_idx:03d}", step_past_key_values)
            v_t = self.denoise_step(
                state,
                step_prefix_pad_masks,
                step_past_key_values,
                x_t,
                expanded_time,
                step_idx=step_idx,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
            step_idx += 1
        self._last_prefix_pad_masks = prefix_pad_masks.detach()
        self._last_past_key_values = self._detach_past_key_values(past_key_values)
        return x_t

    def _run_prefix_kv(self, prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids):
        save_prefix_attn = os.environ.get("PI05_TRACE_SAVE_PREFIX_ATTN", "0") == "1"
        prefix_forward_output = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            output_attentions=save_prefix_attn,
        )
        if save_prefix_attn:
            _, past_key_values, prefix_attentions = prefix_forward_output
            self._save_prefix_attention(prefix_attentions)
        else:
            _, past_key_values = prefix_forward_output
        past_key_values = self._past_key_values_to_tuple(past_key_values)
        trace_utils.save_kv("prefix_current", past_key_values)
        return past_key_values

    def _run_denoise_step_range(
        self,
        step_start: int,
        step_end: int,
        x_t,
        state,
        prefix_pad_masks,
        past_key_values,
        num_steps: int,
        device,
        bsize: int,
    ):
        dt = -1.0 / num_steps
        for step_idx in range(step_start, step_end):
            timestep = torch.tensor(1.0 + step_idx * dt, dtype=torch.float32, device=device).expand(bsize)
            if os.environ.get("PI05_TRACE_SAVE_KV_STEPS", "0") == "1":
                trace_utils.save_kv(f"denoise_step_{step_idx:03d}", past_key_values)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
                step_idx=step_idx,
            )
            x_t = x_t + dt * v_t
        return x_t

    def _sample_actions_step_cutoff(
        self,
        device,
        bsize: int,
        state,
        noise,
        num_steps: int,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_2d_masks_4d,
        prefix_position_ids,
    ):
        cutoff = min(num_steps, max(0, int(os.environ.get("PI05_DENOISE_KV_CUTOFF_STEP", "5"))))
        x_t = self._run_denoise_step_range(
            0, cutoff, noise, state, self._last_prefix_pad_masks, self._last_past_key_values, num_steps, device, bsize
        )
        current_past_key_values = self._run_prefix_kv(prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids)
        x_t = self._run_denoise_step_range(
            cutoff, num_steps, x_t, state, prefix_pad_masks, current_past_key_values, num_steps, device, bsize
        )
        self._last_prefix_pad_masks = prefix_pad_masks.detach()
        self._last_past_key_values = self._detach_past_key_values(current_past_key_values)
        self._last_kv_mode_stats = {"mode": "step_cutoff", "cutoff": cutoff}
        return x_t

    def _sample_actions_layer_accumulate(
        self,
        device,
        bsize: int,
        state,
        noise,
        num_steps: int,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_2d_masks_4d,
        prefix_position_ids,
    ):
        language_model = self.paligemma_with_expert.paligemma.language_model
        layers_per_step = max(1, int(os.environ.get("PI05_DENOISE_KV_LAYERS_PER_STEP", "2")))
        start_layers = max(0, int(os.environ.get("PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS", "0")))
        total_layers = len(language_model.layers)
        ctx = language_model.prepare_forward(prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids)
        hidden_states = ctx["hidden_states"]
        current_layer_count = 0
        x_t = noise
        dt = -1.0 / num_steps

        for step_idx in range(num_steps):
            target_layer_count = min(total_layers, start_layers + (step_idx + 1) * layers_per_step)
            if target_layer_count > current_layer_count:
                hidden_states = language_model.forward_layers(
                    hidden_states, ctx, current_layer_count, target_layer_count
                )
                current_layer_count = target_layer_count
            current_partial_kv = self._past_key_values_to_tuple(language_model.get_accumulated_cache(ctx))
            mixed_past_key_values = self._mix_past_key_values(
                current_partial_kv,
                self._last_past_key_values,
                current_layer_count,
            )
            if os.environ.get("PI05_TRACE_SAVE_KV_STEPS", "0") == "1":
                trace_utils.save_kv(f"denoise_step_{step_idx:03d}", mixed_past_key_values)
            timestep = torch.tensor(1.0 + step_idx * dt, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                mixed_past_key_values,
                x_t,
                timestep,
                step_idx=step_idx,
            )
            x_t = x_t + dt * v_t

        if current_layer_count < total_layers:
            hidden_states = language_model.forward_layers(hidden_states, ctx, current_layer_count, total_layers)
        language_model.finalize(hidden_states, ctx)
        current_past_key_values = self._past_key_values_to_tuple(language_model.get_accumulated_cache(ctx))
        self._last_prefix_pad_masks = prefix_pad_masks.detach()
        self._last_past_key_values = self._detach_past_key_values(current_past_key_values)
        self._last_kv_mode_stats = {
            "mode": "layer_accumulate",
            "layers_per_step": layers_per_step,
            "total_layers": total_layers,
        }
        return x_t

    def _has_previous_prefix(self, prefix_pad_masks) -> bool:
        return self._last_prefix_pad_masks is not None and self._last_prefix_pad_masks.shape == prefix_pad_masks.shape

    def _maybe_inject_vla_cache_donor(self, prefix_pad_masks, device) -> None:
        if os.environ.get("VLA_CACHE_ENABLE", "0") != "1":
            return
        if self._denoise_kv_mode not in {"step_cutoff", "cutoff", "layer_accumulate", "layerwise", "layer_accum"}:
            return
        try:
            from vla_serving.cache import inject_robotwin_donor_kv

            result = inject_robotwin_donor_kv(
                torch_model=self,
                current_infer_dir=trace_utils.current_infer_dir(),
                prefix_pad_masks=prefix_pad_masks,
                device=device,
            )
            if result.get("enabled"):
                self._last_kv_mode_stats = {**(self._last_kv_mode_stats or {}), "cache_query": result}
        except Exception as exc:  # noqa: BLE001
            self._last_kv_mode_stats = {
                **(self._last_kv_mode_stats or {}),
                "cache_query": {"enabled": True, "used": False, "reason": "exception", "error": str(exc)},
            }

    def _past_key_values_to_tuple(self, past_key_values):
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return tuple(zip(past_key_values.key_cache, past_key_values.value_cache, strict=True))
        return tuple((layer[0], layer[1]) for layer in past_key_values)

    def _past_key_values_to_cache(self, past_key_values):
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def _can_reuse_previous_kv(self, prefix_pad_masks, past_key_values) -> bool:
        prev_masks = self._last_prefix_pad_masks
        prev_kv = self._last_past_key_values
        if prev_masks is None or prev_kv is None:
            return False
        if prev_masks.shape != prefix_pad_masks.shape:
            return False
        if len(prev_kv) != len(past_key_values):
            return False
        for old_layer, new_layer in zip(prev_kv, past_key_values, strict=True):
            if old_layer[0].shape != new_layer[0].shape or old_layer[1].shape != new_layer[1].shape:
                return False
        return True

    def _detach_past_key_values(self, past_key_values):
        return tuple((k.detach(), v.detach()) for k, v in past_key_values)

    def _mix_past_key_values(self, current_past_key_values, previous_past_key_values, current_layer_count: int):
        mixed = []
        for layer_idx, previous_layer in enumerate(previous_past_key_values):
            if layer_idx < current_layer_count and layer_idx < len(current_past_key_values):
                mixed.append(current_past_key_values[layer_idx])
            else:
                mixed.append(previous_layer)
        return tuple(mixed)

    def _select_denoise_kv(
        self,
        kv_mode: str,
        step_idx: int,
        num_steps: int,
        current_prefix_pad_masks,
        current_past_key_values,
        old_cache_ok: bool,
    ):
        if kv_mode in {"fresh", "current", "new"} or not old_cache_ok:
            self._last_kv_mode_stats = {"mode": "fresh", "step": step_idx, "current_layers": len(current_past_key_values)}
            return current_past_key_values, current_prefix_pad_masks

        previous_prefix_pad_masks = self._last_prefix_pad_masks
        previous_past_key_values = self._last_past_key_values

        if kv_mode in {"step_cutoff", "cutoff"}:
            cutoff = int(os.environ.get("PI05_DENOISE_KV_CUTOFF_STEP", "5"))
            if step_idx < cutoff:
                self._last_kv_mode_stats = {"mode": "step_cutoff_old", "step": step_idx, "cutoff": cutoff}
                return previous_past_key_values, previous_prefix_pad_masks
            self._last_kv_mode_stats = {"mode": "step_cutoff_fresh", "step": step_idx, "cutoff": cutoff}
            return current_past_key_values, current_prefix_pad_masks

        if kv_mode in {"layer_accumulate", "layerwise", "layer_accum"}:
            total_layers = len(current_past_key_values)
            layers_per_step = int(os.environ.get("PI05_DENOISE_KV_LAYERS_PER_STEP", "2"))
            start_layers = int(os.environ.get("PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS", "0"))
            current_layer_count = min(total_layers, start_layers + (step_idx + 1) * max(1, layers_per_step))
            mixed_past_key_values = self._mix_past_key_values(
                current_past_key_values,
                previous_past_key_values,
                current_layer_count,
            )
            self._last_kv_mode_stats = {
                "mode": "layer_accumulate",
                "step": step_idx,
                "current_layers": current_layer_count,
                "total_layers": total_layers,
            }
            return mixed_past_key_values, current_prefix_pad_masks

        raise ValueError(
            "Unsupported PI05_DENOISE_KV_MODE. Use fresh, sparse_attention, layer_accumulate, or step_cutoff."
        )

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        step_idx: int | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        save_attn = os.environ.get("PI05_TRACE_SAVE_ATTN", "0") == "1"
        save_qk_logits = os.environ.get("PI05_TRACE_SAVE_QK_LOGITS", "0") == "1"
        forward_output = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=self._past_key_values_to_cache(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            output_attentions=save_attn or save_qk_logits,
        )
        if save_attn or save_qk_logits:
            outputs_embeds, _, attentions = forward_output
            if save_attn:
                self._save_denoise_attention(step_idx, attentions)
            if save_qk_logits:
                self._save_denoise_qk_logits(step_idx)
        else:
            outputs_embeds, _ = forward_output

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def _parse_trace_attention_layers(self, total_layers: int) -> list[int]:
        value = os.environ.get("PI05_TRACE_ATTN_LAYERS", "0,3,6,9,12,15,17").strip()
        if value in {"", "all"}:
            return list(range(total_layers))
        layers = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            layer = int(item)
            if 0 <= layer < total_layers:
                layers.append(layer)
        return layers

    def _save_denoise_attention(self, step_idx: int | None, attentions) -> None:
        if attentions is None or not self._trace_image_token_meta:
            return
        layers = self._parse_trace_attention_layers(len(attentions))
        save_full = os.environ.get("PI05_TRACE_SAVE_ATTN_FULL", "0") == "1"
        payload = {
            "step": int(step_idx) if step_idx is not None else None,
            "layers": {},
            "full_layers": {},
            "image_tokens": self._trace_image_token_meta,
            "query_axis": {},
            "description": (
                "Attention from denoise suffix/action queries to image prefix tokens. "
                "`layers` averages over heads and suffix queries; `full_layers` keeps [heads, queries, grid_h, grid_w] "
                "when PI05_TRACE_SAVE_ATTN_FULL=1."
            ),
        }
        for layer in layers:
            attn = attentions[layer]
            if attn is None or attn.ndim != 4:
                continue
            # Shape is [B, heads, suffix_query_len, prefix_len + suffix_len].
            heads = int(attn.shape[1])
            suffix_query_len = int(attn.shape[2])
            action_horizon = int(self.config.action_horizon)
            state_query_count = max(0, suffix_query_len - action_horizon)
            payload["query_axis"] = {
                "num_heads": heads,
                "suffix_query_len": suffix_query_len,
                "state_query_count": state_query_count,
                "action_query_start": state_query_count,
                "action_query_count": max(0, suffix_query_len - state_query_count),
                "action_horizon": action_horizon,
            }
            layer_payload = {}
            full_layer_payload = {}
            for item in self._trace_image_token_meta:
                start = int(item["token_start"])
                end = int(item["token_end"])
                grid_h, grid_w = item["grid"]
                image_attn = attn[0, :, :, start:end].detach().to(torch.float32)
                values = image_attn.mean(dim=(0, 1))
                if values.numel() != grid_h * grid_w:
                    continue
                layer_payload[item["image_key"]] = values.reshape(grid_h, grid_w)
                if save_full:
                    full_layer_payload[item["image_key"]] = image_attn.reshape(heads, suffix_query_len, grid_h, grid_w)
            payload["layers"][f"layer_{layer:02d}"] = layer_payload
            if save_full:
                payload["full_layers"][f"layer_{layer:02d}"] = full_layer_payload
        if not save_full:
            payload.pop("full_layers", None)
        label_step = 0 if step_idx is None else step_idx
        trace_utils.save_attention(f"denoise_step_{label_step:03d}", payload)

    def _save_denoise_qk_logits(self, step_idx: int | None) -> None:
        if os.environ.get("PI05_TRACE_SAVE_QK_LOGITS", "0") != "1" or not self._trace_image_token_meta:
            return
        layers = self._parse_trace_attention_layers(len(self.paligemma_with_expert.gemma_expert.model.layers))
        image_key_filter = os.environ.get("PI05_TRACE_QK_IMAGE_KEYS", "").strip()
        if image_key_filter and image_key_filter.lower() != "all":
            allowed_image_keys = {item.strip() for item in image_key_filter.split(",") if item.strip()}
        else:
            allowed_image_keys = None
        payload = {
            "step": int(step_idx) if step_idx is not None else None,
            "layers": {},
            "image_tokens": self._trace_image_token_meta,
            "query_axis": {},
            "description": (
                "Pre-softmax denoise attention logits and their Q/K inputs. "
                "Each layer stores query_states [heads, suffix_query_len, head_dim], "
                "image_key_states/image_value_states [heads, image_tokens, head_dim], and "
                "image_logits [heads, suffix_query_len, grid_h, grid_w]."
            ),
        }
        for layer in layers:
            attn_module = self.paligemma_with_expert.gemma_expert.model.layers[layer].self_attn
            query_states = getattr(attn_module, "_pi05_last_query_states", None)
            key_states = getattr(attn_module, "_pi05_last_key_states", None)
            value_states = getattr(attn_module, "_pi05_last_value_states", None)
            logits = getattr(attn_module, "_pi05_last_attn_logits", None)
            if query_states is None or key_states is None or value_states is None or logits is None:
                continue
            if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4 or logits.ndim != 4:
                continue
            heads = int(query_states.shape[1])
            suffix_query_len = int(query_states.shape[2])
            action_horizon = int(self.config.action_horizon)
            state_query_count = max(0, suffix_query_len - action_horizon)
            payload["query_axis"] = {
                "num_heads": heads,
                "suffix_query_len": suffix_query_len,
                "state_query_count": state_query_count,
                "action_query_start": state_query_count,
                "action_query_count": max(0, suffix_query_len - state_query_count),
                "action_horizon": action_horizon,
            }
            layer_payload = {
                "query_states": query_states[0].detach().to(torch.float32),
                "image_key_states": {},
                "image_value_states": {},
                "image_logits": {},
            }
            for item in self._trace_image_token_meta:
                image_key = item["image_key"]
                if allowed_image_keys is not None and image_key not in allowed_image_keys:
                    continue
                start = int(item["token_start"])
                end = int(item["token_end"])
                grid_h, grid_w = item["grid"]
                image_keys = key_states[0, :, start:end, :].detach().to(torch.float32)
                image_values = value_states[0, :, start:end, :].detach().to(torch.float32)
                image_logits = logits[0, :, :, start:end].detach().to(torch.float32)
                if image_logits.numel() != heads * suffix_query_len * grid_h * grid_w:
                    continue
                layer_payload["image_key_states"][image_key] = image_keys
                layer_payload["image_value_states"][image_key] = image_values
                layer_payload["image_logits"][image_key] = image_logits.reshape(heads, suffix_query_len, grid_h, grid_w)
            payload["layers"][f"layer_{layer:02d}"] = layer_payload
            delattr(attn_module, "_pi05_last_query_states")
            delattr(attn_module, "_pi05_last_key_states")
            delattr(attn_module, "_pi05_last_value_states")
            delattr(attn_module, "_pi05_last_attn_logits")
        label_step = 0 if step_idx is None else step_idx
        trace_utils.save_qk_logits(f"denoise_step_{label_step:03d}", payload)

    def _save_prefix_attention(self, attentions) -> None:
        if attentions is None or not self._trace_image_token_meta:
            return
        layers = self._parse_trace_attention_layers(len(attentions))
        payload = {
            "layers": {},
            "image_tokens": self._trace_image_token_meta,
            "description": "Mean prefix attention received by each image prefix token from all prefix queries.",
        }
        for layer in layers:
            attn = attentions[layer]
            if attn is None or attn.ndim != 4:
                continue
            # Shape is [B, heads, prefix_query_len, prefix_key_len].
            layer_payload = {}
            for item in self._trace_image_token_meta:
                start = int(item["token_start"])
                end = int(item["token_end"])
                grid_h, grid_w = item["grid"]
                values = attn[0, :, :, start:end].detach().to(torch.float32).mean(dim=(0, 1))
                if values.numel() != grid_h * grid_w:
                    continue
                layer_payload[item["image_key"]] = values.reshape(grid_h, grid_w)
            payload["layers"][f"layer_{layer:02d}"] = layer_payload
        trace_utils.save_attention("prefix", payload)
