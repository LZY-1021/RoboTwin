# PI05 Inference Acceleration Switches

These switches affect the PyTorch `model.safetensors` inference path:

```text
deploy_policy.py -> pi_model.py -> policy.infer() -> PI0Pytorch.sample_actions()
```

Set environment variables before constructing the policy.

## Prefill MLP Reuse

Enable sparse MLP reuse in the Gemma prefix/prefill path:

```bash
PI0_MLP_REUSE=1
PI0_MLP_REUSE_REL_THRESHOLD=0.02
PI0_MLP_REUSE_MIN_SKIP_RATIO=0.0
PI0_MLP_REUSE_UPDATE_CACHE=1
PI0_MLP_REUSE_STATS=1
```

The cache is kept on each `GemmaMLP` layer as `_pi0_mlp_prev_x` and `_pi0_mlp_prev_y`.

## Denoise KV Modes

Default mode keeps the original behavior and `torch.compile` path:

```bash
PI05_DENOISE_KV_MODE=fresh
```

Layer-accumulated replacement uses previous-frame KV for layers that have not yet
been replaced, and cumulatively replaces more layers with current-frame KV each
denoise step:

```bash
PI05_DENOISE_KV_MODE=layer_accumulate
PI05_DENOISE_KV_LAYERS_PER_STEP=2
PI05_DENOISE_KV_INITIAL_CURRENT_LAYERS=0
```

Step-cutoff mode uses previous-frame KV for early denoise steps, then switches to
all current-frame KV:

```bash
PI05_DENOISE_KV_MODE=step_cutoff
PI05_DENOISE_KV_CUTOFF_STEP=5
```

The first request, or any request whose KV shapes do not match the previous
request, automatically falls back to current-frame KV.
