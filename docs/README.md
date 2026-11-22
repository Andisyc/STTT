# ICE-Cal engineering documentation

This directory contains stable engineering explanations and repeatable operator procedures.
It does not define research semantics and does not authorize training or deployment.

## Documentation boundary

| Question | Owner |
|---|---|
| What is ICE-Cal and what should a developer read first? | [`README.md`](../README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| What must an Agent preserve? | [`AGENTS.md`](../AGENTS.md) |
| How should a repeatable engineering operation be performed? | [`runbooks/`](runbooks/) |
| How is the implementation organized? | [`engineering/`](engineering/) |
| What is the current research method or training Contract? | [`note/`](../note/) |
| What happened in one dated implementation or audit unit? | `note/fada/plans`, `note/fada/evidence`, and `note/fada/reviews` |

## Stable engineering documents

- [Server training resource control](runbooks/server-training.md)
- [Engineering document scope](engineering/README.md)

### Action-level latent world model

- Capture executed Planner–Tracker sequences with `uv run scripts/collect_latent_world_model.py +world_model_collection=default`; provide the checkpoint, capture duration, window length, exploration amplitude and risk tilt threshold explicitly. Official 64/29/7 policies also require their source profile file; schema-5 `g1_fada_state_v2` policies use `runtime_config.source_profile` embedded in the checkpoint.
- Train from disjoint episode captures with `uv run scripts/train_latent_world_model.py data.train_dirs=[...] data.validation_dirs=[...] output_dir=...`. Observation width is resolved from the capture contract. Pretraining samples short forecasts with gradient-free history burn-in; `training.max_updates=200000` sets the update budget. Validation and checkpoints default to every 1,000 updates.
- Select `path_following=latent_wm` in `scripts/play_fada_online.py`, bind the independent WM checkpoint, set 29 source-order residual limits, and set `online_adaptation.enabled=false`. Runtime settings live in `conf/sttt/latent_path_following_online.yaml`.
- Evaluate with `scripts/evaluate_path_following.py --config-name latent_path_following_evaluate`; explicitly select `latent_wm` and/or `latent_wm_frozen`, cases, physical thresholds and output directory.

The WM consumes target-order env-input actions. Tracker history retains source-order commands. Single-step motion labels use each interval's starting horizontal body-heading frame in meters/meters/radians; window labels compose those motions with SE(2), rather than summing vectors in rotating frames. Encoder ego-motion inputs use only received measurements. Version-2 datasets and WM artifacts reject old fixed-path-frame artifacts; do not relabel old files as compatible. The reference RSSM and vector decoder are adapted from MIT-licensed TorchRL commit `1d94cfa5c35f9305b402fb71f0378f9cd7f09b28`.

Both the official 64/29/7 and source-bound 66/29/3 execution layouts use the same collection/training/runtime pipeline. Joint intervention and gravity decoding follow the declared layout. Legacy Tracker histories preserve their repeated-observation cold start; WM history masks identify real observations. Imagination advances the source gait clock instead of learning its deterministic phase. Set `profile_path=null` for embedded-profile evaluation; nominal and adapted modes restore the same source environment. Existing plain playback stays unchanged unless `training.play_use_checkpoint_profile=true` is explicitly selected.

Embedded legacy configuration migration removes retired reward-only parameters, translates empty scene/slope/payload-schedule declarations and restores PD gains at the environment root. Playback uses the composed task reward configuration while preserving source gait frequency, target height and termination thresholds; it rejects observation-scaling changes. Source checkpoint metadata stays intact. Active unsupported scene/schedule features and conflicting nested gains are rejected; other unknown fields and reward terms retain strict validation.

From the repository root, a clean, zero-residual baseline capture for the uploaded policy is:

```bash
uv run --no-sync python scripts/collect_latent_world_model.py \
  task=g1_walk_flat/mujoco recording_profile=straight_payload \
  +world_model_collection=default \
  training.play_checkpoint_path=model/mixup_0kg/student_iter_0031.pt \
  online_adaptation.enabled=false env.runtime_payload.enabled=false \
  video_recording.enabled=false \
  world_model_collection.output_dir="outputs/wm_collection/baseline_$(date +%Y%m%d_%H%M%S)" \
  world_model_collection.num_steps=3000 world_model_collection.window_steps=10 \
  world_model_collection.residual_amplitude_rad=0.0 \
  world_model_collection.risk_tilt_limit_rad=0.8 \
  world_model_collection.seed=0 world_model_collection.device=cpu
```

Each invocation captures one episode, stopping on termination or the configured step budget. Use distinct directories/seeds for subsequent captures; collect exploration data with an explicitly selected residual range before formal WM pretraining.

Pretraining writes resumable `last.pt` and deployment `best.pt`, selected by validation window XY RMSE. Continue the same run with `training.resume_checkpoint=.../last.pt` and a larger `training.max_updates`; sampler, optimizer and random-generator states are restored. The legacy epoch/output-checkpoint options are replaced by the update budget and `output_dir`.

For simulated captures, `world_model_collection.observation_augmentation` configures position/yaw noise, receive delay/jitter and dropout with an independent seed. Only received encoder inputs are augmented; physical motion and risk labels remain unchanged. `capture.json` records the recipe and release times. Zero defaults preserve clean inputs; configure the measured deployment noise and latency explicitly.

For a fixed-start WM action-response contrast, use `scripts/probe_latent_response_directions.py` with policy/WM checkpoints and physical probe settings. Four fresh sessions compare zero/positive/negative/zero residuals after identical warmup, using common RSSM random numbers and fixed nominal commands. Measured and predicted endpoint displacements share the probe-start heading frame. Branch NPZs retain both body-motion predictions and reference-relative increments. This entry never trains networks or restores native snapshots.

### Reference-conditioned offline pretraining

- `uv run scripts/pretrain_latent_path_following.py` uses `conf/sttt/latent_pretraining.yaml`: initial clean captures → supervised WM pretraining → frozen-WM actor training → configured physical recapture/WM/A refinement cycles → frozen nominal-versus-A evaluation. Set the policy checkpoint, output directory, capture residual range, risk threshold and all 29 actor residual bounds explicitly. Official policies require `profile_path`; legacy policies use their embedded profile.
- Set `capture.initial=false` and supply disjoint `wm.data.train_dirs` / `validation_dirs` to reuse version-2 captures. Refinement keeps the initial validation split held out and adds only new training captures. Sensor noise and receive augmentation remain zero.
- `uv run scripts/train_latent_path_actor.py` runs A pretraining independently against an existing WM. Provide `world_model_checkpoint`, `policy_checkpoint`, `data_dirs`, `output_dir` and `training.residual_limit_rad`. A reads local reference points, tangents and desired speeds; it does not receive a trajectory-class label. Different reference families share one actor.
- Actor checkpoints record their WM base identity, preview schema, residual bounds and optimizer/sampler/random states. `resume_checkpoint` restores the same training process into a new output directory; `initial_actor_checkpoint` explicitly warm-starts A after WM refinement. WM uses `training.initial_checkpoint` for the corresponding fresh-optimizer warm start.
- Deploy through `path_following=latent_wm` with `path_following.pretrained_actor_checkpoint=...`; select the path in `path_following.learning.online.trajectory`. Online resume and pretrained initialization are mutually exclusive. Planner–Tracker weights and execution mapping stay fixed; initial calibration does not erase pretrained A weights.
- `uv run scripts/evaluate_pretrained_latent_actor.py` uses `conf/sttt/latent_pretrained_evaluate.yaml`. Supply the policy, WM and actor checkpoints and output directory. It runs the same actor and nominal baseline with matched physical starts and no learning, reporting lateral error, forward-speed retention and measured risk for each configured reference.


## Process material

`superpowers/` contains dated implementation plans and design working records produced by the
Superpowers workflow. It is not the current research authority or the canonical operator manual.
When one of those records conflicts with `AGENTS.md`, an active Contract, or a current runbook,
the current authority wins.

