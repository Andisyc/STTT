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

- `uv run scripts/pretrain_latent_path_following.py` uses `conf/sttt/latent_pretraining.yaml`. It reuses or collects at least 1,000,000 real control records over 64 streams, warms up WM for 10,000 updates, then runs 3,000 iterations of **collect 20 native steps per stream → WM 8 updates → Critic 8 passes over 4 minibatches → A 1 update**. Effective batches remain A=64 and WM=256. Totals are A=3,000, WM=34,000, and normally Critic=96,000 optimizer updates.
- The Critic cadence follows [PWM's online configuration](https://github.com/imgeorgiev/PWM/blob/main/scripts/cfg/alg/pwm.yaml). Native-step discount is 0.99; a ten-step window uses `0.99**10`. A optimizes integrated positive tracking costs plus a frozen target Critic's terminal cost. TD-lambda targets use lambda 0.95 and an EMA target with tau 0.01. Actual failure windows retain the absorbing failure tail; time-limit truncations are not failures.
- `capture.conditions` assigns independent stream groups to named physical cases with `source_overrides`; its shipped case is nominal. Desired velocity commands are sampled periodically within `capture.command_ranges` (or the embedded source-profile ranges), integrated into a fixed reference, and presented to one preview-conditioned A. The path origin stays fixed across tracking drift. Noise remains disabled. Completed, terminal and still-growing episodes feed a bounded replay without crossing resets; archived records include the learner versions that produced each held residual.
- Set the policy checkpoint, output directory, exploration amplitude, tilt threshold and 29 actor residual bounds explicitly. Official policies also require `profile_path`. `calibration.response_probe_result` consumes the existing four-branch response probe's `summary.json` and `source.json`; `calibration.require_response_probe=false` explicitly selects an uncalibrated baseline. `actor.costs` declares physical task tolerances and bounded tracking weights; initial capture RMS never changes them. WM motion losses normalize by physical prediction tolerances (1 cm window XY and 0.02 rad window turn by default), while retaining continuous turn supervision and actual Action labels.
  Command ranges must include both left and right turns; a straight-only embedded profile requires an explicitly supplied supported `capture.command_ranges`.
- Set `capture.initial=false` with disjoint `wm.data.train_dirs` / `validation_dirs` to reuse version-2 captures meeting `capture.initial_min_transitions`. Fresh validation uses independent seeds and at least `capture.validation_episodes` completed episodes, rounded up evenly across physical cases. The initial validation split stays held out throughout joint training.
- Version-2 joint `last.pt` atomically saves WM/A/Critic weights, target Critic, optimizers, replay, RNG and separate counters. Resume into a **new output directory**; physics starts new episodes. Version-2 `actor.pt` includes its Critic and objective contract. Legacy A artifacts are rejected; compatible old WM weights and version-2 physical sequence archives remain valid warm-start inputs.
- Independent WM and Actor/Critic entries use the same learners. Deployment requires `path_following.pretrained_actor_checkpoint` for A adaptation. Online cycles update WM LoRA, Critic and A once each on a single background worker. The controller continues with separate inference copies and publishes completed WM/A snapshots only at residual-window boundaries. Reset/close join the worker; version-3 online checkpoints include every optimizer and Critic state.
- `uv run scripts/evaluate_pretrained_latent_actor.py` uses `conf/sttt/latent_pretrained_evaluate.yaml`. It compares nominal/A under matched physical starts across configured references, seeds and physical cases. Reports include measured lateral error, forward-speed retention and failure events, plus closed-loop motion/action errors and low-risk actual-event fractions by forecast lead window. `training_complete` records optimizer completion; physical evaluation metrics remain separate.

The joint command, after setting the source-specific physical inputs, is:

```bash
uv run scripts/pretrain_latent_path_following.py \
  policy_checkpoint="${POLICY_CHECKPOINT:?set policy checkpoint}" \
  output_dir="outputs/latent_joint_$(date +%Y%m%d_%H%M%S)" device=cuda \
  capture.residual_amplitude_rad="${EXPLORATION_RAD:?set measured exploration range}" \
  capture.risk_tilt_limit_rad="${TILT_LIMIT_RAD:?set physical tilt threshold}" \
  "actor.training.residual_limit_rad=${RESIDUAL_LIMITS_RAD29:?set 29 source-order limits}" \
  calibration.response_probe_result="${RESPONSE_PROBE_SUMMARY:?set probe summary.json}"
```


## Process material

`superpowers/` contains dated implementation plans and design working records produced by the
Superpowers workflow. It is not the current research authority or the canonical operator manual.
When one of those records conflicts with `AGENTS.md`, an active Contract, or a current runbook,
the current authority wins.

