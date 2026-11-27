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
- Select `path_following=latent_wm` in `scripts/play_fada_online.py`, bind the WM and stage-2 `path_following.pretrained_actor_checkpoint`, set 29 source-order residual limits, and set `online_adaptation.enabled=false`. `conf/sttt/latent_path_following_online.yaml` configures WM LoRA and persistent context adaptation; A and FiLM weights stay frozen.
- Evaluate with `scripts/evaluate_path_following.py --config-name latent_path_following_evaluate`; explicitly select `latent_wm` and/or `latent_wm_frozen`, cases, physical thresholds and output directory.
- For **Planner/WM dual LoRA without A/FiLM**, use `path_following=planner_wm_lora`, set `path_following.checkpoint_path` to the independent pretrained WM, and disable `online_adaptation.enabled`. `conf/sttt/planner_wm_online.yaml` owns both optimizers and fixed physical loss scales. Each cycle identifies WM from executed ENV actions and measured proprio/odometry, re-encodes retained real history with the updated WM, then optimizes Planner Q/V LoRA through the frozen Tracker/WM against task-time reference poses. No external joint offset is injected. Learning runs off the control thread; only completed WM/Planner pairs are published at window boundaries.
- Pretrain **only WM** with `uv run scripts/train_latent_world_model.py --config-name planner_wm_pretraining data.train_dirs=[...] data.validation_dirs=[...] output_dir=...`; the preset uses 10,000 updates, effective batch 1,024, microbatch 256 and the existing motion-head LR/warmup settings. The Linux resource-controlled launcher defaults to `--stage wm`; pass `--stage base` or `--stage all` explicitly for the previous A/FiLM pipeline. Whole-episode held-out captures remain disjoint.
- Evaluate the identical timed reference with `uv run scripts/evaluate_path_following.py --config-name planner_wm_evaluate`, explicitly selecting `planner_wm_frozen`, `planner_wm_only` and/or `planner_wm_online`, cases, seeds, duration and physical thresholds. Online `save_checkpoint_path`/`resume_checkpoint_path` preserve both LoRA manifests, frozen WM base, optimizer states and counters; streams/history restart from new measured episodes. A/FiLM checkpoints are rejected. These entrypoints do not establish learned-policy quality or robot deployment; the current session factory is MuJoCo-only and cuVSLAM still needs the supported sensor/CUDA runtime.
- Planner parameter-change regularization uses a fixed adaptation-start LoRA anchor, preserved by version-2 online checkpoints; version-1 online checkpoints lack this anchor and cannot resume. Pretrained WM/policy checkpoints are unchanged. Recording `.poses.jsonl` files include per-cycle WM error in meters/radians before fitting new windows and after fitting the same windows (not held-out validation), plus Planner before/after trajectories, heading/position costs, gradients and parameter drift. With `path_following.save_checkpoint_path=.../online_lora.pt`, each completed update also writes `online_lora.updates/wm_XXXXXX_planner_XXXXXX.pt`: executed actions, measured labels, causal inputs, RNG states and both LoRA snapshots. Comparisons replay identical random draws without advancing the training RNG. Detailed diagnostics add learning-thread compute and I/O; `elapsed_s` includes that overhead.

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

For a fixed-start WM action-response contrast, use `scripts/probe_latent_response_directions.py` with policy/WM checkpoints and physical probe settings. Four fresh sessions compare zero/positive/negative/zero residuals after identical warmup, using common RSSM random numbers and fixed nominal commands. Measured and predicted endpoint displacements share the fixed reference heading anchored before warmup. Branch NPZs retain both body-motion predictions and reference-relative increments. This entry never trains networks or restores native snapshots.

### Residual control-chain diagnostics

Run `uv run --no-sync scripts/launch_latent_control_diagnostics.py --run-dir <completed-training-directory>` on the server. The launcher reuses the training launcher's physical-core reservation, effective memory budget, cgroup limits, reduced CPU/I/O priorities and GPU allocation. It returns the prompt, writes a timestamped log and keeps running after SSH disconnects. Checkpoints and training configuration remain unchanged.

The default physical-response sweep contains **472 native physical branches**. Full mode retains logged-Action WM replay versus Planner–Tracker imagination and replaces the retired cost-Critic/A-weight substep with the shared production context optimizer. Its isolated update changes only c, then compares physical continuations under fixed versus updated c.

The CLI owner is `scripts/diagnose_latent_control.py`, configured by `conf/sttt/latent_control_diagnostics.yaml`. `mode=physical` needs no WM/actor checkpoint; without `run_dir`, supply the source policy, window length and 29 residual bounds explicitly. `mode=analyze data_dir=<diagnostic-directory> output_dir=<new-directory>` reuses recorded traces. Full mode reads the stage-2 WM and actor paths from `run_dir/summary.json`. Physical traces retain owned histories, actual actions, measured poses, references and termination. Reports contain physical response, measured feedback, WM replay and context-update comparisons.

### Reference-conditioned offline pretraining

- `uv run scripts/pretrain_latent_path_following.py` uses `conf/sttt/latent_pretraining.yaml`; `stage=all` runs both stages. Stage 1 first fits WM for 10,000 supervised updates, then trains A with **native RSL-RL PPO on real simulation rewards** for 3,000 PPO iterations. Each iteration also performs 8 separate WM replay updates.
- Stage 2 freezes A and WM, enables a 16-dimensional FiLM context, and trains only FiLM plus its offline PPO Critic for 1,000 iterations. At every real residual window, the shared context optimizer differentiates the predicted trajectory error through frozen WM/Planner/Tracker to update each stream's persistent c. PPO stores the c used during collection; it does not differentiate through the c optimizer.
- Each PPO iteration collects 16 residual windows over the configured streams, then performs 5 epochs × 4 minibatches. `base_updates` and `film_updates` count PPO iterations; `wm_updates` counts WM optimizer steps. With defaults, WM totals 34,000 updates and stays fixed throughout stage 2. Dense measured rewards include all 29 joint positions/velocities against untouched Planner references, body velocity/yaw rate, posture, path error, action changes and real termination.
- The physical chain remains A → pose residual composition → frozen Planner–Tracker execution mapping and limits → actual Action. PPO retains raw Gaussian samples/log probabilities and applies tanh × physical residual limits only at execution. WM supervision always pairs logged actual Actions with their measured physical successors. `capture.conditions` supplies named physical cases; references follow resampled velocity commands, with a fixed origin and local path preview. Measurement noise remains disabled.
- Supply the policy checkpoint, output directory, exploration amplitude, tilt threshold and 29 residual bounds; official policies also need `profile_path`. Initial capture collects at least 1,000,000 real control records over 64 streams by default. To reuse version-2 captures, set `capture.initial=false` and disjoint `wm.data.train_dirs` / `validation_dirs`. The validation split stays held out.
- Version-4 joint checkpoints save the phase, A/FiLM, WM, native PPO Critic/optimizer, replay, RNG, c and its optimizer. Resume into a new output directory; physics starts new episodes. Phase exports are `base_actor.pt` / `base_world_model.pt` and `film_actor.pt` / `film_world_model.pt`. Old A and paired joint checkpoints are not PPO resumes; compatible WM weights and physical captures remain reusable.
- Online deployment freezes A/FiLM and updates only c plus the separate real-log WM LoRA learner. Completed WM/c snapshots publish at residual-window boundaries on the existing background lifecycle. Zero trajectory error and reference changes do not reset c. Version-5 online checkpoints retain c, its optimizer and WM LoRA state.
- `uv run scripts/train_latent_path_actor.py` selects the same base-stage PPO pipeline. `scripts/launch_latent_pretraining.py --stage base --data-dir <capture_run>` launches only stage 1 with the existing SSH-friendly resource isolation/background logging; the default `--stage all` runs both stages. `--ppo-minibatches` controls PPO minibatch count.
- `uv run scripts/evaluate_pretrained_latent_actor.py` compares nominal, fixed-c and adapting-c execution under matched physical starts for stage-2 artifacts; base-stage artifacts retain nominal/A comparison. Reports include measured path error, forward-speed retention, failures and forecast diagnostics.

The joint command, after setting the source-specific physical inputs, is:

```bash
uv run scripts/pretrain_latent_path_following.py \
  policy_checkpoint="${POLICY_CHECKPOINT:?set policy checkpoint}" \
  output_dir="outputs/latent_joint_$(date +%Y%m%d_%H%M%S)" device=cuda \
  capture.residual_amplitude_rad="${EXPLORATION_RAD:?set measured exploration range}" \
  capture.risk_tilt_limit_rad="${TILT_LIMIT_RAD:?set physical tilt threshold}" \
  "actor.training.residual_limit_rad=${RESIDUAL_LIMITS_RAD29:?set 29 source-order limits}"
```


## Process material

`superpowers/` contains dated implementation plans and design working records produced by the
Superpowers workflow. It is not the current research authority or the canonical operator manual.
When one of those records conflicts with `AGENTS.md`, an active Contract, or a current runbook,
the current authority wins.

