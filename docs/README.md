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

- Capture executed Planner–Tracker sequences with `uv run scripts/collect_latent_world_model.py +world_model_collection=default`; provide the policy/profile, capture duration, window length, exploration amplitude and risk tilt threshold explicitly.
- Train from disjoint episode captures with `uv run scripts/train_latent_world_model.py data.train_dirs=[...] data.validation_dirs=[...] output_dir=...`. Pretraining samples short forecasts with gradient-free history burn-in; `training.max_updates=200000` sets the update budget. Validation and checkpoints default to every 1,000 updates.
- Select `path_following=latent_wm` in `scripts/play_fada_online.py`, bind the independent WM checkpoint, set 29 source-order residual limits, and set `online_adaptation.enabled=false`. Runtime settings live in `conf/sttt/latent_path_following_online.yaml`.
- Evaluate with `scripts/evaluate_path_following.py --config-name latent_path_following_evaluate`; explicitly select `latent_wm` and/or `latent_wm_frozen`, cases, physical thresholds and output directory.

The WM consumes target-order env-input actions. Tracker history retains source-order commands. Motion labels use the fixed straight-path frame in meters/meters/radians; encoder ego-motion inputs use only received measurements. New artifacts remain separate from existing B checkpoints. The reference RSSM and vector decoder are adapted from MIT-licensed TorchRL commit `1d94cfa5c35f9305b402fb71f0378f9cd7f09b28`, using the repository's existing PyTorch version.

Pretraining writes resumable `last.pt` and deployment `best.pt`, selected by validation window XY RMSE. Continue the same run with `training.resume_checkpoint=.../last.pt` and a larger `training.max_updates`; sampler, optimizer and random-generator states are restored. The legacy epoch/output-checkpoint options are replaced by the update budget and `output_dir`.

For simulated captures, `world_model_collection.observation_augmentation` configures position/yaw noise, receive delay/jitter and dropout with an independent seed. Only received encoder inputs are augmented; physical motion and risk labels remain unchanged. `capture.json` records the recipe and release times. Zero defaults preserve clean inputs; configure the measured deployment noise and latency explicitly.

## Process material

`superpowers/` contains dated implementation plans and design working records produced by the
Superpowers workflow. It is not the current research authority or the canonical operator manual.
When one of those records conflicts with `AGENTS.md`, an active Contract, or a current runbook,
the current authority wins.

