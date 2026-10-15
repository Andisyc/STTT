"""Test online adaptation in simulation with MuJoCo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig
from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.base import registry

# Ensure environments are registered
registry.ensure_registries()


def main():
    # Create environment via registry
    env = registry.make(
        "g1_walk_height",
        num_envs=1,
        sim_backend="mujoco",
    )
    
    # Load trained checkpoint
    from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint
    checkpoint_path = "/Users/chengyuxuan/ArtiIntComVis/FADA_Unofficial/model/v024_checkpoint/planner_idm_close_some_dr_v001.pt"
    loaded = load_fada_policy_checkpoint(checkpoint_path, device="cpu")
    policy = loaded.policy
    config = policy.config
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  obs_dim={config.obs_dim}, action_dim={config.action_dim}")
    print(f"  history_length={config.history_length}, prediction_horizon={config.prediction_horizon}")
    
    # Create online adaptation controller
    lora_config = FADALoRAConfig(rank=8, alpha=16.0, dropout=0.05)
    controller = OnlineAdaptationController(
        policy,
        lora_config,
        lambda_forget=0.99,
        update_interval=10,
        device="cpu",
    )
    
    # Run simulation
    state = env.reset_all()  # type: ignore[attr-defined]
    obs = state.obs["obs"]
    
    print("Starting online adaptation simulation...")
    print(f"  obs shape: {obs.shape}")
    print(f"  action dim: {env.action_space.shape}")
    
    for step in range(200):
        # Get command (walk forward)
        command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
        
        # Inference
        action = controller.act(obs, command)
        action_np = action.detach().cpu().numpy()
        
        # Env step
        state = env.step(action_np)
        next_obs = state.obs["obs"]
        executed_action = state.info.get("current_actions", action_np)
        
        # Collection and update
        result = controller.on_env_step(
            executed_action=executed_action,
            next_observation=next_obs,
        )
        if result:
            print(f"Step {step}: update - loss={result['loss']:.4f}, windows={result['num_windows']}")
        
        obs = next_obs
        
        if step % 50 == 0:
            print(f"Step {step}: running...")
    
    print("Simulation complete.")
    env.close()


if __name__ == "__main__":
    main()
