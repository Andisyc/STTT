"""Matched-noise diagnostics for real-action WM fitting and Planner adaptation."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from unilab.algos.torch.sttt.geometry import wrap_to_pi
from unilab.algos.torch.sttt.path_following.planner_losses import predicted_poses


def capture_rng(model):
    device = next(model.parameters()).device
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


@contextmanager
def replay_rng(model, state):
    """Repeat a diagnostic draw without advancing the learner's random stream."""
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.set_rng_state(state["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], device)
        yield


def input_evidence(inputs):
    decision = inputs.decision
    state = decision.path_state
    return {
        "timestamp_ns": decision.timestamp_ns,
        "sequence_id": decision.reference.sequence_id,
        "tracking_epoch": decision.reference.tracking_epoch,
        "reference_timestamp_ns": decision.reference.timestamp_ns,
        "context": decision.context,
        "path_errors": decision.errors,
        "pose": None if state is None else state.pose,
        "progress": None if state is None else state.progress,
        "ego_motion": inputs.ego_motion,
        "ego_valid": inputs.ego_valid,
        "ego_age_s": inputs.ego_age_s,
    }


def prediction_evidence(losses):
    return {name: losses[name].detach() for name in (
        "body_prediction", "body_target", "body_valid",
        "motion_prediction", "motion_target", "motion_valid",
    )}


def _motion_errors(prediction, target, valid):
    error = (prediction - target)[valid]
    if not error.shape[0]:
        return None
    return {
        "count": error.shape[0],
        "rmse_m_m_rad": error.square().mean(0).sqrt().cpu().tolist(),
        "bias_m_m_rad": error.mean(0).cpu().tolist(),
    }


@torch.no_grad()
def identification_metrics(losses, fresh):
    prediction, target, valid = (losses[name] for name in (
        "motion_prediction", "motion_target", "motion_valid"))
    body_error = (losses["body_prediction"] - losses["body_target"])[losses["body_valid"]]
    fresh_mask = valid & torch.as_tensor(fresh, device=valid.device).unsqueeze(0)
    return {
        "loss": float(losses["loss"].detach()),
        "body_loss": float(losses["body_loss"].detach()),
        "motion_loss": float(losses["motion_loss"].detach()),
        "body_rmse_checkpoint_units": None if not body_error.shape[0] else float(body_error.square().mean().sqrt()),
        "motion": _motion_errors(prediction, target, valid),
        "unseen_window_motion": _motion_errors(prediction, target, fresh_mask),
        "window_prediction_m_m_rad": prediction.cpu().tolist(),
        "window_target_m_m_rad": target.cpu().tolist(),
        "window_valid": valid.cpu().tolist(),
    }


@torch.no_grad()
def planner_metrics(origin, prediction, target, *, ctrl_dt, position_tolerance_m, heading_tolerance_rad):
    poses = predicted_poses(origin, prediction.body_delta)
    error = poses[:, 1:] - target[:, 1:]
    error[..., 2] = wrap_to_pi(error[..., 2])
    velocity = prediction.body_delta / ctrl_dt
    return {
        "position_loss": float((error[..., :2] / position_tolerance_m).square().mean()),
        "heading_loss": float((error[..., 2] / heading_tolerance_rad).square().mean()),
        "position_rmse_m": float(error[..., :2].square().mean().sqrt()),
        "heading_rmse_rad": float(error[..., 2].square().mean().sqrt()),
        "predicted_poses_m_m_rad": poses.cpu().tolist(),
        "reference_poses_m_m_rad": target.cpu().tolist(),
        "endpoint_error_m_m_rad": error[:, -1].cpu().tolist(),
        "mean_body_velocity_m_s_m_s_rad_s": velocity.mean(1).cpu().tolist(),
        "first_env_action": prediction.actions[:, 0].cpu().tolist(),
        "first_source_action": prediction.source_actions[:, 0].cpu().tolist(),
        "env_action_rms": float(prediction.actions.square().mean().sqrt()),
        "env_action_max_abs": float(prediction.actions.abs().max()),
    }


def parameter_metrics(before, after):
    changes = [after[name] - value for name, value in before.items()]
    count = sum(value.numel() for value in changes)
    squared = sum(float(value.square().sum()) for value in changes)
    return {"rms": (squared / count) ** .5,
            "l2": squared ** .5, "max_abs": max(float(value.abs().max()) for value in changes)}
