"""Regression checks for the status shared by video recording and live preview."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unilab.visualization.fada_rendering import FADAPlaybackRendering


@pytest.fixture
def rendering():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    physics = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, physics, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    session = SimpleNamespace(
        env=SimpleNamespace(
            cfg=SimpleNamespace(ctrl_dt=0.02),
            state=SimpleNamespace(terminated=np.array([False]), truncated=np.array([False])),
        ),
        last_update_stats=None,
        step_count=0,
        info={},
        physics_state=lambda: physics[None],
    )
    view = FADAPlaybackRendering(
        session, model, SimpleNamespace(camera_focus_body_name="pelvis")
    )
    return view, data


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((0, 0), "WM LoRA updates=0 | Planner LoRA updates=0"),
        ((3, 0), "WM LoRA updates=3 | Planner LoRA updates=0"),
        ((4, 2), "WM LoRA updates=4 | Planner LoRA updates=2"),
    ],
)
def test_dual_lora_status_uses_cumulative_updates_without_loss(rendering, counts, expected):
    view, data = rendering
    for step in range(1, 4):
        view.session.step_count = step
        view.session.last_update_stats = {
            "path_steps": step,
            "path_wm_updates": counts[0],
            "path_planner_updates": counts[1],
        }
        view.observe_step()
        assert view.status(data).details[2] == expected

    view.session.last_update_stats = {"path_wm_updates": 5, "path_planner_updates": 3}
    view.observe_step()
    assert view.status(data).details[2] == "WM LoRA updates=5 | Planner LoRA updates=3"


@pytest.mark.parametrize("stats", [None, {}, {"path_steps": 1}])
def test_non_learning_steps_preserve_legacy_update_count_and_loss(rendering, stats):
    view, data = rendering
    view.session.last_update_stats = {"loss": 0.5}
    view.observe_step()
    view.session.last_update_stats = stats
    view.observe_step()
    assert view.status(data).details[2] == "LoRA updates=1 loss=0.5"
    view.session.last_update_stats = {"loss": 0.25}
    view.observe_step()
    assert view.status(data).details[2] == "LoRA updates=2 loss=0.25"
