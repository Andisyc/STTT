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


def test_draws_shared_reference_and_measured_slam_not_robot_truth(rendering):
    import torch

    from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, ReferenceTrajectory, TrajectoryConfig
    from unilab.algos.torch.sttt.pose_data import PoseRuntimeConfig, PoseRuntimeUpdate, ReceivedPose, TimedPelvisPose
    from unilab.visualization.fada_rendering import FADADemoRenderConfig

    previous, data = rendering
    session = previous.session
    session.env.get_base_pos = lambda: np.zeros((1, 3))
    session.env.get_base_quat = lambda: np.array([[1., 0., 0., 0.]])
    session.env.get_foot_pos = lambda: np.zeros((1, 2, 3))
    config = TrajectoryConfig(length_m=2., samples=8)
    geometry = ReferenceTrajectory.generate(config, device="cpu", dtype=torch.float64)
    start = TimedPelvisPose(0, 0, 0, np.eye(4), True, (0, 0))
    end_matrix = np.eye(4)
    end_matrix[:2, 3] = [.5, .1]
    end = TimedPelvisPose(0, 0, 40_000_000, end_matrix, True, (40_000_000, 40_000_000))
    reference = MeasuredTrajectory(start, config, reference=geometry, dtype=torch.float64)
    reference.state_at(end)
    session.controller = SimpleNamespace(
        config=SimpleNamespace(pose=PoseRuntimeConfig(odometry_source="cuvslam", slam_enabled=True)),
        reference_geometry=geometry, collector=SimpleNamespace(reference=reference),
        last_pose_update=PoseRuntimeUpdate(
            (), "tracking", 0, 0, 0, False,
            (ReceivedPose(start, 40_000_000), ReceivedPose(end, 40_000_000)),
        ),
    )
    view = FADAPlaybackRendering(
        session, previous.model, previous.args, FADADemoRenderConfig(enabled=True),
    )
    view.observe_step()
    view.observe_step()  # Repeated display reads cannot manufacture new measurements.
    np.testing.assert_allclose(view.measured_path[-1], [.5, .1, .03], atol=1e-12)
    np.testing.assert_allclose(view.reference_route[:, :2], geometry.xy[0].numpy(), atol=1e-12)
    data.xpos[view.focus_body_id] = [100., 200., 300.]
    assert view.status(data).details[0] == "Pose: cuVSLAM SLAM | tracking | measurements=2"
    assert view.status(data).details[1] == "Measured lateral=+0.100m yaw=+0.00deg"
    scene = mujoco.MjvScene(previous.model, maxgeom=100)
    view.append_scene(scene, data)
    measured = [geom for geom in scene.geoms[:scene.ngeom] if geom.rgba[1] > .8]
    assert len(measured) == 1
    np.testing.assert_allclose(measured[0].pos, [.25, .05, .03], atol=1e-7)
