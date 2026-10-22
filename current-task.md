# Current task

Goal: 将新在线目标的调度改为默认首次等待 40 个有效控制步、此后每 10 步使用 10 条新样本更新并清空本轮训练缓存，保持学习公式及已学状态不变。
Do not: 不调整 gamma/sigma/学习率/梯度裁剪或损失归一化，不恢复旧监督目标，不改网络/观测/checkpoint、负重、终止策略、环境、服务器或 Git 状态。不启动新仿真或广泛测试。

Current identity:
- 仓库: `/Users/chengyuxuan/ArtiIntComVis/TTT`；入口: `scripts/play_fada_online.py`；profile: `conf/distill/recording_profile/straight_payload.yaml`。
- 正式 collector/controller/factory/config 已切换为 T=10、warmup_steps=40、update_interval=10、成功后清批；唯一离线 smoke 已通过，未启动物理仿真。
- `playback.py:act_projected_with_snapshot` 已提供同次前向输入、Y_nom 和首动作快照；普通推理接口行为保持不变。
- 本地实际 checkpoint: `/Users/chengyuxuan/ArtiIntComVis/FADA_Unofficial/model/plaidm_v024_close_some_dr/planner_idm_close_some_dr_v001.pt`；原 `model/v024_checkpoint` 路径已不存在。本轮正式加载确认 obs/action/command=66/29/3、H=30、K=6。服务器路径和环境仍须在实际远程运行前核对，本轮未重配环境。

Confirmed decisions:
- Planner 与 IDM 基础权重冻结，只更新 IDM LoRA；θ_base 是部署初始化完成后的固定参数副本，不假设已加载离线训练好的 LoRA。
- 在 t 保存同一次实际前向的 O_t^H、A_t^H、Y_nom(t) 和首动作 U_snapshot(t)；动作历史快照取本次动作追加之前的输入，不事后重新运行 Planner，也不复制整份网络。
- 在 t+1 用 Y_nom(t) 第一帧关节位置参考与实际下一观测计算误差。两边均使用相对同一默认姿态的弧度偏移；U 是动作空间量，不是绝对关节角。部署仍由环境转换 q_cmd = q_default + S_scale * U_exec。
- 固定标签: U_target(t) = U_snapshot(t) + gamma * (q_nominal_relative(t+1) - q_real_relative(t+1)) / S_scale；采集闭合时只算一次，保存的输入与标签均脱离计算图并独立持有，不随后续推理或优化改变。
- 保持新目标公式: 保存的 Y_nom 条件下拟合固定动作快照伪标签，加显式 LoRA 部署参数锚定；不改变动作维求和、窗口样本均值、eval+autograd 或 Adam 设置。
- 本次计划默认 window_size=10、warmup_steps=40、update_interval=10；H=30、K=6 不变。warmup 是每个 episode 的有效控制步阈值，不是墙钟时间；gamma=0.05、sigma=1e-4、learning_rate=1e-4、max_grad_norm=1.0 保持当前值。
- collector 仍单独拥有训练样本，成功 update 后仅清除本轮完成样本。不能用 reset 代替清批；不清推理历史、pending、episode_id/timestep、LoRA、固定锚点、Adam 或 controller 有效步计数。真正 episode reset 仍按既有契约清采集状态并重新计算 warmup，保留已学参数/优化器。
- 沿用既有在线开关时序，不新增等待负重的更新门控。冻结 Planner 只是冻结映射，不保证输出在受扰状态下可达；窗口不保证 PE/自动滤波，参数锚定不保证安全或防谱坍缩，机制实现不等于控制效果已证实。
- 原四步链路工程已完成。本次是独立的局部调度调整，不重新编号旧路线，不把新规划或旧 smoke 当作当前行为已经修改的证据。

Progress:
- [x] 原四步快照/标签/回归/正式入口工程已完成；此状态只证明旧调度链路接通，不证明新方法控制有效。
  - Evidence: `tests/visualization/test_fada_online_playback.py` 曾 11 passed in 0.82s；真实 checkpoint 本地 CPU 录制 0.6s/30 步/6 帧，输出 `/tmp/ttt_snapshot_step4_smoke.mp4`，3 次更新、0.22s 挂载 3kg；这些检查在本次修改前完成，本轮不重跑。
- [x] 调度调整的具体实现包已规划，用户已授权按包实施。
  - Change: `fada/online_collector.py:OnlineSnapshotCollector` 新增 `clear_window()`，只清 `_samples`；保留 get_window 的只读语义与 reset 的回合语义。`fada/online_controller.py:FADAOnlineAdaptationConfig/OnlineAdaptationController.__init__/on_env_step` 增加有限非负整数 warmup_steps，默认 40，window_size 默认改 10；保存 warmup 值，reset 继续将有效步数归零。
  - Schedule: 每次成功 complete 后增加有效步数；仅在 step_count>=warmup_steps、step_count%update_interval==0、collector.ready 同时成立时执行一次 update。默认在第 40/50/60 步更新，每批 10 条；首次消费启动阶段最后 10 条闭合样本，之后每批只含上次成功更新后的新样本。非默认 T/周期组合仍遵守满批+阈值+周期三个条件，不承诺任意配置都每 0.2s 更新。
  - Consumption: `stats=updater.update(collector.get_window())` 成功返回后调用 clear_window 再发布 stats；异常不清批、不吞异常、不自动重试。清批不触发 collector.reset，不重新初始化 θ_base/Adam，也不改变 sample timestep 或动作历史。
  - Callers/config: `visualization/playback_distill_sessions.py:create_fada_playback_session` 显式传递并记录 warmup_steps；`conf/distill/config.yaml:online_adaptation` 改为 window_size=10、warmup_steps=40、update_interval=10。保持 enabled=false baseline、profile、payload 事件和录制入口不变，不新增 clear_after_update 模式开关或旧行为兼容路径。
  - Preserve/forbidden: 不改 online_updater、online_data 的学习公式与标签静态性，不改 LoRA dropout/梯度/裁剪、设备、物理 reset 和已学参数保留规则；不以本次调度变更声称修复无负重失稳。
  - Nearest proof: 实施后只运行一次轻量离线 smoke，真实小型 Planner-IDM 经正式 session/factory 接入，验证第 39 步无更新、第 40/50/60 步各更新一次且每次 10 条、成功后窗口空、下一批不复用旧样本、清批不推进 episode/不重置历史/参数锚点或 Adam、真正 reset 后 warmup 重计，以及一次非默认 warmup 配置确实穿过 factory。异常保留缓存的检查使用真实 updater 的无效数据错误边界，不引入替代更新器。无需 checkpoint、MuJoCo、视频、长训练或广泛测试。
  - Affected tests/docs: 迁移 `tests/visualization/test_fada_online_playback.py` 对更新后保留缓存的断言与依赖它的 reset 取样；小规模既有用例显式设置 warmup_steps=0，新增/调整时序与清批行为检查。`tests/algos/test_fada_online_adaptation.py` 只补 clear_window 与 reset 的区别边界；`README.md` 更新首次/后续时间及每批清空说明，不重写其他研究文档。实现范围仅上述文件与本板。
- [x] 实施调度调整并通过唯一离线 smoke。
  - Evidence: `PYTHONPATH=.../TTT/src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python -c ...` 调用现有小型真实 Planner-IDM 与正式 session/factory，7 个针对场景全部通过，1.79s (命令墙钟 2.20s)。默认第 40/50/60 步更新、每次 10 条、后续新批不重复；非默认 warmup=17 在第 20/30/40 步更新。验证清批保留历史/episode/步号/锚点/Adam，reset 后重新 warmup 且保留已学参数，真实 updater 非有限 loss 异常保留本批且不更新参数。复核既有因果标签、冻结权重及两种 autoreset 生命周期；无真实 checkpoint、MuJoCo、训练或广泛测试。
  - Changed owners: online_collector、online_controller、正式 session factory、distill config；迁移两处现有测试文件并更新 README。未改目标公式、gamma/sigma/LR、负重、物理终止或服务器环境。

Blockers:
- 无工程阻塞；本次调度修改已完成。
- 方法效果仍未定因: `video/left_shoulder_snapshot_ttt.mp4` 的 25s 画面无负重、93 次更新、7 次 reset；旧 `straight_online_no_payload_v3.mp4` 同时刻 122 次更新、0 次 reset。调度对齐用于减少实验差异，不代表已证明新监督目标正确或摔倒已解决。

Next:
- none；本次授权范围已完成，方法的长时行走效果尚未验证。
