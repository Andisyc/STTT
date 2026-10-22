# Current task

Goal: 将现有在线 FADA 改为基于标称未来、时序对齐误差和固定动作快照标签的 LoRA 窗口回归，并通过现有 MuJoCo 录制入口验证链路。
Do not: 本轮仅实施已授权的步骤 4、一次聚焦集成检查和一次 0.6s 本地真实录制；不启动服务器任务、长时训练、真机或 Git 操作。不改变离线 FADA、关闭在线更新的 baseline、负重物理、观测/动作维度或 checkpoint 格式，不新增力矩监督、影子模型、变化检测或航向控制器。

Current identity:
- 仓库: `/Users/chengyuxuan/ArtiIntComVis/TTT`；入口: `scripts/play_fada_online.py`；profile: `conf/distill/recording_profile/straight_payload.yaml`。
- 正式入口已切到 OnlineAdaptationController → OnlineSnapshotCollector/OnlineSnapshotUpdater，使用标称未来条件下的固定标签回归和显式 LoRA 锚定；旧在线真实未来重建/指数遗忘实现已删除。关闭在线更新仍走原 baseline。
- `playback.py:act_projected_with_snapshot` 已提供同次前向输入、Y_nom 和首动作快照；普通推理接口行为保持不变。
- 本地实际 checkpoint: `/Users/chengyuxuan/ArtiIntComVis/FADA_Unofficial/model/plaidm_v024_close_some_dr/planner_idm_close_some_dr_v001.pt`；原 `model/v024_checkpoint` 路径已不存在。本轮正式加载确认 obs/action/command=66/29/3、H=30、K=6。服务器路径和环境仍须在实际远程运行前核对，本轮未重配环境。

Confirmed decisions:
- Planner 与 IDM 基础权重冻结，只更新 IDM LoRA；θ_base 是部署初始化完成后的固定参数副本，不假设已加载离线训练好的 LoRA。
- 在 t 保存同一次实际前向的 O_t^H、A_t^H、Y_nom(t) 和首动作 U_snapshot(t)；动作历史快照取本次动作追加之前的输入，不事后重新运行 Planner，也不复制整份网络。
- 在 t+1 用 Y_nom(t) 第一帧关节位置参考与实际下一观测计算误差。两边均使用相对同一默认姿态的弧度偏移；U 是动作空间量，不是绝对关节角。部署仍由环境转换 q_cmd = q_default + S_scale * U_exec。
- 固定标签: U_target(t) = U_snapshot(t) + gamma * (q_nominal_relative(t+1) - q_real_relative(t+1)) / S_scale；采集闭合时只算一次，保存的输入与标签均脱离计算图并独立持有，不随后续推理或优化改变。
- 主目标: 最近 T 条有效样本在保存的 Y_nom 条件下做首动作均方回归，加 (sigma/2) * ||θ - θ_base||²；不再混入旧指数遗忘权重，不将 AdamW 默认衰减当成非零锚点正则的等价实现。
- H 为推理历史长度，K 为 Planner 未来长度，T 为回归样本数，更新间隔为独立配置。T/gamma/sigma 的取值及窗口预热规则由对应 engineering-plan 明示，不作为隐含实现默认。
- 滑动窗口由 collector 单独持有，updater 消费固定 batch，更新后不清空最近 T 条样本。发生 reset 时清除待闭合样本、推理历史与训练窗口，不拼接跨回合样本；保留已学 LoRA、固定锚点及优化器状态。
- 沿用既有在线开关时序，不新增等待负重的更新门控。冻结 Planner 只是冻结映射，不保证输出在受扰状态下可达；窗口不保证 PE/自动滤波，参数锚定不保证安全或防谱坍缩，机制实现不等于控制效果已证实。
- 四步路线固定编号；每步开工前使用 engineering-plan 形成当步实现包，明确 owner、允许/禁止范围、真实依赖和最近验证；只规划时不自动实施，不额外生成计划文件或哈希。

Progress:
- [x] 旧在线推理、LoRA 更新、左肩负重和无窗口录制链路已建立，本轮已只读核对关键 owner；旧效果结论不升级为新方法证据。
  - Evidence: `playback.py:_act`、`online_controller.py:act/on_env_step`、`online_collector.py:append`、`online_updater.py:update`、`playback_distill_sessions.py:create_fada_playback_session`；历史 smoke 见现有录制与测试，本轮未重跑。
- [x] 总体路线与跨模块数据语义已记录；新路线实现进度为 4/4。
  - Evidence: 本板 Confirmed decisions 和下列四步。原板“满秩采集、指数遗忘、负重事件触发更新”不再作为新方法约束；当前实现从开局收集并在窗口可用后更新。
- [x] 步骤 1: 同次推理快照与样本接口 — Owner: `fada/playback.py`、`fada/observation.py`、`fada/online_data.py`；Depends on: 无；Done when: 可取得独立、无梯度的实际推理输入、Y_nom、首动作和关节参考，普通推理动作与历史推进保持原行为。
  - Evidence: `env PYTHONPATH=/Users/chengyuxuan/ArtiIntComVis/TTT/src uv run --frozen --no-sync python /tmp/ttt_snapshot_step1_smoke.py` → PASS，2.15s；真实小型 eval Planner-IDM 普通/快照路径动作及后续历史一致，单次前向、独立存储、无梯度、源观测/动作/后续推理/reset 隔离、样本标签独立、G1 关节坐标及 legacy action-only 路径均通过。PYTHONPATH 明确选本仓库源码，复用 uv 环境；临时脚本验证后删除，未运行测试套件、checkpoint、训练或仿真。
- [x] 步骤 2: 一步延迟标签与滑动窗口 — Owner: `fada/online_collector.py:OnlineSnapshotCollector`；Depends on: 步骤 1 接口；Done when: t+1 闭合 t 样本，标签固定，保留最近 T 条有效样本，终止/reset 不产生跨回合标签。
  - Evidence: `env PYTHONPATH=/Users/chengyuxuan/ArtiIntComVis/TTT/src uv run --frozen --no-sync python /tmp/ttt_snapshot_step2_smoke.py` → PASS，1.95s；手算 target=0.202、正负误差/向量缩放、固定标签、满窗/淘汰/读取不清空、非法配对、终止忽略重置观测并要求显式 reset、回合编号及真实小型 Planner-IDM 快照到样本链路均通过。临时脚本随后删除，未运行优化器、checkpoint、测试套件或物理仿真；本步仅修改 `online_collector.py` 与本板。
- [x] 步骤 3: 新回归损失与参数锚定 — Owner: `fada/online_updater.py:OnlineSnapshotUpdater`；Depends on: 步骤 1 样本接口；Done when: 使用保存的 Y_nom/目标做窗口均值回归和显式锚定，只有 LoRA 收到更新，标签与 θ_base 不变。
  - Evidence: `env PYTHONPATH=/Users/chengyuxuan/ArtiIntComVis/TTT/src uv run --frozen --no-sync python /tmp/ttt_snapshot_step3_smoke.py` → PASS，2.02s；真实小型 Planner-IDM、基础 dropout=0.2、LoRA dropout=0.35。手算 regression=0.003625、首动作梯度、非零初值锚定及 sigma*(parameter-base) 梯度、零修正近零 loss/梯度、单 batch 前向且不调用 Planner、eval+autograd 与异常后模式恢复、一次离线 Adam step 仅改 LoRA、固定锚点/标签及窗口重复消费均通过。临时脚本随后删除，未启动 checkpoint、训练任务、物理仿真或广泛测试；本步只修改 updater 与本板。
- [x] 步骤 4: 正式入口切换与最小录制验证 — controller/session/factory/config 已统一接通新目标，关闭更新 baseline 保留，旧在线专属实现/手动入口已清理。
  - 行为: 在同次前向保存快照，下一步闭合标签，满窗且满足更新周期时消费窗口；动作缩放读取实际 env.cfg；终止步丢弃，关闭 autoreset 时暂停采集直到显式 reset，reset 保留 LoRA/锚点/Adam。创建 wrapper/controller/session 的失败路径释放资源。
  - 配置: 正式默认 window_size=100、gamma=0.05、sigma=1e-4、update_interval=10、learning_rate=1e-4、max_grad_norm=1.0，删除 lambda_forget；启动日志明确 objective=snapshot_regression。默认 0.02s 控制步长下约 2s 首次更新，无等待负重门控。
  - Evidence: `env PYTHONPATH=/Users/chengyuxuan/ArtiIntComVis/TTT/src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync pytest tests/visualization/test_fada_online_playback.py -q` → 11 passed in 0.82s。覆盖 baseline、下一步标签/实际缩放、满窗/周期、仅 LoRA 更新、两种 reset、学习状态保留及构造失败释放；旧算法测试已迁移但未追加运行算法套件。
  - Evidence: 官方 `scripts/play_fada_online.py` + 上述真实 checkpoint + straight_payload，本地 CPU/CGL 录制 0.6s、30 步、640x360/10fps/6 帧；仅 smoke 覆盖 T=10、负重时间=0.2s。日志 0.2/0.4/0.6s 各更新一次，num_samples 均为 10，回归/锚定值有限；0.22s 挂载 3kg；进程成功退出。
  - Artifact: `/tmp/ttt_snapshot_step4_smoke.mp4`，已解码查看 H.264 视频，画面包含负重挂载提示、在线更新计数，未见 reset。短录制证明新正式链路与编码输出，不证明长时稳定、直行恢复或适应有效。
  - 清理/说明: 删除旧 OnlineTrajectoryCollector、OnlineAdaptationUpdater、scripts/test_online_adaptation.py 及旧调用，保留共享离线 windows/adaptation；两份在线测试迁移到新行为，README 增补入口、参数、坐标、reset 与旧配置移除说明。未新建环境、分支或额外计划文件。

Blockers:
- none，本次四步工程链路已完成。
- 证据边界: T/gamma/sigma 仍是试验初值；旧视频偏航和用户另报约 40s reset 尚未定因，本轮没有修复或评估这些现象，没有进行长时/服务器效果实验。

Next:
- none。长时方法效果评测或服务器运行由用户另行决定并授权。
