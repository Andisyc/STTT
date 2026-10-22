# Current task

Goal: 将在线微调从单步关节误差标签切换为配对 K 步标称/真实状态轨迹经冻结基座 IDM 差分生成的静态动作标签，保留在线 LoRA 累积补偿。
Do not: 工程四步已交付；本轮仅修改 README 和本板并做 shell 语法解析，未运行启动命令/模型验证/仿真。不新增脚本、配置开关、依赖、环境或 checkpoint 格式，不调参数，不连接服务器或操作 Git。后续真实录制需另行授权。

Current identity:
- 仓库: `/Users/chengyuxuan/ArtiIntComVis/TTT`；正式入口: `scripts/play_fada_online.py`；profile: `conf/distill/recording_profile/straight_payload.yaml`。
- 第 1–4 步工程交付完成：正式 controller 已使用 K 步轨迹采集与冻结基座差分标签；旧单步 collector、关节误差标签及专用 helper 已移除。正式 factory 标识 objective=expert_trajectory_difference；README 已提供两组服务器后台录制命令。证据为离线真实网络/session 链路及命令语法，不是负重效果证明。
- 默认 H=30、K=6、obs/action/command=66/29/3；T=10、warmup_steps=40、update_interval=10；有效控制周期 0.02s。
- 本地已确认 checkpoint 路径: `/Users/chengyuxuan/ArtiIntComVis/FADA_Unofficial/model/plaidm_v024_close_some_dr/planner_idm_close_some_dr_v001.pt`。本轮不加载，服务器路径不视为已核实。

Confirmed decisions:
- 方法来源为用户提供的最新架构定案：Planner 冻结；冻结基座只翻译配对轨迹差，执行网络只更新 IDM LoRA；在线训练/部署均以保存的 Y_nom 为未来条件。
- 同一原点 τ 的样本必须保存实际推理的 O_hist、A_hist、command、Y_nom、U_snapshot；不重新运行 Planner 代替快照。Y_exec 是 τ+1..τ+K 的完整 66 维实际状态，不是控制动作或仅 29 维关节角。
- 标签为 `delta = First(base(X,Y_nom)) - First(base(X,Y_exec))`；`U_target = SG(U_snapshot + gamma*delta)`。按用户定案允许的“计算时固化”选择：在实际更新批次形成时，对成熟 Pair 一次性生成独立持有的固定标签；同批 compute_loss/update 不重新生成。基座、历史、轨迹、快照均固定，因此不以优化中的在线输出替换标签基底。删除标签中的关节切片/角度误差/除 action_scale，不删除环境执行端的 action_scale。
- 规划采用最小 cutover：冻结基座固定为本次启动 checkpoint 的 IDM 有效映射，在注入在线 LoRA 前独立复制，只复制 IDM，不重复复制 Planner；eval、无梯度、无优化器参数或可变权重存储共享。theta_base 仍取在线 LoRA 初始化后的固定参数副本，在线更新不得改变基座。
- 接口先冻结：历史 [1,H,66]/[1,H,29]，Y_nom 与 Y_exec 均为 [1,K,66]，U_snapshot/U_target 均为 [1,29]，同序、同观测投影、同 dtype/device、同 episode 和起点 τ；跨 reset/终止片段丢弃，沿用已有 causal-window 对命令切换的语义边界。
- K 是每条未来轨迹长度，T 是成熟训练样本数；片段最早在 τ+K 完成。原点步号不改记为成熟时刻。保存完整快照历史，不额外等待重新积累 H 步才开始记录。
- 保持当前控制步 warmup=40、interval=10、成熟训练批 T=10；仅在阈值/周期/满批三条件成立时更新。成功仅清成熟训练批，不能清尚在等待未来的快照/观测环或推理历史；更新异常不消费本批。默认首次更新仍可在控制步 40。
- 真正 reset 清推理历史、未成熟轨迹、成熟训练批并重新 warmup；保留已学在线 LoRA、theta_base、Adam 与冻结基座。终止观测可能已被环境重置，不能作为轨迹末帧。
- 保留 gamma=0.05、sigma=1e-4、lr=1e-4、max_grad_norm=1、动作维求和/样本均值、eval+autograd 和 Adam(weight_decay=0)。每批仍只做一次参数更新，不混入重复回放。
- 差分是基座估计的等效动作修正，不是实测力矩缺口；冻结算子不冻结输入轨迹，不保证消除所有驻点或抹平倾角。方法效果留给后续授权实验，不阻断当前工程规划。

Progress:
- [x] 既有单步链路与调度调整完成，但未证明方法有效。
  - Evidence: 旧局部离线 smoke 7 个针对场景通过；最新 `video/left_shoulder_ttt_t10_w40.mp4` 在 6s 已 27 次更新且 yaw +98.8deg，7s 出现首次 reset。此证据属于旧标签方案，不迁移为新方案成功证据。
- [x] source-based 总体路线共 4 步，工程交付完成 4/4；真实负重效果未验证。
  - Evidence: 总体规划时核对了 collector、controller、updater、数据类型、正式 factory 和离线 causal-window 语义；确认旧单步 collector 与新 K 步配对需在第 3 步切换消费者，动作回归/锚定形式可复用。
- [x] 1. 冻结基座与数据接口 — 已实现并通过唯一离线 smoke。
  - Changed: `online_controller.py` 在 collector 参数检查后、updater 注入 LoRA 前创建独立 `base_idm`，eval/关闭梯度，只复制 IDM；reset 保留实例。没有修改 optimizer、theta_base、collector 或 factory。
  - Data: `online_data.py:FADATrajectoryPair(snapshot, realized_future, episode_id, timestep)` 保留已拥有的 snapshot，真实未来 detach+clone；检查 [1,K,66] 同形、dtype/device、有限值与非负整数身份。现有两个数据类型及单步训练接口未变。
  - Evidence: `PYTHONPATH=.../TTT/src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python -c ...`，单次离线 smoke 全部通过，1.54s (墙钟 1.92s)。正式 factory/session + 小型真实 Planner-IDM 初始输出一致；真实 LoRA 更新后在线参数变化，基座无权重/buffer 共享、未进 Adam、无梯度且参数/buffer/固定输入输出不变；reset 保留基座/在线参数/锚点/Adam。用同一 session 保存的快照及对应 K 帧观测构造 Pair，验证源张量修改隔离；另验证图脱离、9 个形状/dtype/device/finite/身份错误边界，以及 disabled baseline 行为。未加载真实 checkpoint、未运行 MuJoCo 或广泛套件。
  - Regression: 在两处既有在线测试文件中保留基座隔离/重置和轨迹持有权/错误边界行为用例；本轮仅执行上述单次 smoke。新基座尚不生成标签，K 步正式采集由第 2 步接入，双路动作差分由第 3 步接入。
- [x] 2. K 步真实轨迹采集与配对 — 已实现并通过针对性离线 smoke。
  - Changed: `online_collector.py` 新增 OnlineTrajectoryCollector，复用 FADATrajectoryPair，输出成熟的完整轨迹对，不生成标签。机械提取 `_validate_snapshot` 供新旧 collector 共用，未改旧公开行为；对应行为用例加入既有 `tests/algos/test_fada_online_adaptation.py`。
  - Lifecycle: 第 K 次有效 complete 才关闭原点 0，随后滑动逐步配对；观测环每帧独立持有，成熟窗口最多 T 条。clear_window 只清成熟窗口，保留 pending/在途片段；命令切换丢旧在途片段但保留有效成熟样本；终止优先忽略终止观测并暂停直至 reset，reset 清所有采集状态并推进 episode。
  - Evidence: `PYTHONPATH=.../TTT/src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python -c ...`，同一针对性 smoke 修正测试辅助函数遗漏必需的 device 后复跑，7 个场景通过，0.96s (墙钟 1.25s)。真实小型 Planner-IDM 快照验证 K=1/6 精确帧号配对、成熟队列限长、清批保留 pending/在途、来源观测复用隔离、命令切换、终止/手动 reset、非法调用不消费有效步；同次检查旧 collector 的终止生命周期和有符号 action_scale 标签行为。首次失败在模型辅助构造处，尚未进入新 collector；补 `device="cpu"` 后没有扩大范围。未加载 checkpoint、运行 MuJoCo、训练或广泛套件。
  - Preserve: 正式 controller/factory、基座、数据类型、损失、调度、配置和负重/录制均未改。本步完成独立采集 owner，第 3 步再切换正式绑定并移除旧路径，不宣称新算法已经运行。
- [x] 3. 基座差分标签与更新生命周期接入 — 已实现，针对性离线 smoke 全部通过。
  - Changed: online_updater 新增 no_grad 的 build_expert_samples，共用原始历史双路冻结基座，按 nom-exec 差分叠加保存的在线快照；online_data 的 FADAOnlineSample 只持 trajectory/target，目标独立持有且无梯度。compute_loss 仍输入 Y_nom，原回归范数/锚定/Adam 设置不变。
  - Cutover: controller 改用 OnlineTrajectoryCollector，删除 action_scale 入参；更新前构造一次固定标签批，成功后才清成熟队列。旧 OnlineSnapshotCollector、joint_position_error、snapshot.nominal_joint_positions、专用角度切片 helper 和旧用例已移除，无 alias/旧模式。factory 同步删除传参并更新 objective/K/T 日志；README 方法说明同步。环境动作缩放、离线 FADA、网络、配置、负重/录制、服务器均未改。
  - Evidence A: 首次 `uv run --frozen --no-sync python -c ...` 的解析专家方向/幅度与快照保留、Y_nom 条件回归/首动作梯度、非零 LoRA 初始锚定、K=1/6 因果配对、disabled baseline、真实 LoRA 更新后基座隔离均通过。随后一条迁移测试错误地要求 K=6/T=3 的第 7 步满批；实际只成熟 2 条，修改为核对原点 [0,1] 后再于第 8 步更新。未改算法。
  - Evidence B: 修正后仅重跑失败及尚未执行的 smoke 场景，1.16s (墙钟 1.46s) 全部通过；已通过的 A 项不重跑。正式 factory/session + 真实小型 Planner-IDM(K=6) 完成默认 40/50/60 三次真实 Adam 更新、warmup=17 的独立时序、更新后固定标签不变/重新基座计算不受在线参数影响、失败保留批次且不改参数、reset 重计 warmup并保留学习状态、两种 autoreset、初始化失败资源释放及建环境前配置拒绝。两次执行共覆盖 19 个针对场景，无真实 checkpoint、MuJoCo、训练 campaign、广泛套件或独立审查。
  - Preserve: collector 仍独占成熟 Pair 窗口，controller/updater 无第二份持久缓存；成功清批保留在途片段，基座/theta_base/Adam 不随 reset 重建。gamma/sigma/LR/裁剪/损失归约未调。所有结果只证明定义及工程数据流，不证明挂重物后能恢复行走。
- [x] 4. 正式入口与启动交付 — README 的 Linux/CUDA 后台录制说明已完成。
  - Changed: 新方法与关闭在线更新的负重 baseline 各有完整多行 nohup 命令；同 checkpoint、左肩 3kg 从 0 秒挂载、固定 0.4m/s、侧后跟随、录制 120 秒。显式 T=10/warmup=40/interval=10；时间戳区分视频/日志，提供 tail -f 和最终 Video saved 判据，并说明 reset/更新次数不等于效果。
  - Evidence: 使用现有 uv 环境读取 README 新章节的四个 bash 代码块，只调用一次 bash -n，输出 `PASS bash -n: 2 launch commands and 2 log-view commands parsed; no documented commands executed`。第 3 步的 19 个离线场景未重跑。
  - Scope: 仅 README/本板变更；入口、算法、配置、测试和环境均未修改，未新增脚本。示例服务器路径未远程核验，未加载真实 checkpoint，未启动任何录制；已写明 uv/EGL/进程级 preload 的适用边界及实际效果待验证。

Blockers:
- 本次工程交付无阻塞。
- 证据边界：真实 checkpoint 下差分方向/幅度和负重适应效果尚未验证，属于后续另行授权的实验，不能用语法解析或离线接通替代。

Next:
- none。本次工程任务已关闭；后续服务器录制与负重效果评估不在本轮执行范围。
