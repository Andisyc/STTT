# ICE-Cal

ICE-Cal is a research repository for **In-Context Execution Calibration**: adapting a frozen
planner/execution stack from observable Support trajectories without online parameter updates.
The runtime is derived from UniLab, but this repository documents only the ICE-Cal research
question, its contracts, implementation evidence, and current design transition.

## 维护仓库前必须读取

1. [`AGENTS.md`](AGENTS.md)：环境、配置、Backend、资产、Sim2Sim 和 Git 的硬性契约。
2. [`note/governance/README.md`](note/governance/README.md)：当前三个研究轨道的治理游标与权限边界。
3. [`note/README.md`](note/README.md)：研究文档总索引和权威读取顺序。
4. [`note/fada/contracts/README.md`](note/fada/contracts/README.md)：当前有效及历史 FADA Contract。
5. [`note/architecture/08_in_context_execution_calibration.html`](note/architecture/08_in_context_execution_calibration.html)
   与 [`09_in_context_execution_calibration_design_inspector.html`](note/architecture/09_in_context_execution_calibration_design_inspector.html)：
   当前 Concept Figure 与 Design Inspector。
6. [`CONTRIBUTING.md`](CONTRIBUTING.md) 与 [`docs/README.md`](docs/README.md)：开发环境、提交规范、稳定工程说明和 Runbook 入口。

维护时必须使用 `uv run`；`scripts/` 只组装流程；任务、Reward 和 Backend 优先由 Hydra
配置及 owner 模块表达。训练、仿真、部署、Git 发布和研究语义激活都需要各自明确授权。
测试、静态检查或文档状态不能单独证明策略质量。

## Spatial TTT 实现状态

上一版“冻结动作世界模型、空间损失更新 IDM LoRA”的空间方案已退役。
旧空间采集、训练、动作探针、在线更新入口及其配置和模型加载器不再提供；
历史数据与 checkpoint 保留，但不代表可由当前代码继续使用。
新版方法以 [`METHOD_DESIGN_SLAM_TTT.md`](METHOD_DESIGN_SLAM_TTT.md) 为准。
前六步已接通运行期与有限评估工具；目前证据仅覆盖 CPU 离线验收，真实闭环效果仍待实测。
普通 Planner/Tracker、独立执行层在线适应、几何计算与下述视觉定位诊断仍保留。

`create_fada_playback_session(..., future_transform=callback)` 仅支持 official
Planner-IDM checkpoint 的 policy 模式；不传回调时保持普通推理。回调接收
`FADAFutureQuery`，其中历史、7D 命令、原始绝对 Planner future 为借用只读张量，
source joint order、`ctrl_dt` 与 preprocessing 均由实际源 profile / checkpoint 绑定。
回调必须返回相同 shape、dtype、device 的有限 future，不得原地修改输入。

`sttt.path_following.compose_joint_position_residual` 接受 `[B,K,64]` future
和 source order 的 `[B,29]` 物理弧度残差；调用方必须显式传入 profile 的
`ctrl_dt` 和 preprocessing 的 `term_scale.dof_pos/dof_vel`，不猜默认值、不做裁剪。
仅第一未来帧的 `[3:32]` 关节位置叠加残差，速度 `[32:61]` 叠加其有限差分：
第一帧加 `s_v * residual / ctrl_dt`，存在第二帧时减去同一增量。
其他关节位置、基座角速度和投影重力保持 Planner 预测；零残差完整恢复原输出。

回调可用 `InterventionQueryRecord.from_query(query, adapted_future, residual_rad)`
保存独立、脱图的查询记录，保留原始与合成 future、残差及解释元数据。
公共 `FADAInferenceSnapshot` 结构不变，其 `nominal_future` 始终是实际交给 Tracker
的合成目标，既有执行层差分标签与更新公式不变。查询记录不代表已执行窗口或三维实测样本；
新版 API 与播放入口见下文；旧 `spatial_adaptation` 配置仍拒绝，不兼容退役模型。

`unilab.algos.torch.sttt.path_following.collection_runtime.collect_path_following_data`
是新版仿真预训练采集 API。传入已 compose 的 Hydra `cfg`，显式提供 `root_dir`、
`output_dir`、`num_steps`、`window_steps`、`residual_amplitude_rad`（物理弧度）和 `seed`
（仅控制本地残差随机流）；`shard_size` 默认 256。窗口长度不等于 Planner 预测长度。
每窗独立均匀采样 29 维残差，窗内逐次重规划复用锁定值，候选只在下一边界生效。

调用前须配置 `task=g1_walk_flat/mujoco`、真实 official checkpoint 及其绑定 profile，
并显式关闭命令重采样（`env.commands.resampling_time=0`）、heading command、键盘和 demo。
初始速度指令须为固定 `[vx > 0, 0, 0]`；采集器只检查，不改写速度或冻结步态相位。
采集私有配置关闭执行层在线更新，Planner/Tracker 全程冻结，调用方配置保持不变。
当前仅使用 simulation 真值位姿，不打开相机或启用 cuVSLAM。

每窗保存首个实际执行查询、锁定残差、逐步实际 env-input 动作及物理时间、端点位姿
与插值来源、起点水平航向系 `[dx,dy,dyaw]`（米/米/弧度）和固定初始参考线误差。
延迟位姿按历史控制编号配对，不影响窗口边界；缺步、超时、终止和半窗不生成有效样本。
定位身份改变即终止本轮，不用当前位置重建参考线。

输出为 `source.json` 与 `windows_000000.npz` 等数值分片，格式标识为
`sttt_path_following_windows`、版本 1。元数据绑定 checkpoint 哈希、source/target
关节顺序、预处理尺度和窗口参数；`capture_summary` 记录有效样本与丢弃窗口计数。
`path_following.dataset.iter_path_following_samples` 逐分片读取，
`read_path_following_metadata` 读取解释信息；拒绝旧格式、未完成采集、混合窗口契约
及与位姿不符的标签。非空输出目录不覆盖；只有有效完整窗口才可完成数据集。

### B 离线训练与响应诊断

`path_following.training.train_response_model(cfg)` 接受独立的
`conf/sttt/response_model.yaml` 配置；显式提供 `data.train_dirs`、
`data.validation_dirs` 和新的 `output_checkpoint`。训练与验证必须是独立完整采集组，
不能随机拆开相邻窗口或复制同一份数据冒充验证集；相同路径及完全复制的分片会被拒绝。
窗口步数、控制周期、H/K、source 关节顺序、基础策略身份和预处理/控制契约必须一致；
负载、摩擦等采集条件保留在来源信息中，不作为模型兼容性字段冻结。

`features.pack_response_context` 只拼接起点历史观测、动作历史、有效掩码、7D 命令
和原始 Planner future；不输入合成 future、实测增量、路径误差或执行后动作。
`model.ResponseModel` 用两层 SiLU MLP，将上下文和独立的 29 维物理弧度残差
映射为 `[dx,dy,dyaw]`。H/K 从数据读取，隐藏层宽度等超参数走 YAML。
归一化统计只拟合训练集有效历史，填充保持屏蔽；前向使用固定统计量的可微运算，
不调用共享归一化器带 `no_grad` 的前向。输出保持米/米/弧度，损失为
`0.5 * mean(sum((prediction - measured_delta)^2, axis=channels))`，只更新 B。
全零、固定或缺少任一残差通道变化的数据不能作为 29 维方向预训练数据。

训练按有界缓冲加载与打乱样本，不一次加载全部轨迹，也不创建环境或执行 Planner/Tracker。
B 产物格式为 `sttt_path_following_response_model`、版本 1，独立保存网络、输入统计、
残差覆盖、兼容契约、数据来源及训练/验证结果；原子写入且不覆盖已有产物。
`checkpoint.load_response_checkpoint(path, expected_contract=contract)` 返回的
`.model` 用于推理、`.contract` 用于冷路径绑定、`.metadata` 保留训练信息；
只接受本格式，按 `weights_only=True` 加载，不恢复旧模型或训练优化器。

`diagnostics.evaluate_response_model` 报告独立验证集三通道 RMSE/MAE；
`diagnose_response_jacobian` 冻结 B 参数后，比较对物理弧度残差的自动微分与中心差分，
并标记探针是否超出训练残差范围。该检查包含归一化的梯度链，**不是物理干预方向证明**；
报告中的 `physical_direction_verified` 始终为 `false`，不能凭低预测误差或导数自洽
宣称真实响应方向已验证。该离线训练入口不更新 A，也不提供在线控制或训练恢复。

`scripts/diagnose_response_residuals.py` 提供纯离线的残差输入有效性诊断：固定同一份 B、
起点上下文和实际增量标签，只替换送入 B 的残差为原值、全零或打乱值。
不修改数据中的 residual/adapted future，不训练、不拟合归一化、不创建环境。

```bash
uv run python scripts/diagnose_response_residuals.py checkpoint_path=/path/to/response_b.pt
```

配置为 `conf/sttt/response_residual_diagnostics.yaml`，默认 CPU、batch size 128、
固定种子 0 和 5 次置换。`validation_dirs=[]` 自动读取 B 产物中记录的验证路径，
核对分片内容指纹；显式指定新验证集时仍检查响应契约，并拒绝训练目录、复制的训练数据、
重复输入或已改变的原验证集。没有训练/验证来源记录的 B 产物不能冒充可验证的独立诊断。

每次置换在同一采集的 sequence/epoch/frame 分组内交换完整 29 维向量，
使用 Sattolo 单环算法保证没有窗口身份固定点，不宣称均匀采样所有无固定点置换。
不足两个窗口的分组明确拒绝；全组置换与推理 batch size 无关。
只缓存残差和窗口身份，历史与标签分片按批重读；重复残差导致输入仍相同的比例单独报告。

输出默认是 B 同目录的 `residual_diagnostics.json`，可由 `output_path` 指定，已有文件不覆盖。
报告保存模型/采集指纹、来源条件、分组、置换设置及映射哈希，
并分别给出总体与逐采集的 RMSE/MAE，以及 `ablation RMSE - correct RMSE`。
三通道始终使用米/米/弧度；打乱结果包含各次指标及均值/范围。正差值只代表当前数据上的
正确残差配对有预测收益，接近零不能单独断言完全没学到响应；5 次置换不是 5 次独立实验。
`zero_inside_training_residual_bounds` 标记零残差是否处于预训练残差范围。
此诊断不验证物理干预方向或跨条件泛化，`physical_direction_verified` 始终为 `false`。

### 最小同起点响应探针

`scripts/probe_response_directions.py` 调用 `path_following.response_probe.run_response_direction_probe`，
配置为 `conf/sttt/response_direction_probe.yaml`。这是需要显式启动的有限仿真实验，
不是自动测试或重新训练：固定一个起点、一条随机 29 维方向，执行
`zero_a -> plus -> minus -> zero_b` 四个分支；中间正负顺序可交换，但不能增加分支或重试挑选。
探针任务配置由独立加载器组合，不初始化、清空或替换调用方的全局 Hydra。
同一 API 支持普通 Python 调用与已进入 `@hydra.main` 的 CLI，原启动参数保持不变。

每个分支新建独立的普通 official 播放 session，以相同种子重复零残差热身；
不新增快照恢复框架、不修改后端，不启用 A、OAT、路径纠偏运行期、相机或视频。
环境必须是固定直行、无额外负载、无噪声/随机扰动的 G1 MuJoCo 条件，P/T/B 权重均冻结。
热身动作与位姿、原生高精度起点状态、相位/命令及首个真实推理上下文均逐字段核对；
相同种子不等于检查通过。原生状态只通过现有公开快照接口读取一次，不执行恢复，
也不使用 float32 视频快照。任何起点或上下文不一致即停止本组，不自动放宽检查。

显式提供 policy/profile/B/输出路径、`residual_amplitude_rad`、
`replay_position_tolerance_m`、`replay_heading_tolerance_rad`、`minimum_effect_m/rad`；
默认 CPU、种子 0、每分支热身 50 步。正负残差和零残差都必须位于 B 的训练覆盖范围，
不自动裁剪或扩大。窗口长度和控制周期取 B/实际 profile 契约，而不是另设预测长度。
当前模型若窗口为 10 步、周期 0.02 秒，四个分支共执行 `4 * (warmup_steps + 10)`
个控制步，只有每分支末尾 0.2 秒施加对应锁定残差，逐次重规划仍复用该值。

输出目录非空时拒绝覆盖；`source.json` 保存模型身份与配置，四个分支分别保存
数值 NPZ（物理时间、轨迹、实际 env-input 动作、残差、起点核对字段和首查询上下文），
`summary.json` 保存实际增量、B 预测和正负差异。异常/终止保留已观察的部分数据并停止后续分支，
不把失败结果替换为新起点。每个 session 在完成或异常后都关闭。

两次零残差的窗口轨迹用于重复性检查；超过显式位置/航向容差时，本组 `valid=false`。
比较实际与预测的 `plus - minus` 三通道增量，航向差取主值；实际差异未超过
最小效果阈值和零重复差异时，标为 `unresolved_effect`，不判 B 对错。其他通道分别标记
`same_sign`、`opposite_sign` 或 `prediction_negligible`，同时保留幅度误差。
符号一致不代表幅度准确、全状态雅可比正确或闭环稳定；结论只属于本组有限幅度仿真对照。
CLI 在分支未完成或重复性无效时返回非零；工具的 CPU 合成 I/O 验收不代替真实仿真结果。

### A 零启动与 A/B 分离更新核心

`residual_model.ResidualModel` 将测量误差 `[e_y,e_psi]` 和起点上下文映射为
source order 的 29 维物理弧度残差。隐藏层小方差随机初始化，输出层权重/偏置为零；
不施加手写方向、目标转速、输出裁剪或额外残差惩罚。A 只复制 B 的固定上下文统计，
不会把 B 注册为子模块。共享的可微归一化保持 B 原有前向语义和 checkpoint state keys。

`online_learning.PathFollowingUpdater(loaded_b, cfg)` 独占加载后的 B 及新建的 A；
配置使用 `conf/sttt/path_following_online.yaml`，两个 AdamW 优化器均无权重衰减。
`update(completed_windows, actor_context=None)` 每次最多更新 B、A 各一步，顺序为 B 再 A。
B 只用已执行残差和实测三维增量；A 重新计算候选，通过冻结权重的 B 和
`objectives.actor_loss` 反传旋转合成的新路径误差，不切断候选梯度。B 的统计与预训练覆盖记录不重新拟合。

初始只更新 B；成功使用两个有效零残差窗口（可配置）后才允许更新 A。
重复/逆序窗口在优化前拒绝，未成功使用的窗口不计校准数；零残差校准不证明响应方向。
缺少合法 A 观测时 B 仍可更新，A loss 返回 `None` 并给出跳过原因，不把缺测误差补零。
`ActorUpdateContext.from_aligned_query` 要求查询物理时刻和位姿时刻一致、定位身份与参考线一致；
`from_window_start` 明确保留历史起点时间，不能把它冒充当前观测。

调用方负责按更新间隔 T 触发、选择足够新鲜的配对观测，以及下一窗口边界的候选激活；
本核心不轮询定位、不保存待处理窗口，也不提交或修改执行中的锁定残差。
切换任务需显式 `begin_episode(new_reference)`，测量序列/epoch 必须推进；
时序标记重置但模型及优化器保留，未完成的启动校准重新计数，已完成后不重复冷启动。
调用方应丢弃旧任务待用样本，更新器拒绝跨参考线或定位身份的数据。

A 阶段即使异常也恢复 B 的模式/梯度开关并清除残留梯度；异常直接上抛，不自动回滚或重试。
若 B 已成功更新，随后 A 失败，B 的样本消费仍保留，不能重放整个批次。
该学习核心不直接控制机器人，也不提供 A 离线预训练或 A/B 联合状态恢复；实际接线由下述运行期承担。

### 路径纠偏运行期

现有 `scripts/play_fada_online.py` 通过 `path_following=online` 显式开启；
不选择该配置时，普通播放和独立执行层在线适应保持原路径。入口示例：

```bash
uv run python scripts/play_fada_online.py \
  task=g1_walk_flat/mujoco recording_profile=straight_payload \
  env.runtime_payload.enabled=false path_following=online \
  training.play_checkpoint_path=/path/to/official_policy.pt \
  official_playback.profile_path=/path/to/source_profile.yaml \
  path_following.checkpoint_path=/path/to/response_b.pt \
  path_following.pose.odometry_source=simulation
```

路径配置位于 `conf/distill/path_following/online.yaml`；其默认定位源为 `cuvslam`，
示例显式改为 simulation 真值源。真实 cuVSLAM 需要受支持的 Linux/CUDA 环境，
缺失库、缺初始位姿或跟踪失败均报错，绝不回退真值；simulation 不打开相机或 tracker。
`learning_config_path` 复用第四步配置，`path_following.learning` 直接合并学习超参数覆盖，
不修改学习器严格配置接口。上述入口接通不代表示例中的模型或物理纠偏质量已验证。

`runtime.PathFollowingController` 是唯一最外层控制器：内部复用 official 播放，
可同时组合 `online_adaptation.enabled=true` 的执行层适应；不复制策略或另起物理循环。
工厂在建 env 前核对实际 policy checkpoint 哈希、源 profile、预处理、H/K 和 B 契约；
响应窗口长度直接取 B 产物，更新间隔由 `update_interval_steps` 单独指定。
只支持单环境、official policy、MuJoCo 固定正向直行，拒绝键盘/heading controller/
命令重采样/figure-eight，以及外部 `future_transform` 或隔离回放分支。

每步在原物理时钟上保存查询，按定位返回的历史控制编号配对同一时刻的位姿；
A 的候选与更新使用最新且未过期的真实配对，保留历史时间，不能把旧误差贴到当前查询。
校准前候选为零；校准后每步可重算，但只有窗口边界激活，执行中残差始终锁定。
每隔 T 步先消费新完整窗口更新 B，再用尚未消费的合法决策点更新 A；空间梯度不进入 Tracker。
执行层快照仍以合成 future 为目标，执行证据保存 target-order 的实际 env-input 动作。

`pose.sensor.capture_initial_frame=true` 为本模式提供打开/reset 时刻的双目帧与 FK，
只对已有物理状态做 forward/capture，不推进物理或伪造 IMU。其他传感器使用方默认关闭此项，
保留原采样节奏。cuVSLAM 保留已经产生的 autoreset 初始帧，且不读取评估真值作为定位输入。
参考线必须由任务起点证据建立；初始证据缺失、定位失效/过期、身份变化或时间倒退即停止，
不重新定线、不维持旧非零候选无限行走。terminal 窗口丢弃；显式 reset 清空待配对数据，
保留模型与优化器，之后按新的物理序列建立参考；close 先释放定位源，再由 session 释放环境。

MuJoCo 后端的 FULLPHYSICS 源状态固定保留原生 `float64`，包含供后续步进、
传感器时间戳及 `get_sim_time_ns()` 使用的物理时间；不得经观测 dtype 量化后再读时钟。
公开关节/基座观测通过预分配的 qpos/qvel 投影保持配置 dtype（默认 `float32`），
在普通步进、控制回调、传感器子步、部分 reset、模型分配与快照恢复后原位刷新。
速度扰动必须修改源状态再刷新观测，不能只改投影。`float64` 观测配置直接引用源切片，
不额外分配投影副本。采集器原有 ±1 ns 周期检查不放宽，错误物理周期仍拒绝。

`get_physics_state()` 与精确 rollout 快照使用原生 `float64`；恢复保留严格 dtype/shape
校验并刷新已持有的观测视图。旧 `float32` 精确快照不能无损恢复，不做补精度兼容。
视频用的 `NpEnv.get_physics_state_snapshot()` 保留原导出语义，不能当作精确时钟或恢复源。
这不会改变策略输入 dtype 或 checkpoint 格式，但不承诺与旧量化物理轨迹位级一致。

运行期的 `trace` 是 `max_trace_records` 限定的内存环形记录，保存测量时间、物理年龄、
横向/航向误差、候选/锁定残差、窗口/更新编号和丢弃原因，不自动写新报告文件。
查询缓存和待更新窗口也有独立容量上限，过期查询计数、容量超限报错；不静默覆盖训练样本。
目前只完成 CPU 计算与生命周期证据，未运行 CUDA 定位或机器人闭环，不宣称真实纠偏/抗振荡已通过。

### 有限闭环评估与证据落盘

`scripts/evaluate_path_following.py` 使用独立的 `conf/sttt/path_following_evaluate.yaml`，
调用 `path_following.evaluation.run_path_following_evaluation`。它是真实实验入口，
执行该命令会启动显式请求的仿真/在线适应，不是自动验收或训练准备步骤。

必须显式提供输出目录、policy checkpoint、源 profile、物理时长、五项指标阈值，
以及非空的 `modes`、`conditions`、`seeds`。支持模式为 `nominal`（均不适应）、
`oat`（仅执行层适应）、`oat_path`（执行层适应与路径纠偏）；条件为
`clean` 或 `delayed_payload`。只运行这些显式列表的组合，不自动增加种子、延长或重试。
`oat_path` 还必须提供独立 B 产物；`pose_source` 显式选择 simulation/cuvslam，绝无真值回退。

`source_overrides` 直接 compose 到既有 `g1_walk_flat/mujoco` 与 `straight_payload` 预设，
用于共享场景/命令/负载配置；模式与负载开关由评估器按请求设置。每组重新创建 session
并设置同一种子，不继承其他组的模型、优化器或轨迹。无视频渲染路径；只通过公共 session
单步执行并关闭 autoreset，遇 terminal/truncation/定位失效/异常立即停止该 episode。
异常停止后续组合；正常提前终止可继续其余已请求组合，但不能被标成完整运行。

五项必填阈值为 `lateral_tolerance_m`、`heading_tolerance_rad`、`dwell_s`、
`crossing_deadband_m`、`minimum_forward_progress_m`；不内置论文通过阈值。
时长必须为实际控制周期的整数倍。评分真值通过公开基座位姿读取，仅进入独立评估通道，
不传回定位、A/B 或 Tracker。参考线在 reset 起点固定，三种模式使用相同误差定义。

每个运行输出 `source.json`（配置、模型/源 profile 哈希、种子、实际条件及参考线）、
`trajectory.csv`（物理时间、真值误差、前进位置、实际负载事件）、`runtime.jsonl`
（完整逐步运行期记录）及 `summary.json`；顶层另有组合摘要。非空输出目录拒绝覆盖。
运行期 `drain_trace()` 每步转移记录，不靠最后一次读取有限环形缓冲；JSONL 保留所有到达记录，
但定位统计按被表示的测量时间去重，重复使用旧观测不增加独立测量数。

RMSE 对物理时间作梯形积分，不按到达次数或视频帧数平均；缺测不外推、不跨空洞积分，
并中断持续恢复判据。摘要区分保持、恢复、未恢复、恢复后再丢失、覆盖不足或持续时间不足。
恢复进入时刻须经完整 dwell 确认，同时满足横向与航向容差；从未偏离不算恢复，
站住时即使误差很小也不能通过前进要求。带死区横向过零次数与恢复后峰值单独报告。

负载附加依据 `state.info` 的实际事件记录，时间为对应已执行区间的起点，
不把配置中的计划时间当成已发生事件；另报实际附加后的指标。未实际附加的载荷实验
即使到达时长，也标为未完成。`complete` 只表示实验协议完整，`tracking_success`
才表示该次运行满足配置的跟踪与前进要求；二者都不代表跨场景泛化或雅可比方向已验证。
异常与关闭失败写入失败摘要，后物理步异常仍尽量保留最后真值证据，绝不伪造成功。
CLI 对未完整执行的协议返回非零状态；本轮仅验收了合成 I/O 和解析轨迹，未执行真实实验。

## MuJoCo 双目 / IMU 与 cuVSLAM 诊断

`scripts/diagnose_visual_odometry.py` 采集机载传感器并评估 Rig 轨迹，同时将估计转换为
局部 Z-up 骨盆轨迹、关联历史动作区间；不启用 LoRA。默认只采集：无需 CUDA，
但必须提供真实 G1 FADA Planner-IDM checkpoint；没有零策略或真值定位回退。

```bash
uv run scripts/diagnose_visual_odometry.py mode=capture \
  source.checkpoint_path=/absolute/path/to/planner_idm.pt \
  output_dir=data/sttt/visual_capture source.num_steps=300

# 在已安装官方 PyCuVSLAM wheel 的 Linux + NVIDIA/CUDA 环境回放：
uv run scripts/diagnose_visual_odometry.py mode=replay \
  input_dir=data/sttt/visual_capture odometry.mode=stereo
```

配置位于 `conf/sttt/visual_odometry.yaml`；支持 `--config path.yaml` 和点号覆盖。
`mode=live` 在同一进程内运行采集与定位。`odometry.mode=inertial` 还要求显式填写
`odometry.imu_noise` 的四个密度/随机游走值；这些值是仿真模型假设，不是实机标定。
cuVSLAM 是独立可选依赖；适配器以 [v17.0.1](https://github.com/nvidia-isaac/cuVSLAM/releases/tag/v17.0.1)
的 `Tracker.OdometryConfig` / `Tracker.OdometryMode` 接口为依据，显式关闭 SLAM 建图，保留同步里程计。
安装匹配平台和 CUDA 的 wheel；不假定所有更新版本均兼容。当前 Apple Silicon Mac 不提供 CUDA 定位验证。

采集目录必须为空。每帧 `frame_*.npz` 保存同状态 RGB 双目、物理采集时间、帧间 IMU，
以及同戳 `pelvis_from_rig` 和骨盆局部理想重力上方向；这些相对量不提供世界位置或 heading。
`calibration.npz` 保存左光学 Rig 标定。`controls.csv` 记录同一物理时钟上的动作起止时间、
实际 `env_input` 和执行前速度命令。旧采集缺少运动学或动作记录时明确拒绝，需重新采集，
不能假设腰部锁定。`reference.csv` 是独立的评估真值，绝不传给 Tracker 或骨盆转换器。

回放写出 `odometry.csv`（Rig 误差）、`pelvis_odometry.csv`（估计骨盆位姿）和
`aligned_intervals.csv`（历史动作及骨盆起止位姿，并保留插值两端的原始测量时间）。
骨盆转换采用 `T_OR * inverse(T_PR)`，每定位 epoch 仅用首次有效估计、局部上方向和
骨盆前向建立一次 Z-up 原点，后续不重锚。时间线只在同 episode/epoch 的有效测量间
插值，不外推；超过 `alignment.max_gap_s` 的间隙、失效或过期区间丢弃并计数。
`alignment.max_age_s` 是物理时间等待上限，不是脚本或子 Agent 的运行时限。
Rig 误差评估另用每段首个有效位姿做一次 SE(3) 对齐，不拟合尺度或整条轨迹。
转换器与时间线可独立诊断；测量运行时使用 `PoseRuntimeConfig`，
与空间学习配置解耦。完整 MuJoCo + cuVSLAM 行走运行验收仍需另行执行。

当前仅支持单环境、刚性胸部 IMU 和基础场景模型，不支持随机化模型 variant。
默认 G1 步长对应 150 Hz IMU、30 Hz 双目。启用时按物理子步采样；关闭时保持原批量
步进路径。双目配置仅定义仿真安装位置；视觉地标不参与碰撞，正常任务配置不变。
原生传感器、导出与 reset/close 验证不等于视觉定位质量或 STTT 纠偏效果已经验证。


## Streaming TTT 在线实验

本仓库的在线 LoRA 实验入口为 `scripts/play_fada_online.py`，使用
`task=g1_walk_flat/mujoco recording_profile=straight_payload`，并通过
`training.play_checkpoint_path` 指定真实 Planner-IDM checkpoint。它与下述冻结权重的
ICE-Cal 路径不同：Planner 和 IDM 基础权重冻结，只有 IDM LoRA 接受在线更新。

- `online_adaptation.enabled=false` 保持无在线更新的 baseline；`true` 启用冻结基座轨迹差分标签。
- 在 t 保存实际推理的历史输入、完整 Planner 标称未来 `Y_nom` 和在线首动作快照；
  到 t+K 收齐同一时间段的真实状态轨迹 `Y_exec`。K 来自 checkpoint 的预测长度
  (当前模型 K=6)，不是训练批量 T；终止/reset 或命令切换不能拼成跨边界轨迹。
- 在更新批次形成时，冻结基座对同一历史分别输入 `Y_nom` 与 `Y_exec`，取首动作作
  `nom-exec` 差分。固定目标为 `U_snapshot + gamma * delta`，不使用单步关节误差、
  不除 `action_scale`，不随在线参数重算；环境执行端的动作缩放保持不变。
- 基座在在线 LoRA 注入前独立复制，只复制 IDM，eval/关闭梯度，不进入优化器。
- 默认 `window_size=10`、`warmup_steps=40`、`update_interval=10`，
  `gamma=0.05`、`sigma=1e-4`。控制周期 0.02s、K=6 时，第 40 步 (0.8s) 首次更新，
  使用最后 10 条成熟轨迹；此后第 50、60……步 (每 0.2s) 消费 10 条新成熟轨迹。
  成功只清成熟训练批，不清在途轨迹或推理历史；失败保留批次并抛出异常。
  非默认配置须同时满足 warmup、更新周期和满批条件，不等待负重挂载。
  这些数值是试验初值，不代表已证实的稳定性或适应效果。
- 损失为窗口内首动作误差平方范数的 `1/(2T)` 均值，加相对部署初始 LoRA 参数的
  显式 L2 锚定；不对动作维再平均。更新时关闭 dropout 但开启梯度，使用 Adam，
  不通过 AdamW 的默认权重衰减替代非零参数锚点。
- reset 清空推理历史、在途轨迹和成熟窗口，并重新计算 warmup，但保留已学 LoRA、
  冻结基座、锚点与优化器状态。
  关闭 autoreset 时，终止后仍可继续物理步进，但暂停采集/更新，直到显式 reset。
- 旧在线 `lambda_forget` 配置和 `scripts/test_online_adaptation.py` 手动入口已移除。
  使用 `uv run --frozen --no-sync python scripts/play_fada_online.py`，启动日志应包含
  `objective=expert_trajectory_difference`。视频录制、负重事件和离线 FADA 路径保持原接口。

### Linux/CUDA 服务器后台录制

以下沿用此前服务器的仓库和 checkpoint 示例路径，本轮未连接服务器核验。
如实际路径不同，只调整 `cd` 和 `training.play_checkpoint_path`。
命令使用已经配置好的 uv 环境，不创建环境、不安装依赖。
`LD_PRELOAD` 沿用此前 Linux x86_64 服务器的进程级 libgcc 设置，不是跨平台通用配置。

两组均从第 0 秒向左肩挂载 3kg，固定前进命令为 0.4m/s，侧后方跟随视角，
录制 120 秒仿真时间。命令显式覆盖 profile 默认的 30 秒挂载延迟，不修改 profile。
分别运行两组；除在线开关与产物命名外，checkpoint、物理和录制条件保持一致。
视频和日志名带时间戳；重复启动前确认名称不同，避免覆盖已有结果。

#### 开启基座轨迹差分在线更新

```bash
cd /ssd1/cyx/TTT
mkdir -p video logs
TTT_RUN_ID="$(date +%Y%m%d-%H%M%S)"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD/src" \
  uv run --frozen --no-sync python scripts/play_fada_online.py \
  task=g1_walk_flat/mujoco \
  recording_profile=straight_payload \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/FADA_Unofficial/model/v024_checkpoint/planner_idm_close_some_dr_v001.pt \
  env.runtime_payload.enabled=true \
  env.runtime_payload.added_mass_kg=3.0 \
  env.runtime_payload.attach_time_s=0.0 \
  online_adaptation.enabled=true \
  online_adaptation.window_size=10 \
  online_adaptation.warmup_steps=40 \
  online_adaptation.update_interval=10 \
  online_adaptation.gamma=0.05 \
  online_adaptation.sigma=1e-4 \
  online_adaptation.learning_rate=1e-4 \
  video_recording.enabled=true \
  video_recording.duration_s=120.0 \
  video_recording.progress_interval_s=1.0 \
  video_recording.output_path="$PWD/video/left_shoulder_expert_${TTT_RUN_ID}.mp4" \
  > "$PWD/logs/left_shoulder_expert_${TTT_RUN_ID}.log" 2>&1 &
```

在启动所在 shell 查看本次日志：

```bash
tail -f "/ssd1/cyx/TTT/logs/left_shoulder_expert_${TTT_RUN_ID}.log"
```

#### 相同负重、关闭在线更新的 baseline

这里的 baseline 仍然挂载 3kg，不是无负重 baseline。

```bash
cd /ssd1/cyx/TTT
mkdir -p video logs
BASELINE_RUN_ID="$(date +%Y%m%d-%H%M%S)"

nohup env \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$PWD/src" \
  uv run --frozen --no-sync python scripts/play_fada_online.py \
  task=g1_walk_flat/mujoco \
  recording_profile=straight_payload \
  training.device=cuda \
  training.play_checkpoint_path=/ssd1/cyx/FADA_Unofficial/model/v024_checkpoint/planner_idm_close_some_dr_v001.pt \
  env.runtime_payload.enabled=true \
  env.runtime_payload.added_mass_kg=3.0 \
  env.runtime_payload.attach_time_s=0.0 \
  online_adaptation.enabled=false \
  online_adaptation.window_size=10 \
  online_adaptation.warmup_steps=40 \
  online_adaptation.update_interval=10 \
  online_adaptation.gamma=0.05 \
  online_adaptation.sigma=1e-4 \
  online_adaptation.learning_rate=1e-4 \
  video_recording.enabled=true \
  video_recording.duration_s=120.0 \
  video_recording.progress_interval_s=1.0 \
  video_recording.output_path="$PWD/video/left_shoulder_baseline_${BASELINE_RUN_ID}.mp4" \
  > "$PWD/logs/left_shoulder_baseline_${BASELINE_RUN_ID}.log" 2>&1 &
```

在该启动 shell 查看本次日志：

```bash
tail -f "/ssd1/cyx/TTT/logs/left_shoulder_baseline_${BASELINE_RUN_ID}.log"
```

`Ctrl-C` 退出 `tail` 不会停止后台录制。产物分别位于仓库的 `video/` 和 `logs/`，
以命令中的时间戳对应同一次运行。

#### 启动、进度与完成标准

- 新方法启动日志应包含 `objective=expert_trajectory_difference`，以及实际
  `prediction_horizon`、`window_size`、`warmup_steps`、`update_interval`。
  K 以加载 checkpoint 后的日志为准；在 K=6、0.02s 控制周期、没有提前终止的
  episode 中，默认在第 40/50/60……步更新。
- `Progress: step=当前步/总步` 同时显示仿真时间与帧数；`Update` 表示一次优化完成。
  进度按墙钟间隔在控制步完成后报告，不保证长耗时步骤内部每秒打印。
- baseline 应显示 `Online adaptation disabled`，视频更新计数保持 0，
  但同样应显示 `PAYLOAD +3kg`。
- 最后出现 `Video saved:` 才表示录制流程完成；文件已经存在或
  `Finalizing MP4 encoding...` 都不代表编码已经结束。
- 视频中的 `resets` 增长表示轨迹被终止/重置，不要把重置后的站立当成在线恢复。
  更新次数增加或 loss 较低也不能单独证明负重适应成功。

当前证据是第三步的真实小型网络与正式 factory/session 离线 smoke。
本节命令尚未在服务器实际启动，真实 checkpoint 下的录制与负重效果仍未验证。


## Current status

ICE-Cal has three separately governed research tracks; there is no single project-wide runtime
readiness label:

| Track | Current authority | Status |
|---|---|---|
| Source Oracle | `FADA-METHOD-v024` / `FADA-TRAIN-v024` | implemented and offline module-tested; bounded reset/step runtime smoke passed; full formal audit pending; not trained |
| Target adaptation | `FADA-ADAPT-METHOD-v003` / `FADA-ADAPT-TRAIN-v003` | active Contract; runtime and policy claims remain receipt-specific |
| In-context calibration | `FADA-CONTEXT-METHOD-v009` / `FADA-CONTEXT-TRAIN-v008` | data-driven basis design active; engineering proposal only |

The v024 Oracle replaces the unrun v023 phase-only route. It normalizes small commands to exact
zero, uses a deterministic null/non-null phase state machine, and schedules bounded contact
mismatch instead of a phase-height target. Translation and yaw jointly control a piecewise
`0`/`[f_min,f_max]` clock; Reward coefficients remain engineering parameters constrained by the
confirmed six-relation ordering card. The production selector is
`task=sac/g1_walk_flat/mujoco_fada_phase_contact`; offline tests do not establish simulator
reachability or policy quality. Targeted actuator faults are excluded from Source Oracle training;
left-knee gain attenuation is reserved for downstream failed-rollout collection after the Oracle is
frozen. Generic all-joint Kp/Kd and the remaining physical domain randomization stay enabled.

No design or documentation status authorizes training, simulation, deployment, or policy-quality
claims. See the [project governance index](note/governance/README.md) for the current cursor and
track boundaries.

## 仓库地图

```text
ICE-Cal/
│
├── conf/                                      Hydra 配置：决定“运行什么”
│   ├── offpolicy/                             SAC / TD3 / FlashSAC 配置根
│   │   ├── config.yaml                        离策略训练的顶层默认配置
│   │   ├── algo/                              算法参数
│   │   │   ├── sac.yaml                       SAC 网络、batch、更新频率、学习率
│   │   │   ├── td3.yaml                       TD3 参数
│   │   │   └── flashsac.yaml                  FlashSAC 参数
│   │   ├── task/                              任务与后端的配置所有者
│   │   │   └── sac/g1_walk_flat/
│   │   │       ├── mujoco.yaml                通用 MuJoCo G1WalkFlat
│   │   │       ├── motrix.yaml                通用 Motrix G1WalkFlat
│   │   │       ├── mujoco_clean_baseline.yaml 无 Gait、无特权输入、无 DR 基线
│   │   │       ├── mujoco_fada_source.yaml    v022 phase-neutral Source Oracle
│   │   │       ├── mujoco_fada_phase.yaml     v023 phase-locomotion 兼容配置
│   │   │       ├── mujoco_fada_phase_contact.yaml
│   │   │       │                               v024 command-gated phase/contact Source Oracle
│   │   │       ├── mujoco_fada_target*.yaml   FADA 目标域采集/评估任务
│   │   │       └── mujoco_context_teacher*.yaml
│   │   │                                       Context full-action/residual teacher
│   │   ├── fault/                             显式执行器故障
│   │   │   ├── left_knee_090.yaml             左膝 0.9 增益条件
│   │   │   └── right_knee_090.yaml            右膝 0.9 增益条件
│   │   ├── target_domain/                     目标域环境条件
│   │   │   ├── slope_10.yaml                  10° 斜坡
│   │   │   └── slope_15.yaml                  15° 斜坡
│   │   ├── collection/                        FADA 目标轨迹采集配置
│   │   ├── adaptation/                        IDM LoRA 适配配置
│   │   └── evaluation/                        FADA 评估配置
│   │
│   ├── distill/                               Planner–IDM 蒸馏配置
│   │   ├── task/g1_walk_flat/                 Student 环境和 Source 行为身份
│   │   ├── workflow/                          蒸馏阶段与采集流程
│   │   ├── calibration_playback/              校准策略回放覆盖
│   │   └── context_playback/                  Context 策略回放覆盖
│   ├── fada_context/                          In-Context 校准配置
│   │   ├── calibration_axes/                  Gain / Delay / Offset 等校准轴
│   │   └── calibration_collection/            校准数据采集事务
│   ├── appo/                                  异步 PPO 任务配置
│   ├── ppo/                                   RSL-RL PPO 任务配置
│   ├── ppo_him/                               HIM-PPO 任务配置
│   └── hora_distill/                          HORA 蒸馏配置
│
├── scripts/                                   面向用户的命令入口；只组装流程
│   ├── train_offpolicy.py                     SAC / TD3 / FlashSAC 训练入口
│   ├── train_distill.py                       FADA Planner–IDM 蒸馏入口
│   ├── play_interactive.py                    原生 MuJoCo 交互回放
│   ├── collect_fada_target.py                 FADA 目标域轨迹采集
│   ├── adapt_fada_target.py                   FADA 目标域 LoRA 适配
│   ├── evaluate_fada_slope.py                 斜坡适配前后评估
│   ├── train_fada_calibration*.py             Context 校准 Stage 1/2/3
│   ├── train_appo.py                          APPO 训练入口
│   ├── train_rsl_rl.py                        PPO 训练入口
│   ├── train_mlx_ppo.py                       MLX PPO 训练入口
│   ├── deploy/                                部署入口和适配器
│   ├── motion/                                动作数据处理工具
│   └── manip_loco/                            操作与移动复合任务工具
│
├── src/unilab/                                生产代码
│   │
│   ├── base/                                  框架核心契约
│   │   ├── np_env.py                          NpEnvState、reset/step 环境契约
│   │   ├── registry.py                        Env/EnvCfg 注册与实例化
│   │   ├── observations.py                    观测组公共结构
│   │   ├── curriculum.py                      通用课程机制
│   │   └── backend/
│   │       ├── base.py                        SimBackend 公共抽象接口
│   │       ├── mujoco/                        MuJoCo 实现、XML 与回放
│   │       └── motrix/                        Motrix 实现、场景与回放
│   │
│   ├── envs/                                  任务环境
│   │   ├── common/                            环境共享数学/旋转工具
│   │   ├── locomotion/
│   │   │   ├── common/                        各机器人共用移动逻辑
│   │   │   ├── go1/ go2/ go2w/ go2_arm/      其他腿式机器人任务
│   │   │   └── g1/                            G1 移动任务核心
│   │   │       ├── joystick.py                G1WalkEnv 组合入口与注册
│   │   │       ├── base.py                    G1 资产和基础状态
│   │   │       ├── walk_config.py             Env/Control/Reward/DR 配置结构
│   │   │       ├── walk_commands.py           速度指令采样和重采样
│   │   │       ├── walk_observation_bindings.py
│   │   │       │                               Actor/Critic/FADA 观测组装
│   │   │       ├── walk_control_bindings.py   Action 门控与实际执行控制
│   │   │       ├── walk_reward_bindings.py    Reward 汇总和终止相关代价
│   │   │       ├── walk_runtime_bindings.py   每步状态、课程日志、快照
│   │   │       ├── walk_domain_randomization.py
│   │   │       │                               G1 专属 DR 计划与课程
│   │   │       ├── walk_actuator_randomization.py
│   │   │       │                               执行器强度采样
│   │   │       ├── walk_reset_randomization.py
│   │   │       │                               Reset 状态和物理参数随机化
│   │   │       ├── fada_privileged.py         FADA 特权向量和布局身份
│   │   │       ├── calibration_fault.py       校准用执行故障注入
│   │   │       ├── action_trace.py             Action 权限与调试轨迹
│   │   │       └── walk_{control,reward,observation,math}.py
│   │   │                                       可复用的纯计算函数
│   │   ├── manipulation/                      Allegro/Sharpa/Stewart 操作任务
│   │   └── motion_tracking/g1/                G1 参考动作跟踪
│   │
│   ├── training/                              应用层训练流程
│   │   ├── offpolicy/
│   │   │   ├── application.py                 开始、训练、清理、结束生命周期
│   │   │   ├── factory.py                     Env/Learner/Runner 构建
│   │   │   └── playback.py                    Checkpoint 加载与回放
│   │   ├── backend_adapter.py                 Hydra 配置到 Backend/Env 的适配
│   │   ├── sim2sim.py                         跨后端配置兼容性检查
│   │   ├── experiment.py                      实验身份、摘要和日志生命周期
│   │   └── run.py                             通用运行辅助逻辑
│   │
│   ├── algos/                                 Learner 与研究方法
│   │   ├── torch/
│   │   │   ├── fast_sac/
│   │   │   │   └── learner.py                 SAC Actor/Critic、Loss、优化器
│   │   │   ├── fast_td3/                      TD3 Learner
│   │   │   ├── flash_sac/                     FlashSAC Runtime
│   │   │   ├── appo/                          APPO 实现
│   │   │   ├── offpolicy/                     离策略采集与学习调度
│   │   │   │   ├── worker.py                  Collector 子进程环境交互
│   │   │   │   ├── collector_session.py       Collector 会话生命周期
│   │   │   │   ├── double_buffer_runner.py    双缓冲训练调度与保存
│   │   │   │   ├── runner.py                  基础离策略 Runner
│   │   │   │   └── checkpoint_adapter.py      通用 Checkpoint 适配边界
│   │   │   ├── distill/
│   │   │   │   └── fada/                      FADA 主实现
│   │   │   │       ├── source/                公共阶段 1：Source Oracle
│   │   │   │       │   ├── runtime.py          特权 SAC Runtime 解析
│   │   │   │       │   └── diagnostics.py      Source 诊断入口
│   │   │   │       ├── planner_idm/           公共阶段 2：Planner–IDM
│   │   │   │       │   ├── __init__.py         模型与 Checkpoint 公共 API
│   │   │   │       │   └── training.py         Source 蒸馏流程入口
│   │   │   │       ├── target/                公共阶段 3：目标域
│   │   │   │       │   ├── collection.py       目标 rollout 采集入口
│   │   │   │       │   ├── adaptation.py       IDM LoRA 适配入口
│   │   │   │       │   └── evaluation.py       目标域评估入口
│   │   │   │       ├── model.py               Planner、IDM、组合策略
│   │   │   │       ├── checkpoint.py          Source Checkpoint 契约
│   │   │   │       ├── collector.py/oracle.py Source 轨迹与 Oracle 标签
│   │   │   │       ├── windows.py/replay.py   H/K 窗口和训练样本
│   │   │   │       ├── collection_*.py        采集身份、状态、事务、IO
│   │   │   │       ├── target_*.py            目标采集/适配/评估内部实现
│   │   │   │       ├── adaptation*.py         LoRA 适配与持久化
│   │   │   │       └── legacy_workflow.py     旧 Schema 隔离兼容层
│   │   │   └── fada_context/                  In-Context 校准
│   │   │       ├── calibration_{data,collection}.py
│   │   │       │                               校准样本与采集
│   │   │       ├── calibration_{models,policy}.py
│   │   │       │                               校准模型和冻结策略组合
│   │   │       ├── calibration_training/
│   │   │       │   ├── stage1.py               修正算子
│   │   │       │   ├── stage2.py               系数 Encoder
│   │   │       │   ├── stage3.py               尺度映射
│   │   │       │   ├── pipeline.py             串联三个冻结阶段
│   │   │       │   └── io.py/types.py          阶段产物 IO 与类型
│   │   │       ├── calibration_{runtime,readout}.py
│   │   │       │                               推理读数与修正组合
│   │   │       ├── calibration_evaluation.py  校准前后评估
│   │   │       └── formal_protocol.py         官方路径审计入口
│   │   └── mlx/                               MLX 算法实现
│   │
│   ├── ipc/                                   进程间通信与 Replay
│   │   ├── replay_buffer.py                   CPU 共享 Replay 存储
│   │   ├── async_runner.py                    通用异步 Runner 基础设施
│   │   ├── weight_sync.py                     Collector/Learner 权重同步
│   │   ├── shared_buffer.py                   共享内存基础结构
│   │   └── replay_pipelines/
│   │       ├── cpu_pinned_double_buffer.py    Batch 准备与 H2D 重叠
│   │       └── transfer/                      CUDA/XPU/Torch 传输实现
│   │
│   ├── dr/                                    与后端无关的域随机化契约
│   ├── assets/                                Robot XML、网格、场景、动作
│   │   ├── robots/g1/                         G1 模型和 task-level 场景
│   │   ├── robots/{go1,go2,go2w,...}/         其他机器人资产
│   │   └── motions/g1/                        G1 参考动作
│   ├── terrains/                              地形生成与表示
│   ├── logging/                               训练指标与 Trace
│   ├── visualization/                         渲染和可视化
│   ├── tools/                                 仓库/运行时工具
│   └── utils/                                 通用工具
│
├── tests/                                     与生产边界对应的测试
│   ├── architecture/                          所有权和仓库形状检查
│   ├── config/                                Hydra 组合契约
│   ├── envs/locomotion/g1/                    G1 Reward/Obs/Control/DR
│   ├── algos/                                 SAC、FADA、Context
│   ├── ipc/                                   Replay 与异步生命周期
│   ├── training/                              Training 应用层
│   ├── base/backend/                          Env/Backend 公共契约
│   ├── integration/                           跨模块集成
│   └── scripts/                               CLI 接线
│
├── note/                                      研究权威与证据
│   ├── governance/                            当前项目/轨道治理游标
│   ├── architecture/                          Concept Figure、Design Inspector
│   └── fada/
│       ├── contracts/
│       │   ├── active/                        当前 Method/Training Contract
│       │   └── history/                       已被替代、不可授权的旧 Contract
│       ├── plans/                             带日期的设计和工程方案
│       ├── governance/                        阶段转换与执行 Receipt
│       ├── testing/                           Module Test Card 和测试记录
│       ├── evidence/                          失败、运行与评估证据
│       └── reviews/                           方案/代码可维护性审查
│
├── docs/                                      稳定工程文档
│   ├── engineering/                           实现说明
│   ├── runbooks/                              可重复执行的操作手册
│   └── superpowers/                           历史开发过程记录，非当前权威
│
├── benchmark/                                 性能基准和测试数据
├── notebook/                                  探索性 Notebook，非生产 Owner
├── model/                                     本地 Checkpoint，使用前核对身份
├── logs/                                      训练与运行日志
├── artifacts/                                 FADA 数据集与评估产物
├── work/                                      生成的工作材料
├── pyproject.toml                             Python 依赖与工具配置
├── Makefile                                   仓库验证入口
├── CONTRIBUTING.md                            开发环境与贡献流程
└── AGENTS.md                                  Agent 必须遵守的仓库契约
```

## Oracle Policy 训练链路

以下链路对应当前仓库中可直接执行的 v022 Source Oracle：配置入口为
`mujoco_fada_source.yaml`，实现是 privileged SAC，Checkpoint 使用 sealed lineage 契约。

```text
Hydra 配置
│
├── conf/offpolicy/config.yaml
│     Off-policy 顶层训练配置
│
├── conf/offpolicy/algo/sac.yaml
│     SAC 网络、Replay、batch、学习率和更新频率
│
└── conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_source.yaml
      Oracle 行为、特权输入、Reward、Grouped DR 和 Checkpoint 谱系
        ↓
scripts/train_offpolicy.py
      Hydra 命令入口，只负责组合配置并转交训练应用
        ↓
src/unilab/training/offpolicy/application.py
::run_offpolicy()
  ├── ensure_registries()                         注册环境与算法
  ├── 设置随机种子
  ├── 创建 ExperimentTracker
  ├── build_runner()
  └── runner.learn()
        ↓
src/unilab/training/offpolicy/factory.py
::build_runner()
  ├── BackendAdapter                              生成 EnvCfg override
  ├── 创建一次 G1WalkEnv                          读取 obs/action 维度
  ├── 解析 algo.runtime_resolver
  └── 构造 Learner + DoubleBufferOffPolicyRunner
        ↓
algo.runtime_resolver
src/unilab/algos/torch/distill/fada/source/runtime.py
::resolve_privileged_locomotion_sac_runtime()
        ↓
FADAPrivilegedSACRuntime
  ├── validate_training_config()
  ├── 读取 G1 privileged observation 布局
  ├── 构造 FADAOracleCheckpointContract
  ├── 提供 FADAPrivilegedSACLearner
  └── 提供 sealed Checkpoint saver
        ↓
G1WalkEnv
  ├── 普通任务观测
  │     → Actor 基础输入
  │
  ├── g1_fada_privileged_v1
  │     → 追加到 Critic observation 尾部
  │
  ├── Command
  │     → 速度跟踪目标
  │
  ├── Reward
  │     → phase-neutral，feet_phase = 0
  │
  └── Physical DR
        ├── 定向执行器故障（包括左膝增益衰减）：仅用于 Oracle 冻结后的故障轨迹采集
        ├── 全关节 Kp / Kd：保留
        ├── 摩擦
        ├── base / body mass
        ├── COM
        └── DoF position bias
        ↓
FADAPrivilegedSACLearner
  └── 继承 HoraSACLearner
        └── 继承 FastSACLearner
              ↓
Actor
  ├── 普通任务观测
  ├── 从 Critic observation 尾部提取 privileged vector
  ├── 对 privileged vector 做归一化
  ├── Privileged MLP：256 → 128 → 32
  └── 普通观测 + 32-D privileged embedding → Action
              ↓
Critic
  └── 完整 Critic observation + Action → Distributional Q
              ↓
DoubleBufferOffPolicyRunner
  ├── Collector worker 与 G1WalkEnv 交互
  ├── transition 写入 ReplayBuffer
  ├── CPU-pinned DoubleBuffer 准备 batch
  ├── update_critic()
  ├── update_actor()
  ├── 更新温度参数
  └── 软更新 Target Q
              ↓
FADAOracleCheckpointGateway
  └── 保存 model_<iteration>.pt
        ├── Actor / Critic 参数
        ├── privileged normalization
        ├── oracle_lineage_id
        ├── behavior_profile
        ├── Env / Reward / Algo 配置哈希
        ├── observation / action 布局
        └── robot asset 身份
```

## 官方 FADA Locomotion 源训练

正式入口为 `scripts/train_distill.py --config-name official_fada`，配置位于 `conf/distill/official_fada.yaml`，不再继承旧 SAC teacher、MuJoCo task 或 66/3 站走课程。非 FADA 蒸馏、旧 checkpoint 播放和现有在线 LoRA 方法保持独立。

在仓库根目录启动；将下列路径替换为真实官方 checkout 和 PPO 产物：

```bash
PYTHONPATH="$PWD/src" uv run --frozen --no-sync python scripts/train_distill.py \
  --config-name official_fada \
  training.fada.upstream_root=/absolute/path/to/FADA \
  training.fada.config.expert_checkpoint=/absolute/path/to/oracle_run/model_5000.pt
```

官方源码固定为 `LeCAR-Lab/FADA@290b3a9c5c0ebe2e9851b61350617544be05e9bc`。需要官方可读取的 PPO 配套保存配置、弱 checkpoint 历史，以及 Linux/CUDA/官方原生仿真环境；现有 MuJoCo SAC/APPO 和 66/3 学生不能替代。命令不会下载权重或安装依赖。

### 采集、Replay 与外循环

- `fada.async_runtime.build_official_fada_runtime` 显式接收 `OfficialFADACollectionSpec`、`student_loader` 和 `initial_student`。先通过同一 Runner 的 `prepare` 获取实际 Oracle 身份、预处理、配置和弱策略列表，再保存完整初始 checkpoint、`activate_checkpoint` 并采集；没有缺失文件时随机初始化的 loader 回退。
- `OfficialFADACollectRequest` 明确指定 `optimal` / `suboptimal` / `online`；弱策略请求另带 `weak_checkpoint`，可选 `collect_steps` 以跨全部 env 的原始 transition 总数计。一个常驻 worker 复用官方 env、Trainer 和固定 Oracle shadow；不调用官方完整训练主循环。
- 采集产物使用同一 `FADAShardedSourceWriter` 的官方 schema v8，经 `open_official_fada_source_artifact(...).iter_batches()` 读取原始 13 字段：obs、command、executed/expert action、两类 expert chunk 及有效标志、strict-label 标志、reward、done、env/episode ID。旧官方窗口 v7 不能冒充原始 Replay；历史 v5 reader 不变。
- 父进程的 `OfficialFADAReplayBuffer` 复用固定上游纯 NumPy ReplayBuffer，按轨迹抽取、train/val 划分和 random/fifo 淘汰；容量单位为 transition。离线集可增长，online 跨轮积累；更新时重新 `sample_chunk`，而不是反复训练 worker 预抽的静态窗口。跨 shard 的完整 episode 统一判断。
- 唯一外循环执行 `prepare → offline 采集/缓存 → warmup → DAgger 采集/更新 → save/publish → 下一轮 → close`。optimal 按固定 horizon 分次采集并保留完整 episode；suboptimal 轮换弱 checkpoint 至目标数据量。官方缓存、warmup 开关优先级和恢复语义保留。
- 第一轮 online 前发布 warmup 后权重，每轮更新后发布新权重；`save_every_iter` 仅控制保留的周期快照，不降低权重发布频率。恢复包含模型、两组 optimizer、更新/轮次/采集计数及对应 online train/val Replay；不承诺物理环境或 RNG 的精确断点重放。
- 此路径使用 non-daemon collector 以支持官方 shadow 子进程，继续沿用原错误通道、版本屏障和 POSIX 进程组回收。仍是轮次 barrier，不宣称已实现采学重叠。

### 学生与学习接口

- `fada.official_model.build_official_fada_student` 构建纯 Torch 官方结构，默认 compact/action/command 为 64/29/7，H=30、K=6；保留官方位置编码、pre-Norm、delta 重建与原生权重布局，不借用旧 66/3 模型。
- `OfficialFADALearner.update(idm_batches_by_source, planner_batch)` 每批先更新 IDM，再固定其参数、经 IDM 反传更新 Planner。保留官方增强、mask、AdamW、梯度裁剪与来源配额；输出完整 K 步，但默认 `idm_action_loss_horizon_gamma=0` 仅监督首动作，Planner 状态损失系数默认 0。`sample_update(mode="offline" 或 "mixed")` 先采 Planner，再采五种 IDM 用途。
- 预处理来自 worker 实际解析的 Oracle run；artifact 必须与其身份一致，不猜测缺失的 scale/noise，也不额外启用 IO normalization。
- `save_official_fada_checkpoint` 保存独立 `unilab_official_fada_student` schema v2、模型/训练/预处理配置、原生权重、两组 optimizer、计数和 Replay 引用；`load_official_fada_checkpoint` 严格恢复学习状态，`load_official_fada_student_checkpoint` 仅恢复 eval 模型。旧学生、旧官方诊断 schema 和 Oracle PPO/APPO 不混用。

已完成两类离线证据：固定官方源码的 CPU 数值对照（网络、采样、loss、权重与 AdamW 状态），以及真实入口、原始 Replay、多 shard、warmup、两轮 DAgger、共享权重发布和恢复的短链路验证。后者使用明确标识的诊断轨迹，不代表真实采集。官方原生仿真验收、shadow 状态/RNG 隔离和策略质量仍需实际产物、环境及运行授权。

### 官方学生播放与在线适应（第 5 步）

- 沿用 `scripts/play_fada_online.py` 和同一 playback session。使用 `task=g1_walk_flat/mujoco`、`interactive.action_mode=policy`，将 `training.play_checkpoint_path` 指向官方学生 checkpoint，并显式提供 `official_playback.profile_path`。关闭/开启更新仍由 `online_adaptation.enabled=false/true` 控制，不另建播放或更新循环。
- profile 是从真实 Oracle run 核准的 YAML 或 JSON，必须包含 `source_identity`、`joint_names`、`default_joint_pos`、`action_scale`、`stiffness`、`damping`、`clip_actions`、`ctrl_dt`、`gait_frequency`、`gait_clock_mode`、`phase_init`。身份必须与 checkpoint 一致；关节相关向量按源关节顺序提供 29 项，初始双脚相位为弧度。checkpoint 本身没有完整控制契约，不能用旧 task 的姿态、scale 或 PD 默认值代替。
- 官方 profile 在创建环境前绑定 PD、控制周期和步态时钟；不必手工覆盖 `reward.gait_frequency`。G1 的可选 `env.position_actuator_gains` 使用 `joint_names/kp/kd`，MuJoCo 在冷路径按实际 actuator 对应关节名重排并应用，再建立环境增益缓存；未配置时保留模型默认值。该配置当前仅支持 MuJoCo，其他后端明确拒绝。加载后仍核对实际控制参数，不通过修改 profile 或关闭校验绕过不一致。
- 官方 64D 观测从公共物理状态接口组装：世界系角速度与重力转到基座系，关节位置减源默认角，按源关节顺序施加 checkpoint 预处理。不是截取旧 actor 观测的前 64 维，也不重复施加旧 actor 的缩放。完整 7D command 保留速度和四个步态 sin/cos；零速仍保留相位。
- source 归一化动作经 source clip 后作为快照与历史保存，再映射到目标关节顺序、默认角和 scale；PD、控制周期和相位时钟必须匹配 profile。冷启动及 reset 左补零并携带 bool mask，历史不跨 episode。相位持续变化不会被误判为新的速度指令而清空待成熟轨迹。
- 在线路径保留冻结 Planner/独立 base IDM、Q/V LoRA、固定伪标签、`Y_nom` 回归条件和原窗口/Adam/锚定/reset 语义。官方模型通过原生 command/mask 接口调用；旧 66/3 与既有 adapted checkpoint 的加载和播放路径保留。

CPU 诊断已通过正式 loader/factory/session 的推理、完整 K 步收集、实际 LoRA 更新、partial reset、旧路径及缺失/错配 profile 提前拒绝检查。诊断环境不含物理仿真；真实官方产物绑定、原生播放与负重效果尚未验收。

### 可选性能计时（第 6A 步）

在官方源训练命令末尾追加 `training.fada.performance.enabled=true` 即可开启；默认关闭。开关位于原生算法配置之外，不进入 checkpoint、缓存或 Oracle 身份，也不改变采样量、更新量、loss 或权重发布规则。

- 控制台在 warmup 完成、每轮 DAgger 和最终关闭后输出 `[FADA PERF]`；字段为累计值，比较相邻报告可得到本轮增量。最终返回值另含 `performance`。
- `wall_seconds` 覆盖初始化、采集 RPC、artifact 读取/Replay 入库、采样/H2D、IDM 与 Planner 前反向/optimizer、保存、发布和关闭。最终 `total_elapsed` 的范围为训练 owner 初始化至 runtime 关闭，不含 CLI 启动前的 Python 导入；整条命令的耗时应在实际对照时另外计入。
- `worker_wall_seconds` 保留真实资源初始化、同步、完整 rollout、CPU 打包和写盘耗时。rollout 包含原生 Oracle/shadow/环境事务，不冒充纯物理计算时间。父进程 RPC、worker 总时长与子阶段互相包含，不能直接相加。
- `cuda_stream_seconds` 仅在 CUDA 路径记录本进程当前 stream 的 event 间隔，不是整个 GPU 或 shadow 子进程的 kernel 总耗时；仅在报告边界统一解析，不逐算子强制同步。CPU 下此字段为空。
- `counters` 区分实际采集的原始 transition、抽取的训练窗口、学习更新和 online 轮数；窗口计数不是新增物理数据量。未执行的阶段不填假零值。

已用一套 CPU 诊断短链路验证计时关闭/开启时模型、两组 AdamW、loss、RNG、保存轨迹及发布版本逐项一致，走通 warmup 和两轮 DAgger。原生仿真、CUDA 计时及官方/TTT 性能对照尚未运行；本次没有实现或宣称训练加速。
