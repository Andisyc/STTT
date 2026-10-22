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

## Streaming TTT 在线实验

本仓库的在线 LoRA 实验入口为 `scripts/play_fada_online.py`，使用
`task=g1_walk_flat/mujoco recording_profile=straight_payload`，并通过
`training.play_checkpoint_path` 指定真实 Planner-IDM checkpoint。它与下述冻结权重的
ICE-Cal 路径不同：Planner 和 IDM 基础权重冻结，只有 IDM LoRA 接受在线更新。

- `online_adaptation.enabled=false` 保持无在线更新的 baseline；`true` 启用快照标签回归。
- 在 t 保存实际推理的历史输入、Planner 标称未来和首动作，在 t+1 用标称未来第一帧
  与真实关节位置偏移的差值构造固定动作标签。两边均为相对默认姿态的弧度偏移；
  用环境实际 `action_scale` 换算动作修正，不使用执行后的动作作为监督标签。
- 默认 `window_size=10`、`warmup_steps=40`、`update_interval=10`，
  `gamma=0.05`、`sigma=1e-4`。控制周期为 0.02s 时，第 40 步 (0.8s) 首次更新，
  使用启动阶段最后 10 条样本；之后第 50、60……步 (每 0.2s) 使用 10 条新样本更新。
  成功后仅清空训练样本，不重置推理历史或已学状态；失败则保留样本并抛出异常。
  非默认配置须同时满足 warmup、更新周期和满批条件；不等待负重挂载。
  这些数值是试验初值，不代表已证实的稳定性或适应效果。
- 损失为窗口内首动作误差平方范数的 `1/(2T)` 均值，加相对部署初始 LoRA 参数的
  显式 L2 锚定；不对动作维再平均。更新时关闭 dropout 但开启梯度，使用 Adam，
  不通过 AdamW 的默认权重衰减替代非零参数锚点。
- reset 清空历史、待闭合样本和训练窗口，并重新计算 warmup，但保留已学 LoRA、
  锚点与优化器状态。
  关闭 autoreset 时，终止后仍可继续物理步进，但暂停采集/更新，直到显式 reset。
- 旧在线 `lambda_forget` 配置和 `scripts/test_online_adaptation.py` 手动入口已移除。
  使用 `uv run --frozen --no-sync python scripts/play_fada_online.py`，启动日志应包含
  `objective=snapshot_regression`。视频录制、负重事件和离线 FADA 路径保持原接口。


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
