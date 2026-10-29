# Figure-Eight Demo Step 1 Handoff — Automatic Command Injection

生成时间：2026-10-29  
当前仓库：`/Users/chengyuxuan/ArtiIntComVis/TTT`  
交接对象：负责在 TTT 仓库实施本方案的 Agent

---

## 0. 交接摘要

**当前唯一目标：在 TTT 仓库实现“自动八字速度指令”，并接入现有 `FADAPlaybackSession`。**

这一步只解决 Demo 的指令来源：

- 根据闭合八字参考路线生成分段恒定的三维速度指令 `[vx, vy, yaw_rate]`。
- 播放时按实际控制步把指令写入 `env.state.info["commands"][:, :3]`。
- 旧 Locomotion checkpoint 与后续官方 Locomotion checkpoint 使用同一套 Demo。
- 本步骤不实现脚印、不调整镜头、不录长视频、不改训练或在线更新算法。

后续总体方案还有：

1. **Step 1（本交接）**：自动八字指令。
2. **Step 2**：真实左右脚印记录。
3. **Step 3**：预览/录制共用绘制、高位固定镜头与完整八字 profile。

**不要把 Step 2/3 顺手实现。**

---

## 1. 权威当前代码状态

下面事实来自本会话在 2026-10-29 对当前 checkout 的直接阅读。接手时仍应按这些路径和符号重新读当前文件，因为仓库可能已有变化。

### 1.1 播放 Session 已经有外部速度指令接口

权威文件：`src/unilab/visualization/playback_sessions.py`

`RslRlPlaybackSession.set_external_command()` 已存在，当前范围约在第 260–277 行：

- 从 `self.env.state.info["commands"]` 找到命令 owner rows。
- 通过 `_external_velocity_command_rows(command, commands)` 保持命令 shape。
- 写入 `commands[:, :3] = command_arr`。
- 调用 `env.refresh_state()`。
- 调用 `self.refresh_observation()`，即 `wrapped_env.get_observations()`。

这意味着 **Demo 不需要直接操作 backend 或构造官方 7 维 command**。三维速度写入后，官方适配器会继续补相位。

### 1.2 FADA 策略调用已能分流 legacy / official

同文件 `FADAPlaybackSession`，当前范围约第 342–408 行：

- `reset()` 先调用 `controller.reset()`，再调用父类 reset，最后对 `official_adapter` reset。
- `step_once()` 先用当前 observation/command 计算动作，再执行环境，再调用 `controller.on_env_step()`。
- `_fada_policy()` 读取 `self.info["commands"]`。
- 若存在 `official_adapter`：
  - `official_adapter.observation()` 构造官方紧凑观测；
  - `official_adapter.command(commands, gait_phase)` 把三维速度扩成官方 7 维 command；
  - `official_adapter.env_action()` 把源动作映射到目标关节。
- 否则把 `observation, commands` 直接传给 legacy controller。

因此 **自动指令模块只应产生三维速度，不应关心 checkpoint 类型**。

### 1.3 在线采样会把速度变化当作命令变化

权威文件：`src/unilab/algos/torch/distill/fada/online_collector.py`

`OnlineTrajectoryCollector.begin()` 当前范围约第 111–133 行：

- legacy：整个 command 改变时，清空 in-flight futures 与 future deque。
- official：只比较 `snapshot.command[:, :3]`；连续变化的 `sin/cos` gait phase 不视为改变。
- 命令变化只清除尚未成熟的 futures，不清除已成熟 samples。

工程含义：

- 不要按每个控制步连续改转向速度；应使用**分段恒定指令**。
- 指令保持时间必须足以让 prediction horizon 的 future 完成；当前 checkpoint 常用 horizon 是 `K=6`。
- 预计控制周期 50 Hz 时，每步 0.02 s，K=6 只需要约 0.12 s。为覆盖在线 collection/update 的窗口，建议默认每段至少约 `0.8 s`（40 控制步），最终以配置校验为准。

### 1.4 环境命令可能被环境逻辑覆盖

权威文件：`src/unilab/envs/locomotion/g1/walk_control_bindings.py`

当前事实：

- `_commit_command_state_for_next_observation()` 会采样并发布下一步命令。
- `_sample_next_commands()` 在 `env.commands.resampling_time > 0` 时按 interval 重采样。
- `heading_command=True` 时会执行 heading feedback，改写角速度。
- dead-zone 和 `small_xy_threshold` 可能把小速度清零。

工程含义：

- figure-eight 自动指令启用时，必须让环境随机重采样关闭：`env.commands.resampling_time: 0.0`。
- 必须关闭 heading feedback：`env.commands.heading_command: false`。
- Demo 配置应显式确认 dead zone / small threshold 不会把设计的 yaw rate 清零；若默认值不足以清除，应在校验中 fail-fast。
- `set_external_command()` 已写入并刷新 observation；但如果在 `wrapped_env.step()` 之后还有 owner 会在同一步覆盖命令，Step 1 实施时必须核对完整 step 顺序。当前证据表明更新发生在 completed-action 的 commit 阶段；会话在下一步策略计算前注入是正确位置。

### 1.5 播放入口已经统一旧/新 checkpoint

权威文件：`src/unilab/visualization/playback_distill_sessions.py`

`create_fada_playback_session()` 当前范围约第 345–527 行：

- 解析 checkpoint。
- `LoadedOfficialFADAStudentPolicy` 触发 official path：
  - 读取 `official_playback.profile_path`；
  - 创建 `OfficialFADADeploymentAdapter`；
  - 创建 `OfficialFADAPlaybackController`。
- legacy path 创建 `FADAPlaybackController`。
- 两者最终都进入同一个 `FADAPlaybackSession`。

结论：**Demo 应绑定在 `FADAPlaybackSession` 层，不要放进 official adapter、legacy controller 或 checkpoint loader。**

### 1.6 现有 recording/viewer 不在本步修改

`scripts/play_fada_online.py` 和 `src/unilab/visualization/fada_recording.py` 本轮只需知道：

- 两者都通过 `session.step_once()` / `session.advance()` 推进。
- 相机配置已有 `camera_follow_body`、`camera_distance`、`camera_elevation`、`camera_azimuth`。
- 八字高位镜头和脚印绘制属于 Step 3，不在 Step 1 实现。

---

## 2. 本次实施边界

### 必须实现

1. 一个纯函数/小类组成的 figure-eight 指令生成器。
2. 一个 Hydra 配置节点，默认关闭。
3. 播放 Session 的注入点。
4. 与环境命令自动重采样/heading feedback/键盘控制的冲突防护。
5. 针对路线几何、指令调度和在线样本成熟性的最小验证。

### 明确不做

- 不修改 `OnlineTrajectoryCollector`、LoRA updater、loss、训练数据或 Oracle。
- 不修改 checkpoint loader、official controller、deployment adapter 的语义。
- 不实现脚印、足迹、预测轨迹或高位镜头。
- 不添加位置闭环纠偏控制器。
- 不处理 RoboJuDo_Real 仓库。
- 不创建 Git 分支、不提交、不推远端、不启动长仿真或训练。

---

## 3. 建议实现方案

### 3.1 新模块

建议新增：

`src/unilab/visualization/demo_commands.py`

职责只有一个：**把控制步进度映射为三维速度命令。**

建议接口形态（名称可调，职责不变）：

```python
@dataclass(frozen=True)
class FigureEightCommandConfig:
    width_m: float
    height_m: float
    speed_m_s: float
    segment_duration_s: float

class FigureEightCommandProgram:
    @classmethod
    def from_config(cls, config: FigureEightCommandConfig) -> "FigureEightCommandProgram": ...
    @property
    def duration_s(self) -> float: ...
    def command_at_control_step(self, control_step: int) -> np.ndarray: ...  # shape (3,)
```

### 3.2 曲线生成

用参数曲线表示八字，例如：

\[
x(s)=a\sin(s),\qquad y(s)=b\sin(s)\cos(s)
\]

其中 `width=2a`，`height=b`。这类曲线在中心点连续且通过原点，不必使用会在参数变化下出现非物理切线的普通二次曲线。

实现要求：

1. 用固定数量的样本预计算参考曲线和弧长。
2. 按配置速度把整条路线换算成总时长。
3. 将曲线切成等控制时长的分段。
4. 每段根据相邻参考点计算：
   - `vx = |位移| / 时长`，作为前进速度；
   - `yaw_rate = wrap(角度变化) / 时长`；
   - `vy = 0`。
5. 轨迹长度和单段时长必须保证所有段长有限且为正。
6. 初始切线应与 G1 初始前进方向一致；不要在 Demo 里旋转或重置机器人。

不要只让 yaw rate 正弦变化；这种开放式控制不保证闭合八字，也不能保证回到初始朝向。

### 3.3 分段保持时间

配置建议：

```yaml
demo:
  figure_eight:
    enabled: false
    width_m: 6.0
    height_m: 3.0
    speed_m_s: 0.3
    segment_duration_s: 0.8
```

校验要求：

- `width_m > 0`、`height_m > 0`、`speed_m_s > 0`、均为有限值。
- `segment_duration_s > 0`，并且换算成整数控制步后至少 `prediction_horizon` 步，推荐不少于 `update_interval + prediction_horizon`。
- 指令值必须落在当前环境的 `env.commands.vel_limit` 允许范围内；尤其是 yaw rate。
- 若开启在线采样，`segment_duration_s` 必须保证切换后至少能成熟一个窗口；不够时明确 `ValueError`，不要静默清空样本。

### 3.4 Session 注入点

在 `FADAPlaybackSession` 增加可选 command program，而不是在 `_fada_policy()` 里写复杂逻辑：

建议：

- 构造函数接受 `command_program: Any | None = None`。
- `reset()` 时重置 program 的 episode 进度。
- `step_once()` / `_build_actions()` 之前，若 program 存在：
  1. 用当前 episode control step 得到 command；
  2. 调用 `set_external_command(command)`；
  3. 再正常 `_build_actions()`。
- `step_once()` 成功后推进一次 program；环境 done/autoreset 时重置 program。

注意：

- 当前 `step_count` 是累计值，不能直接当 episode-local command phase，除非它能满足 reset 语义。建议 command program 自己维护 episode step 或明确与 session reset 对齐。
- `advance()` 受 `PlaybackControls` 控制；没有 step permission 时绝不能推进指令。

### 3.5 环境命令冲突

`create_fada_playback_session()` 中，在创建 env 后、创建 session 前检查：

- `env.commands.resampling_time` 必须为 0，否则自动命令会被随机重采样覆盖。
- `env.commands.heading_command` 必须为 false，否则 heading feedback 会改写 yaw。
- `interactive.keyboard` 必须为 false，否则两个速度来源冲突。
- command dead zone / small threshold 不得吞掉 figure-eight 的指令。

优先通过配置校验 fail-fast；不要在 Demo 层偷偷覆盖用户配置，除非选择 profile 时已明确写入这些值。

---

## 4. 文件级实施计划

### Owner A：纯指令生成器

**文件**：新增 `src/unilab/visualization/demo_commands.py`

**Change**：
- 定义 figure-eight 配置与 program。
- 预计算闭合参考曲线、弧长、分段速度表。
- 提供 reset / command query 语义。

**Why**：
- 让八字路线可离线验证，并与 MuJoCo、网络和策略类型解耦。

**Preserve**：
- 不接触 checkpoint、网络 forward、action mapping、backend 或 XML。

**Done when**：
- 纯 NumPy 离线检查能验证：
  - 指令表非空、有限；
  - 所有速度在配置/环境限制内；
  - 按分段速度积分后回到起点；
  - 末段朝向回到初始朝向；
  - 每段保持时间符合配置。

### Owner B：Session 注入

**文件**：`src/unilab/visualization/playback_sessions.py`

**Change**：
- `FADAPlaybackSession` 持有可选 `command_program`。
- reset、step、done/autoreset 时同步 program 生命周期。
- 每次策略查询前通过 `set_external_command()` 写入当前 command。

**Why**：
- 保证旧/新 FADA 共用同一速度输入，且不绕过现有 observation refresh。

**Preserve**：
- 不改变 `_fada_policy()` 的 legacy/official 分流。
- 不改变 `controller.act()`、`controller.on_env_step()` 的接口。

**Done when**：
- 开启 program 时，`env.state.info["commands"][:, :3]` 在每次策略调用前反映当前分段。
- 关闭 program 时，现有行为逐字节不变。

### Owner C：配置与 Session 创建

**文件**：
- `conf/distill/config.yaml`
- `src/unilab/visualization/playback_distill_sessions.py`

**Change**：
- 新增 `demo.figure_eight` 配置节点，默认 `enabled: false`。
- 在 `create_fada_playback_session()` 中构造 program 并注入 session。
- 对 resampling、heading、keyboard、velocity limits 做 fail-fast 校验。

**Why**：
- 保持配置驱动，不修改现有默认播放。
- 避免多个命令来源同时写 `info["commands"]`。

**Preserve**：
- 不改 `recording_profile/straight_payload.yaml`。
- 不改 `official_fada.yaml` 的训练语义。

**Done when**：
- 默认配置不创建 program。
- `demo.figure_eight.enabled=true` 时 session 获得 program。
- 冲突配置产生明确错误。

---

## 5. 最小验证计划

本步完成后只做**一次针对性验证**，不要把多个测试套件打包运行。

建议按风险最近处验证：

1. **离线几何检查**
   - 用临时脚本调用 `demo_commands.py`。
   - 检查八字闭合、有限速度、有限 yaw rate、段保持时间。
   - 不打开 MuJoCo viewer，不录视频。

2. **短程真实 session 检查**
   - 用现有旧 checkpoint 和 MuJoCo backend。
   - 时长只需跨越至少两个指令切换点。
   - 断言/观察：
     - 每次 policy 调用前的 command 与 program 当前段一致；
     - 在线 adaptation 开启时仍能成熟至少一个 future/window；
     - `demo.figure_eight.enabled=false` 时 command 不会被 program 覆写。

3. **默认回归**
   - 仅运行能覆盖 session/config 创建的最小既有测试或临时 smoke。
   - 不运行 `make test-all`，除非用户明确要求。

若实施期间发现真实 step 顺序会在策略调用后、环境 commit 阶段覆盖外部命令，立即停止扩展，回到 system-engineering 更新路由；不要用每步写两次命令的方式掩盖。

---

## 6. 证据与当前判断的限制

已核实：

- 旧 checkpoint 和新 official checkpoint 可以由同一个 `FADAPlaybackSession` 播放。
- `set_external_command()` 是统一三维速度入口。
- 在线收集器对 command 变化的语义已经明确。
- 环境 resampling 与 heading feedback 可能覆盖命令，必须在配置层关闭或校验。

未在本步验证：

- 实际旧 checkpoint 能否稳定走完完整八字；这需要 Step 3 完成录制后评估。
- official checkpoint 的真实播放兼容性取决于 `official_playback.profile_path` 和对应源控制 profile；本步只保证三维速度入口一致。
- RoboJuDo_Real 的接口尚未核对，不能假设可直接复用文件。

---

## 7. 与后续步骤的接口预留

Step 1 实施时不要提前实现后续内容，但接口不要挡住它们：

- command program 应能暴露当前段索引、仿真时间和当前三维速度，便于 Step 3 overlay 显示。
- 脚印记录属于 Step 2，将放在控制步之后采样 `get_foot_pos()` 与 contact sensors；不要在 Step 1 里把 foot/contact 逻辑塞进 command program。
- 高位固定镜头属于 Step 3，不应改 command 逻辑。

---

## 8. 给执行 Agent 的首条指令

先读以下文件的相关区域，再修改：

1. `src/unilab/visualization/playback_sessions.py`：`set_external_command`、`FADAPlaybackSession`
2. `src/unilab/visualization/playback_distill_sessions.py`：`create_fada_playback_session`
3. `src/unilab/algos/torch/distill/fada/online_collector.py`：`OnlineTrajectoryCollector.begin`
4. `src/unilab/envs/locomotion/g1/walk_control_bindings.py`：命令 resampling / heading feedback
5. `conf/distill/config.yaml`：interactive / online_adaptation / recording_profile 配置结构

然后只实施 Step 1：自动八字指令生成、配置、Session 注入和一次短程验证。完成后报告改动文件、验证命令与结果；不要扩展脚印或镜头。
