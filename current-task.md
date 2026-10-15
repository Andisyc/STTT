# Current task

Goal: 在现有 FADA 离线链路基础上，新增在线 Streaming TTT 链路（在线轨迹收集 + 在线更新机制），实现挂载事件触发的 LoRA-IDM 在线适应。
Do not: 不修改现有离线 FADA 链路的行为；不引入变化检测、瞬态掩蔽、影子验证等扩展机制（原始 idea 不包含）；不处理 Planner 更新；不实现 RLS（仅 weighted SGD）。

Current identity:
- 基座代码: FADA_Unofficial 复制，离线 LoRA 链路完整
- 目标链路: 新增在线推理 + 在线适应的闭环

Confirmed decisions:
- 三轴框架: 架构解耦（已有）+ 满秩轨迹采集（新增）+ 指数遗忘（新增）
- 数据收集: 跨相位轨迹窗口，非单步 streaming
- 更新机制: 指数加权最小二乘，非 RLS
- Planner 冻结，仅更新 IDM LoRA
- 仿真负重通过 `SimBackend.set_body_mass_and_com` 设置每个环境的绝对质量与 body-local 质心；事件 owner 负责在 episode reset 时显式恢复基线
Progress:
- [x] 步骤 1: 接口定义 — Done when: $A_{real}$ 来源与轨迹窗口格式明确
  - Evidence: $A_{real}$ 从 `env.step()` → `info["current_actions"]` 获取；轨迹窗口复用 `FADACausalWindow` 结构
- [x] 步骤 2: 在线轨迹收集器 — Done when: 实时构建跨相位 FADATargetBatch
  - Evidence: `online_collector.py` 创建，测试通过（窗口形状正确）
- [x] 步骤 3: 在线更新机制 — Done when: 指数加权 LoRA 更新运行
  - Evidence: `online_updater.py` 创建，测试通过（loss: 0.718，指数加权正常）
- [x] 步骤 4: 在线链路集成（当前四步路线的第 3 步） — Done when: FADA session 的实际动作经过 online controller，transition 在每个 step 后进入 collector
  - Evidence: `uv run --frozen --no-sync pytest tests/visualization/test_fada_online_playback.py -q` → 13 passed；schema 5 真实 checkpoint 经官方 session + fake env 连续执行 35 步后完成一次 LoRA 更新，loss=0.345505625，IDM 恢复 eval；未启动 physics/Viewer
- [x] 步骤 5: 端到端录制链路 — Done when: 无窗口执行真实 MuJoCo、负重事件与在线 LoRA 更新，并输出可播放 MP4
  - Evidence: `scripts/play_fada_online.py` 默认无窗口录制；本地 CPU + schema 5 checkpoint 实跑 0.8s / 40 个控制步，3 kg 负重在 0.22s 的 post-step 状态可见，0.8s 完成一次 LoRA 更新（6 个窗口）；输出 H.264 MP4（320x240、10 fps、8 帧），已查看视频预览
- [x] 负重步骤 1: Backend 运行时接口 — Done when: 不重置 physics state 即可按环境设置 body 质量与质心
  - Evidence: `SimBackend.set_body_mass_and_com`、MuJoCo `BatchEnvPool.set_field_indexed` 实现；聚焦测试 1 passed
- [x] 负重步骤 2: G1 环境负重事件 — Done when: 按环境定时挂载负重，并在 episode reset 前恢复标称质量与质心
  - Evidence: `G1RuntimePayload`、`env.runtime_payload` 配置与 `G1WalkEnv` 生命周期接入完成；`uv run --frozen --no-sync pytest tests/envs/locomotion/g1/test_runtime_payload.py -q` → 4 passed
- [x] 服务器录制入口：`video_recording` 配置时长/FPS/分辨率/路径，逐帧编码、不按墙钟 sleep、不累积整段 RGB；Linux 默认 EGL，允许通过 `MUJOCO_GL` 指定其他后端；`video_recording.enabled=false` 保留本地 Viewer
- [x] 直行评估录制：`recording_profile=straight_payload` 固定标称 reset、零初速度与 `[0.4, 0, 0]` 命令，关闭随机扰动，侧后方 135° 跟随相机；episode 上限 180s，120s 视频在 30s 加载 3kg，负重后观察 90s；真实 termination/reset 保留并在画面和日志标记
  - Evidence: 本地 CPU + 真实 checkpoint 录制 0.8s / 40 步（仅为 smoke，将负重时间缩为 0.2s），输出 640x360 H.264；已查看视频，确认无负重/负重状态切换、挂载提示、横向位移/航向偏差、reset 次数和侧后方视角
  - Protocol: 先用同一 profile 覆盖 `online_adaptation.enabled=false env.runtime_payload.enabled=false video_recording.duration_s=30`，检查原策略无负重直行；通过后再录 120s 的在线适应视频。后者保持现有从开局在线更新的行为；要判断适应的因果效果，还需同样负重、关闭更新的对照
Blockers:
- 服务器 30s / 1500 步录制已完成（已从 tmux 日志确认）；需使用 `LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1` 避免 mujoco-uni 异常展开符号冲突。新 profile 的 30s 基线及 120s 负重行为效果尚未验证

Next:
- 上传新 profile 与叠字录制代码；先检查 30s 无负重、无更新基线的横向/航向偏差和 reset 次数，再决定是否运行长时负重适应对照
