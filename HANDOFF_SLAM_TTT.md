# Handoff: SLAM-Guided Test-Time Training (Spatial TTT) via Differentiable World Models

> **Target Repository**: `/Users/chengyuxuan/ArtiIntComVis/STTT` (`https://github.com/Andisyc/STTT.git`)  
> **Source Session Context**: `/Users/chengyuxuan/ArtiIntComVis/TTT`  
> **Hardware Platform**: Unitree G1 Humanoid Robot (29 DoF)  
> **Document Status**: `AUTHORITATIVE` (System Design, Literature Synthesis & Research Blueprint)  
> **Date**: 2026-10-30

---

## 1. 核心问题定位 (The Core Problem)

### 1.1 现实痛点与动力学崩溃
人形机器人（Unitree G1）在真实世界部署时，面临剧烈的**分布外（OOD）物理扰动**：
- 典型场景：单侧肩部突加 3kg 偏载、电机热衰退、非对称传动间隙、地面摩擦突变。
- **物理后果**：左右腿推力不平衡产生持续的绕 Z 轴偏航外力矩（Yaw Moment），导致机器人严重偏航甩尾；若无自适应干预，欠驱动小支撑面的人形机器人将在 **2 秒内直接失稳摔倒**。

### 1.2 现有自适应范式的局限与断裂
1. **离线 FADA (Feature-Action Domain Adaptation) 的时序破产**：
   - 依赖“先在真实环境采集 2 分钟 Rollout，离线微调 LoRA，再重新上线”。在 3kg 强扰动下机器人 2 秒即跌倒，**根本无法在目标域收集到有效轨迹**。
2. **纯本体感觉 TTT (Proprioceptive Streaming TTT) 的“空间盲区”**：
   - 当前在 `TTT` 仓库中已验证成功的基座逆动力学映射 $\pi_{\text{base}}(q_{\text{predict}} - q_{\text{real}})$ 能够毫秒级吸收关节力矩缺口，保证机器人抗住 3kg 负重站立行走不摔倒；
   - **但本体感觉只能评价“局部物理舒适度”，对“世界坐标系下的绝对漂移”彻底失明**。机器人自认为力矩平衡，实际上正以 15 度的偏航角笔直走向悬崖。
3. **真实世界部署的“评价真空（Evaluation Vacuum）”**：
   - 仿真训练有上帝视角的 Dense Reward 实时打分；
   - 真机部署期没有任何外部 Reward 或动捕（Mocap）真值。好坏全靠人类肉眼观察，无法实现真正的具身自主闭环。

### 1.3 核心科学问题
**如何利用机载 SLAM 提供的真实空间几何轨迹，在真机部署阶段（Test-Time）构建完全自主的自监督/监督闭环，在线更新人形机器人运动策略网络（LoRA 权重），在几秒内彻底消除物理偏载导致的宏观偏航漂移？**

---

## 2. 方案演进与理论破局 (Solution & Conceptual Framework)

### 2.1 为什么不能用常规思路？（排除伪路径）
- **排除路线 A：传统强化学习在线试错（Model-Free RL / PPO）**  
  双足机器人在挂载 3kg 偏载瞬间处于翻滚边缘，任何探索噪声（Exploration Noise）都会直接导致硬件摔裂碎件。真机**零试错容忍度**，无模型 RL 彻底出局。
- **排除路线 B：将 SLAM 位姿直接作为策略输入（Observation-Only）**  
  这属于传统感知运控，策略权重完全冻结（$\nabla_\theta = 0$），没有任何在线学习能力，面对未见过的重物依然无法抵消动力学偏置。
- **排除路线 C：直接拿 SLAM 轨迹对物理真机做 `Loss.backward()`**  
  从电机动作 $u$ 到 SLAM 轨迹 $X$，中间隔着整整四道不可导的硬墙（电机延迟与死区 $\to$ 足端非平滑撞击与单向接触力学 $\to$ 传感器光子/点云离散采样 $\to$ **SLAM 内部的非凸 C++ 高斯-牛顿优化求解器**）。真实世界不可导，PyTorch 链式法则瞬间断裂。
- **排除路线 D：经典解析雅可比转置 $J^T(q)$ 投影**  
  在双足行走的高频接触断裂（单双支撑相切换、冲击奇异性）与浮动基座欠驱动下，解析 $J^T$ 发生除零发散，在真机上算不出来。

### 2.2 理论破局：构建“动作条件可微空间世界模型（Differentiable Spatial World Model）”
我们不强求真实物理世界和 SLAM 求解器可导，而是在 GPU 内存中为它们建立一个**“紧凑、可导的神经物理透镜”**：
1. **输入端**：当前本体感觉状态历史 $S_t^H$（触地相位、关节角）+ 未来候选动作序列 $a_{t:t+H-1}$；
2. **输出端**：直接预测未来的**相对空间位姿流形增量（Relative SLAM Trajectory Delta）** $[\Delta x, \Delta y, \Delta \text{yaw}]_{1:H}$；
3. **求导闭环**：SLAM 测出的几何测地线误差作用于世界模型输出端，通过**时间反向传播（BPTT）**，世界模型自动输出平滑、无奇异点的**一阶解析“神经雅可比（Neural Jacobian）”**，无损倒灌进策略网络的 LoRA 权重中。
4. **确定性单向收敛**：根据经典自适应控制的 KYP / 严格正实性（SPR）引理，只要神经雅可比的方向符号正确（如左腿向后蹬产生右转倾向），策略权重将以确定性、零探索、单向收敛的方式把偏航力矩彻底吃掉。

---

## 3. 先前检索文献对比：SLAM + 腿足运控 (Literature on SLAM & Locomotion 2024–2026)

| 工作 | 论文与单位 | 核心机制 | 与本课题的本质界限 |
| :--- | :--- | :--- | :--- |
| **Gallant** | CVPR 2026 (Nov 2025)<br>上海 AI Lab (InternRobotics) / 清华<br>Unitree H1 / G1 人形机器人 | 机载 3D 激光雷达跑 **Fast-LIO SLAM** 构建局部动态体素网格，将体素特征与航点输入给 Transformer RL 运控策略。 | **权重绝对冻结（$\nabla_\theta = 0$）**。<br>SLAM 仅作为策略的前向感知输入（Observation），**没有任何在线学习与梯度更新**。面对重载偏航依然无能为力。 |
| **Wild Visual Navigation (WVN)** | Science Robotics 2025 / 2024.10<br>ETH Zurich (RSL) / Oxford<br>ANYmal 四足机器人 | 提出野外自监督导航：机器人利用本体感知的足端触地力矩反馈，在线更新视觉可通行性地图。 | **因果流向完全相反**。<br>WVN 是**“本体感知 $\to$ 监督外感知地图”**，底层运控策略完全固定；而我们是**“外感知 (SLAM) $\to$ 监督本体运控策略”**。 |
| **Legged Parkour / LucidSim** | CoRL 2023 / CoRL 2024<br>CMU / MIT<br>Unitree Go1 四足 / 双足跑酷 | 机载深度相机与 T265 视觉里程计提供机体位姿跟踪，指导策略选择起跳与落脚点。 | **纯 Sim-to-Real 开环策略部署**，没有真实世界测试时自适应机制。 |
| **GeoFlowSlam** | IROS 2025 最新开源<br>地平线 (Horizon Robotics) / HKUST | 专为腿足剧烈晃动设计，将腿部正向运动学与 RGBD-Inertial 紧耦合融合。 | **纯状态估计定位算法（State Estimator）**，不包含任何控制或策略微调。可作为本课题高精度 SLAM 话题来源。 |

---

## 4. 核心理论背书：《SuperMimic》(CoRL 2026) 深度解析

- **论文信息**：*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (Qiayuan Liao, Pieter Abbeel, Koushil Sreenath - UC Berkeley, CoRL 2026, arXiv:2610.28480v1, 2026-10-20)
- **硬件本体**：**Unitree G1 人形机器人 (29 DoF)**。
- **核心贡献**：
  - 证明了在 Unitree G1 人形机器人上，模型无关强化学习（PPO）由于无法求导，需要 128 张 L40 GPU 暴力跑 7 天（SONIC 工作）；
  - SuperMimic 训练了一个**可微前向世界模型（Differentiable World Model）** $s_{t+1} = f_\omega(s_t, a_t, \xi)$，通过在短时程想象视界（$H=32$ 步）内执行 **BPTT（时间反向传播）**，为策略提供一阶解析梯度（Analytic Policy Gradient）；
  - **训练效率提升超 1000 倍**：仅用 1 张 H100 训练 3 小时，即超越了 128 卡跑 7 天的效果，且在真机 G1 上成功实现了高动态全身模仿。
- **与本课题的继承与跃升**：
  - **继承点**：SuperMimic 权威证明了“用神经网络充当物理世界可微代理、通过 BPTT 反传解析雅可比更新 G1 策略”在物理上是确定可行的；
  - **跃升点**：SuperMimic 仅用于**仿真阶段替代 PPO 离线加速**，真机部署时依然冻结；**本课题首次将可微世界模型推向真实物理世界部署阶段（Test-Time Training）**，直接连接真实 SLAM 闭环几何真值。

---

## 5. 世界模型必须如何改造？（从 60 维状态到 SLAM 轨迹）

不能照搬 SuperMimic 的世界模型，必须进行**针对 SLAM TTT 的重构**：

### 5.1 SuperMimic 世界模型的局限
SuperMimic 的世界模型预测的是机器人全身的机体局部状态序列：
$$s = [q, \dot{q}, v_x^b, v_y^b, \omega_z^b, g^b] \in \mathbb{R}^{60+}$$
- 预测 60 维高维关节状态计算量大；
- 如果再在外面挂一个手写的运动学积分层去算空间轨迹，既臃肿不美观，又无法应对真实世界中 3kg 负重下的脚底摩擦滑移。

### 5.2 改造方案：端到端相对 SLAM 轨迹世界模型 (Ego-Centric SLAM World Model)
**砍掉所有高维关节状态预测，世界模型直接输出局部关键帧下的 6-DoF 相对运动流形！**

```text
       输入：[ 当前本体感觉历史 S_t^H (包含触地接触相位) ] + [ 候选动作序列 a_{t:t+H-1} ]
                                         │
                                         ▼
                     【极简时空相对世界模型 M_ϕ】 (例如 3层 MLP / GRU)
                                         │
                                         ▼ 纯净紧凑输出
                     预测未来 H 步的局部 SLAM 相对位姿流形：
                     [ (Δx, Δy, Δyaw)_1,  (Δx, Δy, Δyaw)_2,  ...,  (Δx, Δy, Δyaw)_H ]
                                         │
                                         ▼ (与机载 Fast-LIO 实测的期望相对位姿做差)
                     Loss_SLAM = Σ || [Δx, Δy, Δyaw]_k - [Δx*, Δy*, Δyaw*]_k ||²
                                         │
                                         ▼ BPTT 一键求导 (仅更新策略 Tracker-LoRA)
```

#### 为什么必须这样改造？
1. **模型极度紧凑（60 维降到 3 维）**：只关心动作对机体在空间坐标系下的相对航向影响，前向与反向耗时降到微秒级；
2. **隐式吸收未建模物理（Implicit Physics）**：训练数据直接来自真机/仿真的 SLAM 位姿增量，模型天然将足端形变、触地打滑等手写公式算不出来的效应隐式拟合进网络；
3. **短时程（Short Horizon）严格受控**：视界控制在 $H = 10 \sim 20$ 步（0.2s ~ 0.4s），刚好覆盖一个步态支撑相位，保证动力学局部强凸且不发生梯度弥散。

---

## 6. 检索文献佐证：直接预测空间轨迹的 2024–2026 先例

| 论文 | 出处 / 单位 | 验证事实 |
| :--- | :--- | :--- |
| **MWM: Mobile World Models** | CVPR 2026 最新发表 (2026.03)<br>北京大学团队 | 提出专门针对移动机器人的**动作条件世界模型（Action-Conditioned World Model）**，直接以动作序列预测相对位姿变化与空间流形。证明了预测空间相对增量（Delta SE(2)）的方差远低于预测高维局部状态。 |
| **No More Marching** | IEEE Humanoids 2025 / 2026.04<br>Oregon State Univ (Alan Fern 团队) | 针对双足人形机器人，严格证明了在短时程视界内，双足电机的动作序列与局部 $SE(2)$ 位姿增量 $(x, y, \psi)$ 具有极高信噪比的单射关系。 |
| **VertiAKD** | 2026.09 最新预印本 | 在未知非平稳地形与重力偏载下，利用动作条件运动动力学模型直接预测轨迹漂移，证实了**未知物理扰动下动作到空间位移的导数方向依然保持严格的一致性（SPR 成立）**。 |
| **Do Robotic World Models Follow Actions?** | 2026.08 理论诊断研究 | 从信息论角度证明：以积分为基础的空间轨迹在短时间窗口（$H \le 20$）内具备优异的局部李普希茨连续性，是避免 BPTT 梯度爆炸的最优表征。 |

---

## 7. 下一步行动清单 (Immediate Execution Tasks for STTT)

迁移至新会话后，直接在 `/Users/chengyuxuan/ArtiIntComVis/STTT` 启动以下工作：

1. **[数据准备] 收集动作-位姿增量训练对**：
   - 在已验证平衡的基座策略下，记录 `(proprioceptive_history, action_sequence)` 与对应的机载 Fast-LIO `[delta_x, delta_y, delta_yaw]` 增量数据。
2. **[模型构建] 构建极简可微 SLAM 世界模型 `M_phi`**：
   - 3 层 MLP 或轻量因果 GRU，输入 29 维动作序列，输出 $H=15$ 步的 3 维相对位姿增量。在 PyTorch 中验证其 Autograd 梯度流畅度。
3. **[双环损失打通] 组装在机自监督流式 TTT 循环**：
   - **外环**：$L_{\text{SLAM}} = \text{BPTT}(\hat{\mathcal{T}}_{\text{slam}} - \mathcal{T}_{\text{ref}})$（消除长程累积偏航）；
   - **内环**：$L_{\text{balance}} = \text{MSE}(\pi_\theta - \pi_{\text{base}})$（保住直立与接触安全下限）。
4. **[真机 3kg 偏载实验验证]**：
   - Unitree G1 真机左肩悬挂 3kg，运行闭环自适应，记录前 5 秒偏航到第 15 秒直线收敛的轨迹对比曲线。
