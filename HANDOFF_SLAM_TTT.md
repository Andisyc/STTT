# Handoff: SLAM-Guided Spatial Drift Correction for Humanoid Locomotion

> **历史交接说明（2026-11-06）**：当前问题与候选方法请优先阅读 [`HANDOFF_SLAM_PATH_FOLLOWING.md`](HANDOFF_SLAM_PATH_FOLLOWING.md)。用户已明确目标是“偏离后回到参考路径”，不是仅消除偏航或接触冲量泄漏。下文保留为历史记录，其 `AUTHORITATIVE` 标记不再代表当前方法合同；物理定因、绝对安全和收敛保证不得直接用于实现。

> **Target Repository**: `/Users/chengyuxuan/ArtiIntComVis/STTT` (`https://github.com/Andisyc/STTT.git`)  
> **Source Session Context**: `/Users/chengyuxuan/ArtiIntComVis/TTT`  
> **Underlying Baseline Paper (CVPR 2027)**: `/Users/chengyuxuan/ArtiIntComVis/CVPR2027` (*"Walk It Off: Test-Time Training for Humanoid Locomotion"*, OAT Method)  
> **Hardware Platform**: Unitree G1 Humanoid Robot (29 DoF)  
> **Document Status**: `AUTHORITATIVE` (Rigorous Concept Mapping, Theoretical Boundaries & Next-Session Protocol)  
> **Date**: 2026-11-06

---

## 1. 已确立的事实与基准 (Established Baseline & Ground Truth)

1. **第一项已完成成果 (OAT / Tracker TTT)**：
   - 载体：逆动力学模型（IDM / Tracker $I_{\psi,\theta}$）。
   - 机制：纯本体感觉流式 TTT，利用冻结基座模型 $I_{\mathrm{ref}}$ 构造动作差分伪标签：
     $$\tilde{a}_i = \operatorname{sg}\Bigl[ a_i + \gamma \Pi_1 \big( I_{\mathrm{ref}}(\hat{Y}_i^K) - I_{\mathrm{ref}}(Y_{i,\mathrm{exec}}^K) \big) \Bigr]$$
   - 物理实测表现：在左肩挂载 3kg 不对称偏载下，系统在 50Hz 毫秒级消除动力学力矩缺口，**完全恢复标称姿态，恢复短距离（3~5米）直线行走能力，100% 保证不跌倒**。

---

## 2. 本课题要解决的唯一核心问题 (The Unsolved Problem)

### 2.1 物理现象与时空尺度差异
- **短距离直 $\neq$ 长距离直**：机器人在已校准 Tracker 的保护下虽然短距离走直且不跌倒，但当行走距离拉长到 **30~100 米**时，机体在三维世界系中出现**单调累积的航向角漂移（Yaw Drift）与横向偏离（Cross-Track Error）**。

### 2.2 物理与接触力学根源 (Contact-UAN 铁证)
- 依据 MIT CoRL 2026 最新论文《Contact-UAN》：MuJoCo 采用软接触惩罚模型（`solref`, `solimp`），而 G1 机器人脚底是无被动柔顺的一整块刚性平板。
- **仿真存在度数级、毫米级的接触法向失真**：脚掌在硬地面着地瞬间只要存在 $1^\circ$ 的微小俯仰/侧倾偏差，接触就会退化为边缘线接触，导致**足底抗扭转摩擦力矩（Torsional Friction $T_z$）骤降**；
- 加上 3kg 负重导致的左右法向力不平衡，左右脚交替蹬地时，**每一个步态周期都在机体竖直轴上泄漏出一个微小但非零的净角动量冲量（$\Delta H_{z,\text{stride}} = \delta \neq 0$）**。

### 2.3 教材权威形式化定义 (Textbook Problem Formulation)
- **移动机器人学（Siegwart《自主移动机器人导论》第 3.4 节）**：  
  定义为 **“非完整移动机器人的系统性航位推算二次方累积发散问题 (Systematic Dead-Reckoning Divergence)”**。系统性接触偏差 $\epsilon_\theta$ 被行进距离 $s$ 连续积分，导致横向位置发生二次方发散：$y(s) \approx \frac{1}{2} \epsilon_\theta s^2$。
- **足式力学（Tedrake《欠驱动机器人学》）**：  
  定义为 **“双足浮动基未驱动自由度（Unactuated Floating-Base DoFs）的开环漂移问题”**。水平面内的 $(x, y, \psi)$ 没有天然弹簧恢复刚度，单步冲量泄漏被开环时间积分放大。

---

### 3. 已彻底否定并封死的伪路径 (Discarded False Paths)

1. **绝对不拿 SLAM 梯度微调 Tracker**：  
   Tracker 负责 50Hz 关节力矩平衡（毫秒级快环），SLAM 是 10Hz 带 100ms 延迟的空间慢信号，混在一起会导致相位滞后自激振荡和顺从性坍塌。
2. **绝对不在外部修改速度指令（严守指令纯洁性）**：  
   高层 VLN 发送的直行指令 $c_{\text{pure}} = [v_x, 0, 0]^T$ 必须保持纯洁。外挂 PID 改指令破坏了二自由度控制的分离原理，且在 3kg 偏载下强制让机器人做空载转弯步态，会直接引发法向力倾覆，**跌出动态生存核（Viability Kernel）摔倒**。
3. **绝对不通过物理世界端到端反传微调 Planner 权重**：  
   - 证明了纯 URDF 正向运动学（FK）的梯度在标称直行步态下**恒等于零（$\nabla \mathcal{L} \equiv 0$）**，因为真实物理扰动根本没进入 FK 前向图；
   - 违背 MRAC 公理（Planner 是定义健康的参考模型 $W_m$，不能为了硬件缺陷降低参考模型标准）；
   - 1 维 SLAM 误差对数十万 Planner 权重的信用分配极度病态，会引发正交子空间参数漂移，破坏转弯、停步等通用步态先验。
4. **绝对不搞 OAT 的生搬硬套（伪标签自循环）**：  
   通过手写公式求出 $\omega_z^*$，再用 $P_{\text{ref}}(c+\omega_z^*) - P_{\text{ref}}(c)$ 构造姿态差分，本质是**神经网络自我蒸馏手写启发式规则**，真实物理并未参与反向传播，已彻底否定。

---

## 4. 最终收敛的正确理论架构 (Converged Architecture)

```text
       【上层 VLN / 用户】 (下发纯洁直行指令 c_pure: [0.5, 0, 0], 保持纯洁绝不修改)
                     │
                     ▼
            【冻结的 Planner】 (输出理想标称步态姿态序列 Y_nom)
                     │
                     ▼ (+) ──<── 接触对齐配平变分 ΔY (Kinematic Trimming Offset)
                     ▼
             合成姿态 Y_adapted = Y_nom + ΔY
                     │
                     ▼ 
      ┌──────────────┴─────────────────────────────────────────────┐
      │  【高频快环：纯本体流式 TTT】(50Hz, 已验证成功的第一项成果)   │
      │  - Tracker IDM 实时消除关节力矩缺口，保证 3kg 下平衡不摔倒    │
      └──────────────┬─────────────────────────────────────────────┘
                     │ 输出电机控制动作 a
                     ▼
            【真实物理世界 (3kg 偏载 + 硬地面未踩实)】
                     │
                     ▼
            【机载 3D 激光 SLAM】 (以世界系直线为基准，测出绝对空间误差 e_y, e_psi)
                     │
                     ▼ 驱动外环空间校准
      ┌──────────────┴─────────────────────────────────────────────┐
      │  【低频外环：空间接触自适应校准】(低频 / Stride-to-Stride) │
      │  - 将 SLAM 测得的长程漂移，直接转化为足底脚踝与下肢姿态的   │
      │    运动学配平残差 ΔY                                        │
      │  - 物理功能：强迫刚性脚掌踩平，单步角冲量泄漏归零 (ΔH_z → 0)│
      └────────────────────────────────────────────────────────────┘
```

---

## 5. 核心概念对比表 (Concept Cheat-Sheet)

| 理论要素 | 第一篇论文（OAT - 已完成） | 第二篇论文（Slam TTT - 本课题） |
| :--- | :--- | :--- |
| **解决的问题** | 局部动力学力矩失配（防摔倒） | 长程空间未驱动自由度漂移（防走偏） |
| **物理根源** | 3kg 负重导致质量矩阵 $\mathbf{M}$ 与重力 $\mathbf{g}$ 突变 | 刚性足底微观未踩实导致单步角冲量泄漏 $\Delta H_z \neq 0$ |
| **教材概念** | **系统动力学参数在线辨识 (Inverse Dynamics ID)** | **系统性航位推算发散与运动学接触配平 (Kinematic Trimming)** |
| **干预位置** | **Tracker 动作输出层** | **Planner 姿态输出流形 ($Y_{\text{nom}}$)** |
| **自适应机制** | 动作空间的内模积分律（$\pi_{\text{base}}$ 驱动） | 基于机载 SLAM 空间真值的接触姿态对齐 |
| **稳态目标** | 关节跟踪误差收敛，力矩平衡 | 刚性脚掌 100% 踩平，长程 100 米绝对走直 |

---

## 6. 新会话切入点 (Next Session Starting Point)

进入新会话后，直接从以下具体技术问题开始推演：

> **“在已知 SLAM 测量的空间误差为 $(e_y, e_\psi)$、且干预切入点锁定为姿态层加性配平项 $\Delta Y$ 的前提下：如何利用真机实测数据，在不搞伪标签自循环、不穿透不可导物理黑盒的条件下，最严谨地求解出这个能将单步偏角冲量泄漏彻底归零的接触配平姿态 $\Delta Y$？”**
