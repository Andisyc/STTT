# Current task

Goal: 将官方 FADA 的 G1 Locomotion Oracle→Planner–IDM 源训练接入 TTT，复用 UniLab 的进程/调度基础设施，产物可供现有播放与在线适应使用；接通后测量并优化吞吐。
Do not: 用户已明确仅在本地完成代码，后续由用户亲自上传服务器验证；不连接服务器、不下载/安装、不启动原生仿真、GPU训练或benchmark。不重做已通过的代码/CPU场景，不以旧66/3或诊断模型替代官方产物，不改变研究方法、采样/更新量，不操作Git/机器人。6C/D必须依据用户提供的6B实测热点再规划，不能盲改优化或将计时代码当成提速完成。

Current identity:
- 当前任务状态：第5步5A–5D代码及CPU正式session验证、第6A计时及CPU开关等价验证已完成；本轮双scout复核确认仍缺官方PPO原始run与64/29/7学生/profile。当前Mac不能执行官方Linux/CUDA/IsaacSim源链路，剩余原生验收与6B–D尚未开始；没有GPU实测热点或加速倍率。
- 官方参考：LeCAR-Lab/FADA，commit `290b3a9c5c0ebe2e9851b61350617544be05e9bc`；若实施前版本变化，只按固定版本实现，升级另作明确决定。
- 本地入口：`scripts/train_distill.py:run_fada_training`；配置组 `conf/distill/`；FADA owner 为 `src/unilab/algos/torch/distill/fada/`。
- 历史实验：v24 checkpoint 为 66/29/3、H=30/K=6，仅保留既有实验用途，不当成官方学生权重或默认可转换产物。
- 上一任务已关闭：`expert_trajectory_difference` 在线链路已完成离线工程验证，真实负重效果未验证；保留 README 现有操作说明及已有测试证据，不把旧结果迁移为新基座成功证据。

Confirmed decisions:
- 用户范围：仅接 Locomotion；采用 workflow-governance 轻量记录、system-engineering 总体路线、engineering-plan 逐步细化，Multi Agent 分担独立切片。
- 最新执行边界：用户明确“不连接服务器，在本地完成代码构建，后续我亲自上传至服务器验证代码正确性”。原生验收及性能测量由用户执行；已有本地代码交付不因服务器证据待补而重复实施。
- 整合边界：官方负责网络/数据/学习语义；UniLab 负责唯一训练入口、任务/进程生命周期和权重交接。不引入第二个完整训练主循环或另一套通信协议。
- 源仿真先保持官方环境和 Oracle 条件。优先把官方采集包装成现有 `DaggerCollectorService` worker；不先移植为 MuJoCo，不伪装成已有 UniLab backend。部署端与源训练端分别校准。
- 当前本地 persistent_async 路径是“等采集→训练→发新权重”，已有子进程不等于采学重叠。先保留轮次语义，性能优化不得悄悄引入陈旧策略采集。
- 接口不按维度硬凑：官方 compact observation 的字段/缩放与 7 维命令（含步态相位），不同于现有 g1_fada_state_v2 66 维状态/3 维命令；Oracle 的 actor 特权输入与学生 compact 输入是两套接口，不能混用。
- 学习语义更正（第3步源码复核）：官方输出K步动作，但默认 `idm_action_loss_horizon_gamma=0.0` 使损失只监督首帧；不能把K步输入/输出等同于全K帧均匀监督。Planner未来状态监督系数默认0，动作误差经冻结IDM反传。先前“官方默认监督全K帧”的记录有误，以固定源码config.py及weighted_horizon_mse为准；不擅自更改horizon权重。
- 现有在线研究方法保持不变：冻结 Planner 与独立 base_idm；用相同历史计算 base(X,Y_nom)-base(X,Y_exec)，叠加保存的 U_snapshot 形成静态目标；执行 IDM 只更新 LoRA，compute_loss 仍以 Y_nom 为未来条件。保留原 reset/批次/锚定/Adam 语义，不借接入改方法或调参数。
- 新基座以明确模型/观测/schema 身份保存，不覆盖旧 checkpoint、旧 run 或已有数据；不兼容即明确拒绝，不截断/补零/静默 strict=False。保留旧实验播放能力不等于维护两套正式源训练主循环。

Progress:
- [x] 完成总体路线与并行边界规划。
  - Evidence: 直接读取本地 workflow_setup、observation、入口和 runtime 接口；两个只读 scout 分别检查官方/本地采集与 learner；直接核对固定版本 compact_obs。未执行任何运行验证。
  - Multi Agent：A=采集/Oracle，B=模型/学习；主 Agent 负责接口、Runner、公共配置/schema 和本板。独立切片可并行，共享文件由主 Agent 单写。子 Agent 不跑测试/格式化/训练，主 Agent 合并后做每步一次针对性证明。

## 总体路线与逐步 engineering-plan

后续每步开始时，只细化当前包：补齐实际符号/调用者/允许文件和唯一检查；不重新全仓探索，不提前为后续步骤写详细伪代码。第 1 步的接口结论冻结后，第 2/3 步可并行，其余按依赖推进。

- [ ] 1. 冻结官方输入、输出与数据来源（已执行，部分完成；待绑定官方PPO产物）。
  - Owner / scope：两个只读scout分别核准输入控制端和时序数据端，主Agent汇总并唯一执行离线对照；未修改生产代码/配置、旧checkpoint或在线差分方法。
  - 已完成源码校准：固定官方commit `290b3a9c5c0ebe2e9851b61350617544be05e9bc`；不把本地SAC/APPO或普通98维actor混同于官方Oracle。下表为源码基准，不代表未取得的实际run overrides。

  | 对象：生产者→消费者 | 已核准语义 / 来源 | 本地接入位置与状态 |
  |---|---|---|
  | Student compact：官方obs terms→Planner/IDM | 64维按 `base_ang_vel(3)*0.25, (q-q_default)(29), dof_vel(29)*0.05, projected_gravity(3)`；角速度/重力在base系，世界重力[0,0,-1]。`compact_obs.py`＋`managers/observation/terms/locomotion.py`。纠正早期判断：官方关节观测也是相对角，不是绝对角。缩放实际应从选定Oracle配置解析，不能只硬编码默认值。 | `observation.py`/官方采集adapter，不能直接复用历史66维投影；语义与临时映射已离线对照。 |
  | Task command：command manager→Planner | 7维=`vx,vy,wz,sin(left/right phase),cos(left/right phase)`；速度3＋相位4，零速度不等于相位4维清零。`common/current_command.py`。 | 原3维command接口需明确扩展；尚未修改生产实现。 |
  | Oracle actor：专用observation manager→PPO | 有序11字段：base_ang_vel、base_lin_vel、dof_pos、dof_vel、actions、projected_gravity、oracle_actuator/contact/terrain/command_gait/randomization_compact_state。`g1_29dof_loco_oracle`，不是普通locomotion98维preset。源码PPO empirical_normalization=False、FADA normalize_io=False；实际权重身份/normalizer仍须随原始run核对。 | `oracle.py`不能走现有SAC专用loader；真实官方PPO checkpoint与配套配置尚未绑定。 |
  | Actions：policy→环境；历史/标签→Replay | 源码默认P控制、clip ±100，q_target=q_default+0.25*executed_action；29关节顺序为左右腿各6、腰yaw/roll/pitch、左右臂各7。初态与各关节PD使用固定官方`config_values/robot.py:g1_29dof`完整精度配置，不手抄近似值；ctrl_dt默认0.02s（200Hz/decimation4）。实际run覆盖值未绑定。`executed_act`含探索噪声后clip的真实下发动作，expert_act/chunk为clip后的Oracle动作；不是力矩。 | `collection_contract.py`及worker适配；源码边界已明确，未验证物理执行。 |
  | Histories：Replay→Planner/IDM | 官方默认H=30/K=6（`planner_idm/config.py`，与历史本地值碰巧相同但来源独立）；hist_obs止于t，hist_act对应各观测前一步、最后止t-1；冷启动左补零＋valid_mask，不是重复初态。 | `windows.py`/`collection_windows.py`及model/checkpoint identity需要承载mask；原始Replay实测通过。 |
  | Future/action targets：shadow或真实trajectory→loss | 真实动作t..t+K-1配未来观测t+1..t+K；strict expert chunks与trajectory targets字段分离，真实未来不足时trajectory_target_valid=False，不得把零填充当标签。缓存按(env_id,episode_id)隔离。严格标签有效性须沿用producer的strict_label_valid/branch-done结果；无官方role0/1枚举可直接照搬。 | `source_artifact.py`/replay reader/采集adapter；原始Replay的组窗、mask、invalid-tail和reset隔离实测通过；原生shadow状态恢复尚未执行。 |
  | Artifact identity：真实run→loader/worker | 必须绑定官方source commit、已解析配置、actor/compact/command/action schema、normalizer、H/K/mask、source角色与权重版本；不兼容即拒绝，不能strict=False或补零截断。 | 第4步已切至官方raw source v8及独立学生checkpoint v2；真实Oracle产物绑定仍待补齐。 |

  - 唯一离线对照：`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt_fada_step1_calibration_290b3a9.py`，一次执行PASS。临时脚本直接执行固定官方的观测/指令函数原始函数体（隔离仿真依赖，不替换函数体）和完整原始 `ReplayBuffer.add_batch/sample_chunk`；对照独立命名状态映射。没有导入/启动IsaacSim，没有用自写reference替换上游实现。
  - Evidence：不同关节非零值＋非零默认角＋非单位姿态下，float32观测最大绝对误差`5.960464477539063e-08`，command误差0（atol/rtol各1e-6）；同次对照检查8个正常/冷启动/reset锚点，obs/action历史止点、左补零mask、future/action首索引、shadow/真实目标分离和不完整轨迹invalid全部通过。H4/K2只是诊断fixture，未冒充官方默认。临时脚本已删除，未新增永久测试或流程产物。
  - 实际产物只读核实：`FADA_Unofficial/model/oracle_official_appo_seed1/appo_final_update_5000.pt` 经 `torch.load(map_location="meta",weights_only=True)` 读取metadata；schema=`fada_official_appo_oracle_v1`、algorithm=`fada_official_appo`、backend=`mujoco`、actor/critic322、student66/action29/command3、obs_normalization=False。虽source_commit同上游，但这是本地APPO迁移分支，不是官方发布PPO/64+7学生链路；不擅自替换本轮目标。相关owner在相邻FADA_Unofficial仓库，TTT未发现该official owner，未移植或修改。
  - 证据范围：未构建/运行官方Oracle、未验证物理动力学/原生shadow、未证明正式adapter接通或策略质量；离线Replay使用人工标明的episode/strict有效性，因此不把它当成真实终止producer验证。原始官方checkpoint/config尚未本地读取，不能把源码默认当作实际run值。
  - Done when：源码表与离线对照已完成；真实官方PPO checkpoint及配套配置/normalizer身份核准后才能整步勾选。第2/3步可依据已核准契约继续规划，但不得悄悄以本地APPO产物代替官方源仿真路线。

- [ ] 2. 接入官方源采集与 Oracle（代码接入已完成；2D原生采集未验证）。
  - Change / Why：在既有UniLab采集服务内包装官方环境、PPO Oracle和影子标注，保留官方源条件。不是把官方PPO权重塞进本地SAC/APPO loader，不迁到MuJoCo，不调用官方完整训练主循环。
  - 主Agent先冻结最小共享接口：资源包提供初始化后的官方env、Oracle推理、strict-shadow服务及close；采集包接受这些资源和真实student推理接口，产出带明确schema/来源身份的采集artifact。资源/采集两包不直接修改彼此或公共文件；字段名称、形状、mask及错误行为在派工前由主Agent给定。
  - [x] **2A 资源与启动（OfficialResources Agent实现，主Agent集成）**：已新增 `fada/official_source_runtime.py`，真实调用固定上游FADAConfig、build_env_and_expert、StrictLabelWorkerClient、build_student_policy和FADATrainer采集helper；optimizer/lr_scheduler均为None，不调用warmup/run_loop。缺原始配置即失败，官方依赖仅worker内导入；共享权重键/形状在复制前严格匹配。主Agent修正预处理helper属于上游train.py而非trainer.py、禁止覆盖产物、允许无关日志/docs而拒绝上游源模块变更；close释放模型引用并处理部分初始化。
  - [x] **2B 采集事务（OfficialCollection Agent实现）**：已新增 `fada/official_source_collection.py`，三种source调用真实上游_collect_with_policy，事务使用独立组窗buffer；online含配置噪声、optimal由最终Oracle驱动、suboptimal由弱权重驱动且strict worker保持最终Oracle，finally恢复最终expert。主Agent要求移除额外64轮补采/禁止replacement策略，现每事务只采一次配置rollout，随后沿用原生sample_chunk(num_windows)。不造数据、不创建optimizer/App/进程，保留历史mask与真实trajectory有效性。
  - [x] **2C 公共接线（主Agent）**：`collection_contract.py`新增OfficialFADACollectionSpec/OfficialFADASourceBatch；同一`source_artifact.py:FADAShardedSourceWriter`支持显式官方schema与独立reader（第3步由v6升至含真实预处理的v7），沿用原子提交、分片身份校验，旧v5 reader/行为保持；`fada.async_runtime.py:build_official_fada_runtime`复用PersistentDistillationRuntime、共享权重与Dagger请求/结果，不引入新Runner或IPC。正式训练CLI未切换（第4步）。
  - 公共payload：保存官方 `history_obs[B,H,64]`、`history_act[B,H,29]`、`history_valid_mask[B,H]`、`current_command[B,7]`、`target_actions[B,K,29]`、`target_next_observations[B,K,64]`、`trajectory_target_actions[B,K,29]`、`trajectory_target_next_observations[B,K,64]`、`trajectory_target_valid[B]`，按source声明哪些目标可用于哪些loss；头部绑定request/iteration、source角色、官方revision、Oracle/config/观测/动作身份与observed_weight_version。来源追踪必须来自真实采集锚点，不从浮点状态反推。writer/reader对新schema显式校验，旧数据不被误认新数据。
  - 有效性规则：strict-shadow标签沿用官方strict有效性，真实轨迹未来不足时仅标trajectory_target_valid=False；不得因此误删仍合法的shadow样本，也不得把补零轨迹送入真实动力学损失。Reader在相应source/用途上筛选，禁止用未定义的role0/1猜映射。
  - 已实现进程接缝：`AsyncRunner`新增可选collector_daemon（默认True不变），经PersistentDaggerCollectorRunner/PersistentDistillationRuntime传递；官方factory设False。POSIX worker独立session，正常关闭先走服务close，超时/异常由父Runner清理整组子进程，request-scoped退休路径也复用清理。修改前已用LSP核对共享接口调用者；未创建第二个Runner。
  - 生命周期/错误：父进程不启动官方仿真；worker独占资源；正常stop与异常均先通知/等待shadow退出，再关闭其IPC及env/App。超时/中断沿现有Runner guard传播并回收资源，不能只把collector改为non-daemon就忽略孙进程。失败抛异常，由已有guard传回父进程，不伪造success=False或半成品DaggerCollectResult；仅在完整artifact落盘并校验后返回成功结果。
  - Preserve / Forbidden：不改官方奖励/随机化/控制参数、模型loss/网络/在线目标或默认超参数，不运行完整DAgger/warmup训练，不覆盖旧run/artifact；工厂显式选择官方source身份，不静默改变旧配置含义。旧源训练的最终切换/移除在第4步处理，不承诺永久保留两条正式训练链。
  - [ ] **2D 唯一官方原生采集验证**：尚未执行。需要绑定官方PPO产物、官方运行环境和真实学生模型后，经正式Dagger请求做一次有界采集；核对reader、实际shadow前后状态/RNG、真实reset、动作/future语义、权重版本与原生进程释放。离线进程/存储smoke不替代此项，也不证明学习效果/加速。
  - Done when：2A/2B接线、显式schema与生命周期全部实现，且2D有实际证据；之前只标子包完成，不标第2步整体完成。第1步离线sample_chunk结果不替代此处真实采集。该验证不评价长期策略效果或稳定提速。
  - Depends on：第1步剩余的官方PPO checkpoint/config身份核准；可用的官方运行环境；第3步真实student模型/权重加载接口。2A初始化及2B源数据处理可按已确认契约独立开发，但完整学生闭环与2D不能在第3步前用假student验收。下载/安装、目标机器访问与真实采集须有各自授权，本轮均不执行。
  - 集成Evidence：主Agent执行 `PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt_official_source_step2_smoke.py`。首次在产物commit前因测试临时目录未resolve真实路径而触发身份拒收；仅修正fixture为Path(directory).resolve()后重跑失败及尚未完成的场景，PASS。九字段往返完全一致、无效真实尾段仍保留合法shadow；坏mask/错误source/版本/格式混用/分片篡改均被拒，旧v5读写保持；真实SharedWeightSync发布版本1→2，嵌套spawn计算值31→62，collector/子进程关闭后均退出；真实官方factory在缺checkout时通过错误管道明确失败，未导入native或生成成功artifact。最后一条预期缺依赖错误有进程trace，不是隐藏的成功fallback。
  - 验证范围：未跑测试套件/formatter/独立审查。子Agent曾做一次局部导入/AST检查；主Agent集成smoke之外，仅因发现close仍持有模型引用而补做一个针对性弱引用检查，确认正常/部分初始化close幂等并释放实际torch模型对象；未重跑已通过集成场景。临时smoke脚本已清理；README记录API/前置条件和证据边界，无新流程文档。

- [x] 3. 接入官方网络与学习规则（代码及唯一CPU数值对照已完成）。
  - **3A / OfficialStudentModel Agent**：新增 `official_model.py`，移植固定上游纯Torch Planner/IDM和位置编码；`build_official_fada_student`保留原生键/形状、pre-Norm、delta/mask/历史命令flag，完整K步输出。默认64/29/7、H30/K6，不经safe_torch_import或native模块。
  - **3B / OfficialStudentLearner Agent**：新增 `official_learning.py:OfficialFADALearner`，保留默认增强/掩码、horizon权重、两个AdamW与每批IDM→Planner；Planner反传时IDM参数固定，之后恢复。真实分支先过滤无效行，但source加权仍用原始行数。主Agent去掉重复序列化的派生来源开关，绑定模型共享配置，并接通计数恢复。
  - **3C / 主Agent**：`replay.py:OfficialFADAReplayBuffer`与`replay_sampling.py:official_replay_source_counts`接收九字段窗口，保留三种物理来源、五种IDM用途、offline/mixed配额、Python取整、Planner先于IDM采样与原始replacement规则。完整原始轨迹保留/淘汰分布仍在第4步处理，不以minibatch一致替代整轮一致。
  - **预处理身份 / 主Agent**：worker metadata导出Oracle run实际解析的compact scale/noise；`source_artifact.py`官方schema升至v7，缺值的旧v6产物拒收，旧v5不变。learner必须显式绑定预处理，不从hash反推或套旧66/3默认值。
  - **3D / 主Agent**：`checkpoint.py`增加官方学生保存、严格学习恢复与`load_official_fada_student_checkpoint`纯推理loader；完整模型/训练/预处理、原生state_dict、两optimizer与更新计数持久化。独立格式拒收旧学生/PPO/APPO，不加权重hash或另一manifest；loader可供既有runtime使用。
  - **3E Evidence**：主Agent统一执行一次CPU原代码对照场景（`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-official-step3-1793209402180/verify.py`）。首次仅因fixture把源码中待推断的obs_dim=None当作64默认值而停止，修正该断言后场景PASS：真实上游ReplayBuffer生成窗口与采样顺序/配额相同；两组位置编码/delta/历史命令flag前向相同；默认噪声、掩码、dropout及实际触发裁剪下的weighted loss、IDM→Planner权重与AdamW状态逐项相同；gamma0后续动作标签不改变梯度；strict恢复两optimizer/计数及SharedWeightSync通过。未启动任何仿真或完整训练。临时reference与脚本在交付前清理。
  - **Preserve / 边界**：历史66/3模型/旧checkpoint/在线LoRA方法未修改；正式CLI、完整warmup/DAgger循环、播放适配与提速仍分别属于第4–6步。子Agent仅写各自模块并做局部静态检查；未跑测试套件、formatter或独立审查。第1/2步真实产物和原生采集缺口不随本步关闭。

- [ ] 4. 接通唯一正式训练入口（代码与离线验证完成；待原生验收）。
  - [x] 共享接口：同一Runner队列增加显式prepare；初始化学生仅注册共享布局，实际Oracle metadata返回后才建立learner、保存并发布真实checkpoint。阶段请求显式区分optimal/suboptimal/online，采集输出13字段原始transition（source v8）；旧v5及旧学生播放不变，官方窗口v7不冒充raw。
  - [x] 采集owner：`official_source_runtime.py` / `official_source_collection.py`复用同一native env/Trainer/fixed Oracle shadow，取消worker提前抽窗口；optimal按horizon分次采集，弱策略轮换至官方目标预算。实际native路径尚未运行。
  - [x] Replay/外循环owner：`replay.py` / `persistent_workflow.py`复用原始NumPy ReplayBuffer，跨shard完整episode判断、轨迹抽取/划分、跨轮online累积、整episode淘汰与动态sample_chunk；完整offline缓存、train/val保存恢复、两optimizer/采集计数配套。第一轮online前发布warmup权重，每轮发布不受周期快照间隔影响；恢复不重跑warmup。
  - [x] 配置入口owner：`scripts/train_distill.py`、`workflow.py`、`workflow_setup.py`、官方runtime factory、checkpoint与`conf/distill/official_fada.yaml`接通；唯一正式配置为`--config-name official_fada`，去掉旧SAC/66/3源训练装配及废弃legacy外循环。原始学习默认值与固定官方FADAConfig对照一致，不增IO normalization；旧源入口明确拒绝，非FADA蒸馏及在线LoRA语义不变。
  - [x] 近端离线集成：真实入口及真实persistent Runner/共享内存，使用明确标识的诊断轨迹完成prepare→offline/warmup→两轮DAgger→发布→第三轮恢复。核对原始13字段多shard往返、完整episode保留、原版轨迹split/random淘汰、动态采样、NPZ恢复、warmup后首轮权重及每轮策略输出改变；save_every_iter=2仍逐轮发布，恢复不重复warmup。另验证缺checkout由已有错误通道回传且不生成假checkpoint。
    - Evidence: `PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-official-step4-1793212935353/verify.py` 的Replay与完整workflow段通过；只继续未完成的`entry failure`段后全部通过。初次诊断脚本的文件名碰撞及错误入口开关已修正；未因此改动生产逻辑。临时诊断脚本/参考数据清除，不新增永久测试或报告。
  - [ ] 最终Done when：绑定真实官方PPO、弱checkpoint历史及保存配置，在获授权的Linux/CUDA/IsaacSim环境由正式入口完成有界offline/warmup和两轮DAgger，核对发布版本、实际raw累积、严格恢复及worker/shadow回收。离线证据不替代该原生验收。
  - Multi Agent：主Agent先冻结共享接口，再并行A采集端、B Replay/外循环；主Agent完成入口/持久化集成与统一离线验证，无独立审查步骤。未进行采学重叠、提速、物理播放或策略质量实验。

- [ ] 5. 接回播放与现有在线适应（5A–5D代码与CPU验证完成；真实原生播放验收待前置）。
  - [x] 5A 共享接口：新增纯Torch runtime spec/原生调用接缝；明确format先于数字schema分流，官方strict loader与旧/适配模型分支分开；snapshot携带64/66契约与bool mask。
  - [x] 5B 官方播放：新增官方64/29/7 controller；物理状态投影、完整相位command、source clip和首动作快照、左补零/mask及partial reset已接通，旧controller不变。
  - [x] 5C 在线接口：collector/updater按runtime spec接收command/mask；冻结base双路差分与当前IDM回归使用同一原生接口。相位变化不再清空K步待成熟轨迹。保留LoRA层发现、固定标签、loss、窗口、Adam/锚定及reset保留已学参数规则。
  - [x] 5D 正式session：factory按明确模型身份选择controller；显式source-bound profile在创建env前核准，公共物理接口负责观测与source→target动作映射，检查PD/dt/相位契约。缺失profile不回退旧task；无新增在线checkpoint格式、env/backend业务逻辑或RoboJuDo_Real改动。
  - Evidence：主Agent统一CPU场景3项PASS（本轮终端）：非单位缩放/非零默认角/关节重排/相位、原生前向与不可变快照；正式session完整K步成熟、实际LoRA-only更新、Planner/base不变及reset保留Adam；旧schema2播放/更新和官方format分流、缺失/错配profile在创建env前拒绝。首次场景发现partial reset泄漏上个episode动作，修复后同场景通过。诊断transition不是物理仿真；临时脚本/模型已清理。
  - Done when / Depends on：代码实施无需重做；完整验收仍需第1步真实官方权重/run配置绑定、第4步相关原生前置和运行授权，再通过正式session关闭/开启更新的真实短运行。当前不宣称sim2sim、负重效果或加速。

- [ ] 6. 测量并实现有依据的训练提速（6A完成；6B–6D待运行前置及实测）。
  - **目标 / Preserve**：在相同硬件、官方checkpoint/初始学生、配置、env/rollout/batch/更新量、source比例、strict-label、采样顺序/seed、cache状态与权重版本条件下衡量端到端吞吐。区分官方原生→TTT的整合收益与TTT改前→改后的优化收益；不把不同工作量或缓存命中当算法加速。
  - [x] **6A 测量接线**：新增轻量`fada/timing.py`复用已有StageObservationAccumulator；`workflow.py`与`conf/distill/official_fada.yaml`提供默认关闭的`training.fada.performance.enabled`，不进入原生算法配置/缓存/checkpoint。`persistent_workflow.py`汇总worker耗时，计量初始化、RPC、artifact/Replay、batch、学习、保存/发布和close；`replay.py`记录原始sample与H2D。结果在warmup/每轮/close后由现有日志输出，最终返回performance，无新进度/报告/哈希体系。
  - [x] **6A Multi Agent**：主Agent先落地共享计时接口，再并行A采集端（`fada/async_runtime.py`、`official_source_runtime.py`、`official_source_collection.py`）和B学习端（`official_learning.py`）。A计量资源初始化、已完成的rollout/CPU打包，并保留原metrics；B计量IDM→Planner前向/反向/optimizer。主Agent集成Replay、配置、正式外循环与统一验证；未设review席位，子Agent仅做局部编译/AST检查。
  - [x] **6A唯一离线Evidence**：`PYTHONPATH=src OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --no-sync python /tmp/ttt-fada-timing-1793242829867/verify.py`一次运行PASS。计时OFF/ON经过正式owner/factory、真实resident worker prepare/collect、采集事务/打包、raw v8、固定官方原始NumPy Replay、真实Torch learner、checkpoint及SharedWeightSync；仅native资源bootstrap替换为明确CPU诊断数据生产者，不冒充PPO/物理仿真。warmup+两轮DAgger共3次更新、128条online transition，模型/两AdamW/loss/RNG/持久化轨迹/发布版本逐项相同，collector平面metrics、实际计数与close后汇总通过。临时源码/诊断文件清除，不新增永久测试或报告。
  - **6A口径**：默认关闭时不创建timer/CUDA event、不新增tensor或取CPU标量；开启时CUDA event只在报告边界统一解析，原生worker只报已完成事务wall time。parent RPC/worker/rollout属于包络与子阶段，不重复相加；未测阶段不填零。累计输出区分raw transition、sample window与update；total_elapsed覆盖训练owner初始化至close，不含此前CLI/Python导入。原生GPU计时尚未验证，CPU诊断耗时不当成吞吐基线。
  - **6B 有界基线测量（主Agent；依赖6A、4+5验收前置及运行授权）**：使用固定版本官方原生入口作为reference，和当前TTT正式入口串行运行同一已核准工作量；不改官方训练主循环、不在TTT建立第二条正式训练路线。执行前固定具体设备、预算、初始模型/optimizer、env数、horizon、warmup/DAgger更新量、cache与日志开关及独立输出目录；分别记录冷启动、稳态、实际transition/update/source配额、版本与阶段占比。该TTT结果作为后续候选的改前基线，不重跑已取得基线。Done when：能够指出哪个阶段实际主导端到端时间及收益上限；没有官方原生对照时不能声称相比官方加速。
  - **6C 条件式Multi Agent优化（依赖6B，测到后才冻结具体工程小包）**：只对有实测占比的独立热点派工。候选A拥有`official_source_collection.py`/`official_source_runtime.py`/`source_artifact.py`的采集打包或I/O冗余；候选B拥有`replay.py`/`official_learning.py`的staging/复制或学习同步冗余。公共schema、正式入口、checkpoint/Runner发布及共享buffer均由主Agent单写；若热点只在一侧就只派一位，若热点耦合就合并，不为填团队而派工。A/B开始前必须冻结实测问题、精确符号、输入/输出、独占文件和等价性判据；两者不依赖对方未来产物且主Agent有集成工作时才同批派出。子Agent不跑GPU性能任务、测试套件或review，不再派助手。本轮不预选compile/AMP/关闭shadow/缩减样本或更新、跨轮prefetch/采学重叠；后两者涉及版本/RNG语义时单独决定。
  - **6D 集成与对照（主Agent；依赖已接纳6C）**：按改动边界统一做一个近端等价场景，核对原始trajectory/样本顺序、mask/source配额、loss与梯度/optimizer更新、checkpoint恢复及下一轮发布版本，不默认重跑全套。再在同一GPU串行执行一次候选固定工作量，与6B既有基线比较端到端时间、phase占比及吞吐；不能只报局部kernel改善。GPU比较由主Agent独占资源，模型协议/RNG和数值容差在选定优化包时明确，不默许精度下降。
  - **Done when / Stop**：有固定工作量、身份一致的实际对照和等价性证据，只有端到端实测改善才保留优化并报告倍率；无改善撤回候选，记录未取得加速，不降低工作量或自动追加训练。短时吞吐不替代策略质量；6B仍依赖原第6步的4+5前置及单独运行授权。6A完成不等于热点优化/性能对照完成，未预选compile、AMP或采学重叠。

Blockers:
- 真实产物：本轮双scout复核TTT与FADA_Unofficial的model/logs及候选权重metadata，未找到官方PPO原始run（权重、保存配置、弱策略历史），也未找到兼容64/29/7学生及source-bound控制profile。现有候选为MuJoCo SAC/APPO或g1_fada_state_v2学生66/29/3，不替代本轮目标。正式官方学生格式以`checkpoint.py`实际常量`unilab_official_fada_student`为准；主Agent已纠正scout报告中的错误格式名。
- 运行环境：实际metadata为Darwin/arm64、Torch 2.7.0、CUDA不可用、isaacsim/omni未安装；`official_fada.yaml`的upstream_root/expert_checkpoint仍未绑定。第2D/4/6B由用户在服务器验证；第5步真实MuJoCo验收还需兼容学生/profile。用户已明确禁止本轮连接服务器，因此不再请求SSH授权或探查远端。6C/D等待用户的实际阶段计时与等价性验收结果。

Next:
- 用户上传当前实现，绑定真实官方run并执行原生验收与6B计时；收到具体错误或性能热点后，主Agent再按engineering-plan冻结修复/6C优化包，并以Multi Agent实施。当前无待执行的已规划本地模块，不重复CPU验证，不宣称原生验收或提速已完成。
