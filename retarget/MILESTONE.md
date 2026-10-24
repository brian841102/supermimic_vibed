# nemo_v3 花式滑冰动作重定向：计划书与开发记录（MILESTONE）

> 本文件是**计划书 + 开发过程中遇到的坑 + 因此做的调整**。安装依赖和常用指令见 [README.md](README.md)。

- 日期：2026-10-24（v2：已读取 nemo_v3 模型和 booster_mjlab 的现有做法，第 2 节的数据都是实测）
- 目标机器人：nemo_v3（轮足 humanoid，**两条腿末端各一个驱动轮，没有踝关节**），全身版 `nemo_v3_2_fullbody`
- 模型来源（已拷到 `retarget/assets/nemo_v3/`，随仓库提交）：
  - `booster_mjlab_nemov3/xmls/nemov3.xml`：booster_mjlab 训练用的 MJCF，**以它为准**（关节顺序、限位、actuator）
  - `nemo_v3_2_fullbody/urdf/train_v2.urdf` 等 6 个 URDF 变体（见 2.3 节）

---

## 0. 结论先行

| 问题 | 建议 |
|---|---|
| 从哪个仓库出发？ | **GVHMR（视频 → SMPL-X）+ GMR（SMPL-X → 机器人）**。最关键的理由是：booster_mjlab 的 `MotionFile`（`src/booster_mjlab/motion/motion_data.py`）**本来就是 GMR 的 pkl 格式**，GMR 的输出只要把关节列补齐就能直接用（见 2.2 节）。另外 GMR 用 CPU 就能实时跑 IK，加机器人只要写一份 JSON |
| 其他方案怎么用？ | 基线跑通后做对照。**UMR 已经是 booster_mjlab 的 submodule**（`submodules/UMR`，有 K1 / G1 / H2 / T800 的配置，但还没有 nemov3），对照成本最低，排第一。其次用 **GEM-X** 对比 GVHMR 的人体估计，用 **soma-retargeter** 对比重定向。最后按下游训练效果来选 |
| 轮足最关键的差别是什么？ | ① 人的"滑行"在机器人上就是**轮子滚动**，所以不能做"脚贴地不动"的防滑后处理；② **没有踝关节**：轮子朝向只能靠 hip yaw，外倾只能靠 hip roll；③ 轮关节不参与 IK，事后按 `ω = v/r`（r = 0.075 m）算出来，训练时用速度控制 |
| 当前硬件的限制 | 头 3、腰 3、腰部推杆 4、腕 pitch 2，共 **12 个关节焊死**（`NEMOV3_FIXED_HEAD_WAIST_JOINTS`），实际可动 **20 个**：每只手臂 5 个（肩 P/R/Y、肘、腕 roll），每条腿 5 个（髋 P/R/Y、calf、轮）。**躯干不能相对骨盆转动**，身体前倾只能靠髋 pitch。<br>**重定向做两个版本**（5.1 节）：`full` 版放开腰和腕 pitch（只焊头、推杆、轮子），给 tracking 和将来的硬件用；`hw` 版和当前硬件一样焊死，给 AMP 用 |
| 动作从哪来？ | 首选**自己拍**：固定机位、高帧率。**花样轮滑（artistic roller skating）**比冰上花滑更接近轮子的物理，建议两种都拍 |
| 先做哪些动作？ | 先做双轮着地的滑行加上肢编排、弓步滑行、双轮慢转、蛇形滑。**压步（crossovers）降级**：髋只能内收 10°，左右轮距只有 17–25 cm，两腿交叉基本做不到。详见第 8 节 |

---

## 1. 目标与范围

**目标**：从视频中得到一批 nemo_v3 能执行的花式滑冰参考动作（kinematic reference），存成 booster_mjlab 的 `MotionFile` pkl。

**下游用法**：
- 现有的 nemov3 训练是 **AMP 速度任务**（`tasks/velocity/config/nemov3/`），只用了 3 段直线蹬冰片段：skating01、skating02、skatingfast06；
- 新做的花滑动作可以扩充这个 AMP 数据集；
- 也可以给 tracking 任务用（booster_mjlab 有 `tasks/tracking/`，目前没有 nemov3 的配置）。

**本计划覆盖**：
- 视频 → 人体动作
- 人体动作 → nemo_v3 关节轨迹
- 轮子后处理
- 质量检查
- 导出 `MotionFile`

**本计划只给接口和注意事项**（不展开）：训练的奖励设计和 sim-to-real，见第 6 节 Phase 6。

---

## 2. 已知条件（实测）

### 2.1 nemo_v3 模型信息（来自 `nemov3.xml`、`train_v2.urdf`、`nemov3_constants.py`）

| 项目 | 值 |
|---|---|
| 总质量 | 14.58 kg |
| 关节 | 32 个（`NEMOV3_JOINT_ORDER`，= MJCF 声明顺序 = `dof_pos` 的列顺序）+ 根 free joint |
| 根 body | `link_leg2pelvis`（IMU site 在 `leg_base_link` 上） |
| 腰（3） | `joint_pelvisYaw` ±1.396、`joint_pelvisPitch` [0, 0.524]、`joint_pelvisRoll` ±0.175 —— **当前硬件焊死** |
| 腰部推杆（4） | `joint_[rl]LinearOut`（hinge）+ `joint_[rl]Linear`（slide ±0.01 m）—— 并联机构，**焊死** |
| 头（3） | `head_roll/pitch/yaw_joint` —— **焊死** |
| 手臂（每侧 6） | ShoulderPitch [−3.14, 1.57]、ShoulderRoll [−1.57, 1.91]、ShoulderYaw ±2.79、Elbow ±1.57、WristRoll ±2.79、WristPitch ±1.57（**焊死**） |
| 腿（每侧 5） | `*_hip_pitch_joint` ±1.571 → `*_hip_roll_joint` [−0.174, 1.047] → `*_hip_yaw_joint` ±2.356 → `*_calf_joint` [0, 1.571] → `*_wheel_joint`（continuous） |
| 腿的结构 | **没有踝关节**：轮子直接装在 calf 下面。轮子朝向（yaw）由 hip yaw 决定，外倾（camber）由 hip roll 决定 |
| hip roll 的正方向 | 左腿轴 +x、右腿轴 −x，所以两侧**正值都是外展**：最多外展 60°、内收 10° |
| 轮子 | 半径 **0.075 m**（mesh，宽约 6 cm）；`<velocity>` actuator，kv 0.5，力矩 ±6 Nm；URDF 速度上限 25 rad/s（≈ 1.9 m/s） |
| 其他 actuator | 都是 `<position>`，XML 里 kp 50 / kd 0.5；腿的 forcerange ±18（calf ±20） |
| 硬件实际 PD（`NEMOV3_DEPLOY_PD_GAINS`） | 肩 pitch/roll kp 18；**肩 yaw、肘、腕 roll 只有 kp 1.5**；URDF 力矩上限：肩 pitch/roll 5.6 Nm，其余臂关节 1.27 Nm |
| 关节速度上限（URDF） | 腿 10 rad/s，肩 pitch/roll 23.9 rad/s，其余臂关节 40.2 rad/s |
| 训练的速度范围 | 0–1.3 m/s（`NEMOV3_WHEEL_ACTION_SCALE = 20` rad/s） |

**几何尺寸**（零位姿态、根在原点，FK 实测）：

| 尺寸 | 值 |
|---|---|
| 腿伸直时根离地高度 | **0.51 m**（轮心在根下方 0.435 m，再减轮半径） |
| HOME 蹲姿 | 根高 0.50 m（`HOME_KEYFRAME`：hip pitch −0.26、calf 0.43、hip roll 0.08） |
| hip pitch 关节 | 在根下方 0.097 m |
| 大腿（hip pitch 关节 → calf 关节） | ≈ 0.19 m |
| 小腿（calf 关节 → 轮心） | ≈ 0.18 m |
| 髋 → 轮心（腿伸直） | ≈ 0.34 m |
| 左右轮距 | 零位 0.172 m；HOME 约 0.25 m |
| 肩关节 | 在根上方约 0.10 m（站姿离地约 0.62 m） |
| 上臂 / 前臂 | 0.103 m / 0.092 m |
| 零位的手臂姿态 | **水平侧平举，就是 T-pose**，可以直接当 GMR / UMR 的 T-pose |

**两个容易踩的坑**：
- MJCF 里的 `left/right_wheel_ground` site 挂在 **wheel_link** 上，局部坐标 (0, 0, −0.075)，**会跟着轮子转**。只有轮角为 0 时它才在轮子最下面。所以不能直接拿它当接地点，要用"轮心 − r"来算。
- 旧的滑冰 NPZ 数据里，两个肩 roll 关节的符号和 MJCF 相反（`NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS`）。GMR 直接读 MJCF，**不需要**做这个翻转；但如果混用旧数据，要注意。

### 2.2 booster_mjlab 的 motion 格式：`MotionFile` pkl

`src/booster_mjlab/motion/motion_data.py` 里的 docstring 写着："follows the same structure as the motion files in GMR"。字段如下：

| 字段 | 形状 / 类型 | 说明 |
|---|---|---|
| `fps` | float | 帧率 |
| `root_pos` | (N, 3) | 根 body `link_leg2pelvis` 的世界坐标位置 |
| `root_rot` | 长度 N 的 list，每项是 **xyzw** 四元数 | GMR 的输出就是 xyzw；`nemov3_skating_to_pkl.py` 也是转成 xyzw |
| `dof_pos` | (N, **32**) | **按 `NEMOV3_JOINT_ORDER` 排列，含两个轮关节**；焊死的关节填 HOME 值 |
| `local_body_pos` | (N, B, 3)，可选 | 各 body 在根坐标系下的位置；目前只在左右镜像增强里用到 |
| `link_body_list` | 可选 | 与 `local_body_pos` 对应的 body 名称 |

可以参考 `src/booster_mjlab/scripts/nemov3_skating_to_pkl.py`，它就是把旧数据转成这个格式的现成例子。生成后用 `src/booster_mjlab/scripts/visualize_motions.py` 检查。

### 2.3 已有的资产和工作（不要重复造轮子）

| 位置 | 内容 | 对本计划的用处 |
|---|---|---|
| `nemo_v3_2_fullbody/urdf/train_v1.urdf`、`train_v2.urdf` | 32 个可动关节。**FK 实测和 `nemov3.xml` 完全一致**（误差 1e-16），轴向、限位也一致；只是 URDF 的关节声明顺序不同，**以 MJCF 的顺序为准** | 等价于训练模型 |
| `urdf/train_v2_payload_0p8.urdf` | 带 0.8 kg 负载的版本 | 不用 |
| `urdf/robot.urdf` | 58 个可动关节（含手指、耳朵、眼睛） | 不用 |
| `urdf/robot_retarget.urdf`、`robot_retarget_ros.urdf` | 26 个可动关节（腰 3、臂 12、头 3、腿 8）：轮子和腰部推杆已固定，腕 pitch 下面多了一个固定的 `*_hand_base_link`（向下 3.9 cm）。**FK 实测（200 组随机姿态）**：把两个肩 roll 的符号取反后，和 `nemov3.xml` 的连杆位姿误差 < 0.4 mm；但它是**旧的符号约定**（= `NEMOV3_NPZ_TO_MJCF_JOINT_SIGNS`），而且**限位不同**：calf [−2, 2]（训练模型 [0, 1.571]，会允许膝盖反折）、hip roll [−0.15, 1.571]（训练 [−0.174, 1.047]）、肩 pitch [−3.14, 0.35]（训练 [−3.14, 1.57]） | **不直接用**。只借用它的 hand_base 点当手的末端；如果肩 pitch 0.35 是真实的硬件限位，要问硬件同事后再同步进训练模型 |
| `config/kinematics/*.yaml` | 腰、颈、眼睛的并联机构求解配置 | 不用（这些关节都焊死了） |
| booster_mjlab `assets/amp_sources/nemov3_skating/` | 旧的滑冰数据：`Robot_Model0530_V2_Skating*_IK_control`、SkatingFast01/03/06、Left/Right 转弯等，CSV 共 10 段（其中一部分另有 NPZ）。是另一个项目做的**机器人 IK 动作**，不是人体视频，只有直线蹬冰和简单转弯 | 新数据的**基准**：新花滑片段的根高度、蹲姿、轮速都可以拿它来对照 |
| booster_mjlab `assets/amp/nemov3_skating_straight/` | 转换好的 3 段 `MotionFile` pkl | 格式范例 |
| booster_mjlab `submodules/UMR` | UMR，commit `b45d45c` | Phase 5 直接用 |
| booster_mjlab `tasks/velocity/mdp/wheel_feet.py` 等 | nemov3 的轮足相关 MDP 项 | 训练时参考 |

### 2.4 资产与数据的保密 / 版权

- 本仓库（`supermimic_vibed`）是**公开 fork**。
- **2026-10-24 调整**：一开始的做法是 nemo_v3 资产不提交（只放在已忽略的 `retarget/assets/`）。用户决定**提交资产**，`retarget/assets/` 随仓库公开；`third_party/` 下的 GVHMR、GMR 改成 **submodule**，固定在验证过的 commit。
- 仍然不提交的：`retarget/data/`（拍摄 / 下载的视频有肖像和版权问题，还有估计结果、对比视频）、GVHMR 权重（在 submodule 的 `inputs/checkpoints/` 里）、SMPL / SMPL-X 人体模型（需要官方授权，只用软链接）。
- 公开文档已去敏：去掉了内网 IP、主机路径、私有仓库地址。

---

## 3. 核心难点：人类滑冰 → nemo_v3

| 人类滑冰 | nemo_v3 上的对应 | 重定向时怎么处理 |
|---|---|---|
| 冰刀沿刃方向滑行，几乎不侧滑 | 轮子沿滚动方向滚，侧向有摩擦 | **物理上很像**。把冰刀方向（脚的 yaw）映射成轮子朝向，靠 **hip yaw**（±135°，范围很大） |
| 蹬冰：冰刀斜着向外推 | 轮子斜着摆、沿自身方向滚动外推（类似单排轮滑的 swizzle） | 保留脚的轨迹和朝向；但轮子有电机，**不蹬冰也能前进**，所以蹬冰只是表演动作 |
| 刃的内刃 / 外刃（脚踝侧倾） | 轮子外倾 | **没有踝关节**，只能靠 hip roll 让整条腿倾斜来近似，而且会连带改变轮子的位置。降低这一项的期望 |
| 脚尖点冰、踮脚 | 轮子没有"脚尖" | 丢掉脚的 pitch，只保留位置和 yaw |
| 压步：外侧腿从内侧腿前面交叉过去 | 髋只能内收 10°，轮距 17–25 cm，两腿交叉会撞在一起 | **基本做不到**，降到 B / C 级。可以用"两轮一前一后错开"的近似版 |
| 双足旋转 | 两轮反向滚动（差速），绕竖直轴转 | 用滚动约束算轮速即可（5.3 节） |
| 单足滑行、燕式 | 单轮着地，侧向平衡像独轮车 | 难，放到 B 级 |
| 单足高速旋转（3–5 圈/秒） | 单轮原地自转要靠侧滑或扭转摩擦，**物理上很难** | 暂不做，或者放慢很多倍 |
| 跳跃（空中转体） | 14.6 kg 的轮足能小跳，但空中转体基本做不到 | 不做 |
| 躯干扭转、侧弯 | **腰焊死**，躯干和骨盆是一个刚体 | 只能整体转动骨盆；人的躯干扭转会被 IK 折中掉 |

**结论**：要跟踪的是"身体 + 轮子的位置和朝向"，脚的滑动是**有意的、要保留的**。这和普通 humanoid 重定向的习惯正好相反，普通流程会把脚钉住防止滑动。

---

## 4. 方案调研

### 4.1 视频 → 人体动作

| 方案 | 人体模型 | 相机 | 防滑后处理 | 输出 | 许可 | 备注 |
|---|---|---|---|---|---|---|
| **GVHMR**（zju3dv，SIGGRAPH Asia 2024） | SMPL-X 参数（`smpl_params_global` / `smpl_params_incam`） | 加 `-s` = 静止相机；不加 = 用视觉里程计估计相机运动（README 里有 DPVO 和自研 SimpleVO 两种说法，以代码为准）；可用 `f_mm` 指定焦距 | demo 里 `postproc=True` 写死在代码里；raw 网络输出保存在 `net_outputs`，**可以对比后处理前后脚的轨迹有没有被钉住** | `hmr4d_results.pt` | 未核实，用前看 LICENSE | GMR 直接支持 |
| **GEM-X**（NVlabs，2026-03 发布） | SOMA（77 关节，含手和脸） | 默认估计相机运动（VO）；ROBOTIS fork 有 `--static_cam` | `--postprocess` 打开 Contact/IK 修正，**滑冰时先关掉** | `motion.pt`；ROBOTIS fork 的 `demo_soma.py --retarget` 会同时导出 SOMA BVH | 代码 Apache-2.0，权重另有 NVIDIA 条款 | 和 soma-retargeter、UMR（BONES-SEED / SOMA 输入）衔接 |

### 4.2 人体动作 → 机器人

| 方案 | 输入 | 方法 | 加新机器人的工作量 | 脚接触处理（对滑行的风险） | 许可 / 成熟度 |
|---|---|---|---|---|---|
| **GMR**（YanjieZe，ICRA 2026） | SMPL-X（AMASS / GVHMR）、BVH（LAFAN1 / Nokov / Xsens）、FBX | mink + MuJoCo 两阶段 IK（`ik_match_table1/2`） | 写一份 `smplx_to_nemo_v3.json`，在 `params.py` 加 4 行 | README 里没有提到钉脚类的后处理（**待在代码里确认**，见 Phase 2）→ 预计对滑行最友好 | MIT；17+ 机器人，含 Booster T1 / K1；**输出格式 = booster_mjlab `MotionFile`** |
| **UMR**（hanyang9，arXiv 2609.02134） | SMPL / SMPL-X、BONES-SEED (SOMA)、LAFAN1 等；GVHMR 没有现成 adapter | 学习人体和机器人表面点云的对应关系，再做带约束的优化 | 复制一份 robot config，在 UMR Studio 里摆 T-pose（**nemo_v3 的零位就是 T-pose**），不用写映射；第一次运行会自动建点云对应 | 论文强调接触；滑行时会不会被当成静止接触**需要验证**；轮子表面和人脚表面的对应效果也要验证 | MIT；论文报告在 G1 上下游 tracking 比 GMR 好；**已在 booster_mjlab submodule 里** |
| **soma-retargeter**（NVIDIA） | SOMA BVH；SMPL 系列可经 SOMA-X 转换 | GPU IK（Newton + Warp），带接触稳定化和关节限位裁剪 | GUI 配置器（`robot_config_generator.py`）：导入 URDF/MJCF → 设比例 → 设零位 → 做映射；**不能在 GUI 里新建虚拟 link** | 有 foot-plant 修正：慢速滑行可能被当成"踩地"钉住 → 对滑冰片段要设 `enable_contact_processing=false` | Apache-2.0，**active beta**；内置 `booster_t1` |
| **robot_retargeter**（HighTorque-Locomotion，fork 自 ccrpRepo）<br>（你说的 "robot_retargetor" 我推测是这个） | SMPL-X npz、其他机器人的 CSV（robot → robot） | mink + MuJoCo；逐段按连杆长度缩放；膝角偏置 | 在 `asset/` 放模型，在 `config/` 写 YAML（schema 文档里没写，要看代码） | 有 `contact_pos_fixed_factor` 防滑 → **滑冰片段要关掉** | 许可未写明；支持 **G1 → 其他机器人**，可以复用现成的 G1 动作 |

### 4.3 为什么选 GMR + GVHMR 当出发点

1. **格式零摩擦**：booster_mjlab 的 `MotionFile` 就是 GMR 的 pkl 结构，只需要把关节列从 IK 用的 23 列（full）或 18 列（hw）扩成 32 列（5.3 节）。
2. **链路最短**：`gvhmr_to_robot.py` 一条命令从 GVHMR 结果直接出机器人动作；CPU 实时，调参快。
3. **加机器人最透明**：JSON 里每个 link 跟踪哪个人体关节、位置 / 旋转权重、偏置都能直接看到和修改，方便做 5.1 节的"轮子虚拟点"技巧。
4. **预计不会钉脚**：README 没有提到脚接触修正。Phase 2 会用滑行片段实际验证，看脚的速度有没有被压成 0。
5. 你本来的判断（"GVHMR + GMR 即可完成"）是对的。UMR 已经在 booster_mjlab 里，作为第二阶段对照的成本很低。

---

## 5. 轮子的专门处理（本计划的核心）

### 5.1 生成两份重定向专用 MJCF：`full` 和 `hw`

**用 `nemov3.xml` 生成，不从 URDF 重新转**。这样关节命名、轴向、限位、零位和训练用的模型完全一致。`train_v2.urdf` 和它完全等价；`robot_retarget.urdf` 的符号约定和限位不同，不用（见 2.3 节）。

具体做法是用 `mujoco.MjSpec` 读入 `nemov3.xml`，复用 booster_mjlab `nemov3_constants.py` 里的 `_weld_joint()` 焊死关节，焊在 `HOME_KEYFRAME` 的角度（没列出的为 0）：

| 版本 | 焊死的关节 | IK 可动关节 | 用途 |
|---|---|---|---|
| **`nemov3_retarget_full.xml`** | 头 3、腰部推杆 4（`joint_[rl]LinearOut`、`joint_[rl]Linear`）、轮子 2 | **23 个**：腰 3、臂 12（含腕 pitch）、腿 8 | 给 tracking（SuperMimic / BeyondMimic）和将来放开腰、腕的硬件用 |
| **`nemov3_retarget_hw.xml`** | 上面这些，再加腰 3、腕 pitch 2（= `NEMOV3_FIXED_HEAD_WAIST_JOINTS` + 轮子） | **18 个**：臂 10、腿 8 | 给现在的硬件和 AMP 用 |

**为什么要两份，而不是只做 full 版再把腰的列删掉**：
- booster_mjlab 已经有 `drop_fixed_head_waist_joints()` 可以删列，但只适合**原本就没用到腰**的数据；
- 如果参考动作用腰前倾了 20°，删掉腰的列以后，剩下的髋角是"腰已经帮忙弯了"的髋角。焊死腰的机器人要做同样的躯干姿态，髋就得多弯，这时 AMP 判别器反而会把这个正确的补偿判成"不像专家"；
- tracking（按 body 位姿跟踪）则没这个问题：用 full 版当参考，焊死的机器人会自己用髋去补偿。

两份用**同一份 IK 配置**（5.2 节的 JSON 只引用两份模型都有的 body），从同一份人体数据各跑一次 GMR。IK 在 CPU 上实时，多跑一次几乎不花时间。

**腰部的额外注意事项**：
- 腰是**双直线推杆的并联机构**（`config/kinematics/` 里的 `torso.dual_linear.v1`）。MJCF 用串联的 pelvisYaw/Pitch/Roll 来等效，限位是 pitch [0, 0.524]、roll ±0.175 的"方盒"；真实的推杆工作空间是 pitch 和 roll **耦合**的，方盒会高估。将来上真机前，要用这个求解器把串联角度换算成推杆长度，再检查一次。
- **腰只能前倾**（pitch 0–30°），不能后仰，所以 Ina Bauer、后仰这类动作即使放开腰也做不出来。
- URDF 里腰 pitch/roll 的 effort 和 velocity 都是 0（因为它们是被推杆带动的），不能当作速度上限用。
- URDF 里**左腕 pitch** 的 effort 5.6 / velocity 1.27，右腕是 1.27 / 40.2，看起来是左边填反了，建议和硬件同事确认。

**两份共同的步骤**：
1. **焊死两个轮关节**（角度 0）：IK 不能去动轮子，否则会出现"靠转轮子来凑脚的位置"这种无意义的解。
2. **加 4 个无质量的虚拟 body**，全部挂在 **calf link** 上（calf 不随轮子转）：
   - `left_wheel_center`：位置 = 轮心。左腿在 `left_calf_link` 坐标系下是 `(0, −0.043, −0.175)`（实测 `left_wheel_link` 相对 calf 的位置）；
   - `left_wheel_heading`：轮心沿滚动方向（calf 坐标系的 +x）往前 0.12 m，即 `(0.12, −0.043, −0.175)`；
   - 右腿同理，用 `right_calf_link` 里 `right_wheel_link` 的实测位置（y 的符号以模型为准）。

   **轮心和 heading 两个点一起决定轮子朝向，而且只用位置任务**，避开了"人脚的 pitch 在轮子上没意义"的问题。
3. **加 2 个手末端虚拟 body**：`link_[lr]WristPitch` 下方 `(0, 0, −0.039)`，取自 `robot_retarget.urdf` 的 `*_hand_base_link`，用来跟踪人的手。
4. 保存成 xml（`spec.to_xml()`），mesh 路径指回 `booster_mjlab_nemov3/xmls/meshes/`。最后检查：
   - IK 可动关节数（full 23、hw 18）；
   - 零位是 T-pose；
   - 零位时轮心离地 = 0.075 m。

### 5.2 GMR 的 IK 配置：`retarget/configs/smplx_to_nemo_v3.json`

结构参考 `smplx_to_t1.json`。顶层字段：

| 字段 | 值 |
|---|---|
| `robot_root_name` | `"link_leg2pelvis"` |
| `human_root_name` | `"pelvis"` |
| `ground_height` | 0.0 |
| `human_height_assumption` | 1.8（之后按拍摄者实际身高调整） |
| `use_ik_match_table1/2` | 都为 true |

每条 `ik_match_table` 的格式是：
```
robot_body: [human_body, pos_weight, rot_weight, pos_offset_xyz, rot_offset_wxyz]
```
初版映射如下。人体 body 名称是 SMPL-X 的标准关节名，具体键名以 GMR 的人体数据为准；`rot_offset` 要在 viewer 里对齐坐标系后再定。

| robot body | human body | 位置权重 | 旋转权重 | 说明 |
|---|---|---|---|---|
| `link_leg2pelvis` | `pelvis` | 100 | 10 | 根位置 |
| `link_torso` | `spine3` | 0 | 30 | full 版：腰的 3 个关节直接跟胸口朝向；hw 版：躯干和根是同一个刚体，这一项让 IK 在骨盆朝向和胸口朝向之间折中，偏向胸口，因为滑冰的"线条"看上身 |
| `left/right_wheel_center` | `left/right_ankle` | 50 | **0** | 只要位置。z 偏置 ≈ 0.075 −（人体踝高 × 缩放），再按冰鞋高度微调 |
| `left/right_wheel_heading` | `left/right_foot`（SMPL-X 的脚尖） | 20–30 | 0 | 决定轮子朝向（hip yaw） |
| `left/right_calf_link` | `left/right_knee` | 10 | 0 | 膝的位置 |
| `link_[lr]Elbow` | `left/right_elbow` | 20 | 0 | 上肢编排是花滑观感的重点，权重比 T1 高 |
| `[lr]_hand_point`（5.1 节第 3 步） | `left/right_wrist` | 20 | 5 | 手的位置和朝向；full 版会用到腕 pitch，hw 版靠腕 roll 跟一点前臂扭转 |
| `link_[lr]ShoulderPitch` | `left/right_shoulder` | 5 | 0 | 肩的位置，弱约束 |

**`human_scale_table` 初值**（按身高 1.75 m 的成人估算，Phase 2 再调）：

| 部位 | 计算 | 缩放 |
|---|---|---|
| 腿、躯干 | 机器人髋 → 轮心 0.34 m / 人体髋 → 踝约 0.82 m | **≈ 0.40** |
| 手臂 | 上臂 0.103 / 0.29、前臂 0.092 / 0.25 | **≈ 0.36** |

作为对比，T1 用的是 0.6–0.7。nemo_v3 大约只有成人尺寸的 40%。

**GMR 的两个已知坑**：
- `gvhmr_to_robot.py` 把输出重采样成 **30 fps**（`tgt_fps = 30` 写死在脚本里）。旋转和快速步法建议改成 50 或 60 fps，和训练的 50 Hz 对齐。
- `--loop` 和 `--save_path` 一起用时，正常流程下不会存档。

### 5.3 后处理：`retarget/scripts/wheel_postprocess.py`（待写）

输入是 GMR 的 pkl（full 版 23 列、hw 版 18 列 `dof_pos`）。输出是 booster_mjlab 的 `MotionFile`（都是 32 列，文件名带上 `_full` / `_hw` 后缀）。用**原始 `nemov3.xml`** 做 FK，逐帧计算：

1. **扩列**：按名称把 IK 的列放进 `NEMOV3_JOINT_ORDER` 的 32 列。焊死的关节填 `HOME_KEYFRAME` 的值（没列出的填 0）；两个轮关节由第 6 步计算。
2. **轮轴方向** `a(t)`：calf link 旋转 × 局部 y 轴（轮关节轴是 `[0, 1, 0]`）。
3. **滚动方向** `d(t) = normalize(a × n)`，其中 `n` 是地面法向 `(0, 0, 1)`。符号要和轮子正转的方向一致，用一段直线前进的数据来定。
4. **轮心速度** `v(t)`：用轮心位置做有限差分（先平滑，见第 9 条）。
5. **是否着地**：轮底高度 `h_b = z_c − r·cos(φ)`，其中 `z_c` 是轮心高度、`φ` 是外倾角、`r = 0.075`。**不要用 `*_wheel_ground` site**（它会随轮子转，见 2.1 节）。`h_b < 2 cm` 就算着地。
6. **轮速和轮角**：着地时 `ω = (v · d) / r`；离地时 `ω` 逐渐衰减到 0。轮角 `θ = Σ ω·Δt`，写进 `dof_pos` 的轮关节列。同时检查 `|ω| ≤ 25 rad/s`。
7. **侧滑比** `slip = |v · a| / (|v| + ε)`：着地帧里 `slip > 0.2` 说明这段动作要求轮子横着走，物理上做不到。可能是人体估计出错，也可能是动作本身不适合轮子。标记出来，人工判断要不要剔除。
8. **地面对齐**：按着地帧 `h_b` 的中位数整体平移根的 z；如果有轮子穿地，抬高根让轮子刚好接触。
9. **平滑**：根轨迹和关节角用 Savitzky–Golay 或低通滤波（截止频率约 6–8 Hz）。旋转片段要先 unwrap 角度再滤波。
10. **速度和限位**：
    - 关节角裁剪到限位以内（训练里用的是 `soft_joint_pos_limit_factor = 0.9`，留出余量）；
    - 腿关节速度超过 10 rad/s、根水平速度超过 1.3 m/s 时，对整段做**时间拉伸**，或者**只缩放根的水平位移**，不要逐帧硬裁剪；
    - **手臂**：硬件上肩 yaw、肘、腕 roll 只有 kp 1.5、力矩 1.27 Nm，快速甩臂跟不上。手臂部分建议单独低通（约 3 Hz）。
11. **`local_body_pos` / `link_body_list`**：用 FK 算出来一并写入，镜像增强会用到。写法参考 `nemov3_skating_to_pkl.py`。

### 5.4 下游训练要配合的地方（给 booster_mjlab）

- 轮关节已经是 `<velocity>` actuator，不进关节角跟踪或 AMP 的关节特征。现有代码里已经有排除 `.*_wheel_joint` 的做法。
- 如果做 tracking：只跟踪非轮子 body 的位姿；轮子 body 只跟踪位置，不跟踪绕轮轴的旋转。
- **去掉所有"脚滑动惩罚"项**：这里滑行是目标。
- 新动作的蹲姿和 `HOME_KEYFRAME` 差得多的话，要注意 `nemov3_constants.py` 注释里记录过的问题：AMP 判别器会被"默认姿态的静态偏差"轻易区分。
- 轮子接触建模可以参考单排轮滑 humanoid 的论文（arXiv 2606.31807，Booster T1 + 被动轮）：他们用**球形 / 椭球形轮子 geometry** 来减少仿真接触伪影。

---

## 6. 实施步骤

每个阶段都写了产出和验收标准，**验收通过再进入下一阶段**。

### Phase 0：模型盘点（✅ 大部分已完成，2026-10-24）

- ✅ 已从 lyh 的电脑拷贝 `nemo_v3_2_fullbody/`（171 MB）和 booster_mjlab 的 `robots/nemov3/`（85 MB）到 `retarget/assets/nemo_v3/`（后来改为随仓库提交，见 2.4 节）；
- ✅ 第 2.1 节的表已用 MuJoCo 实测填完；
- ✅ 已确认 motion 格式（2.2 节）和已有工作（2.3 节）；
- ✅ 已用 FK 核对：`train_v2.urdf` 和 `nemov3.xml` 完全等价；`robot_retarget.urdf` 是旧的肩 roll 符号约定，限位也不同（2.3 节）；
- ✅ 已生成 `retarget/assets/nemo_v3/retarget_mjcf/nemov3_retarget_{full,hw}.xml`（5.1 节）：
  - 生成脚本：`retarget/scripts/make_retarget_mjcf.py`；
  - 检查脚本：`retarget/tools/check_retarget_mjcf.py --render <png>`，截图在 `retarget/data/retarget_mjcf_check.png`。

  检查结果：

  | 检查项 | 结果 |
  |---|---|
  | IK 可动关节 | full 23 个、hw 18 个 |
  | 和 `nemov3.xml` 的连杆位姿（200 组随机姿态，焊死关节取焊接角度） | 误差 ≤ 4e-7 m / 4e-7 rad |
  | 轮心标记 | 和原模型 `*_wheel_link` 的原点重合 |
  | heading 标记 | 在轮心前方 0.12 m，且垂直于轮轴 |
  | 腿伸直站立 | 根高 0.5099 m 时，轮心正好离地 0.075 m |
  | 截图 | 零位为 T-pose；full 版能弯腰和转腕 pitch，hw 版不能 |

  重新生成：`python retarget/scripts/make_retarget_mjcf.py && MUJOCO_GL=egl python retarget/tools/check_retarget_mjcf.py --render retarget/data/retarget_mjcf_check.png`（需要 mujoco；本机是在 `supermimic:latest` 容器里跑的）

### Phase 1：搭环境，用现成机器人跑通整条链（1 天）

```bash
# GVHMR（需要 GPU；按它的 docs/INSTALL.md 安装，并下载 SMPL / SMPL-X 模型和 checkpoint）
git clone https://github.com/zju3dv/GVHMR retarget/third_party/GVHMR
cd retarget/third_party/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
# 产出 outputs/demo/tennis/hmr4d_results.pt

# GMR（CPU 即可）
git clone https://github.com/YanjieZe/GMR retarget/third_party/GMR
cd retarget/third_party/GMR && uv sync      # 或 conda + pip install -e .
# SMPL-X 模型放到 assets/body_models/smplx/
uv run scripts/gvhmr_to_robot.py --gvhmr_pred_file <hmr4d_results.pt> --robot booster_k1 --record_video
```

**验收**：Booster K1（booster_mjlab 里也有这台）能跟着示例视频动起来，证明环境没问题。之后出问题就只可能是 nemo_v3 的配置。

### Phase 2：把 nemo_v3 加进 GMR（2–3 天）

1. 把两份 retarget MJCF 放到 `GMR/assets/nemo_v3/`（或用软链接），在 GMR 里注册成 `nemo_v3_full` 和 `nemo_v3_hw` 两台机器人。
2. 在 `general_motion_retargeting/params.py` 里加：
   - `ROBOT_XML_DICT["nemo_v3_full"]`、`ROBOT_XML_DICT["nemo_v3_hw"]`
   - `IK_CONFIG_DICT["smplx"]` 的两项都指向同一个 `smplx_to_nemo_v3.json`
   - `ROBOT_BASE_DICT[...] = "link_leg2pelvis"`
   - `VIEWER_CAM_DISTANCE_DICT[...]`（约 1.5，因为机器人小）
3. 写 `smplx_to_nemo_v3.json`（5.2 节）。
4. 先用**普通动作**调参：AMASS 的走路、转身、蹲起，或者 GVHMR 示例视频。重点看：
   - 轮子朝向跟不跟脚；
   - 深蹲时 calf 有没有超限位（calf 只有 0–90°）；
   - 根高度对不对；
   - 两轮会不会互相穿插。
5. 在 GMR 代码里搜一遍 `ground` / `contact` / `foot` 相关的处理，确认它不会把滑动的脚钉住（README 没提，但要以代码为准）。
6. 写好 `wheel_postprocess.py`（5.3 节）。用旧的 `skating01.pkl` 反算 `slip` 作为"合格数据"的基准；再在普通走路动作上验证 `slip` 会明显偏高，因为人是在迈步。
7. 生成的 pkl 用 booster_mjlab 的 `visualize_motions.py` 播放，确认格式能被读进去。

**验收**：5 段普通动作，限位违规为 0，穿地 < 1 cm，肉眼看不出镜像或扭转的肢体；pkl 能被 booster_mjlab 正常加载。

### Phase 3：拍视频，比较人体估计方案（3–4 天，可以和 Phase 2 并行）

1. 按第 7 节的规范拍 **5 段测试片段**：直线蹬冰、弓步滑行、双足慢转、蛇形滑、一段上肢编排。
2. 同一批片段跑两套人体估计：
   - **GVHMR**：固定机位加 `-s`；
   - **GEM-X**：
     ```bash
     gem-infer motion-video --task body --input clip.mp4 \
         --config docs/configs/body_video_native77.yaml --soma-assets assets/soma --output motion.pt
     ```
     先**不加** `--postprocess`，再加上对比一次。
3. 比较以下几项：
   - 全局轨迹是否合理（直线滑行应该是直线，没有漂移或跳变）；
   - 滑行段脚的速度有没有被压成 0（**被钉脚是滑冰数据的头号杀手**）；
   - 旋转圈数是否和视频一致；
   - 遮挡、动作模糊时会不会崩。
4. GVHMR 还要对比 `net_outputs`（后处理之前）和 `smpl_params_global`（后处理之后），确认后处理有没有把滑行改掉。

**验收**：选定一套人体估计方案，写下理由和数字。如果选 GEM-X，因为 GMR 不吃 SOMA，改走 UMR（支持 BONES-SEED / SOMA 输入）或 soma-retargeter。

### Phase 4：批量重定向和质量过滤（2–3 天）

1. 建一个片段目录表 `retarget/clips.csv`，列包括：
   - 片段 id、来源、动作类型、fps、起止时间；
   - 拍摄者身高；
   - 质量标记（slip、穿地、超限位、是否时间拉伸）。
2. 批量执行：人体估计 → GMR → `wheel_postprocess.py` → 质量检查（第 9 节）。
3. 输出 `MotionFile` pkl，放进 booster_mjlab 的 `assets/amp/nemov3_figure_skating/`（新目录名可再定）。
4. 用可视化工具逐段过一遍：booster_mjlab 的 `visualize_motions.py`，或者照本仓库 `tools/viser_play.py` 的思路做一个浏览器播放器。

**验收**：至少 10 段 A 级动作通过全部质量检查。

### Phase 5：重定向方案对照（3–5 天，可选但建议做）

在同一批片段上跑：
- **UMR**（booster_mjlab `submodules/UMR`）：
  - 复制 `humanoid_retarget_booster_k1_example.json` 改成 nemov3；
  - 在 UMR Studio 摆 T-pose（零位即可）；
  - 写 GVHMR → SMPL-X npz 的 adapter；
  - 同样要焊死轮子和腰。
- **soma-retargeter**：用 `robot_config_generator.py` 注册 nemo_v3，**关掉 contact processing**。

和 GMR 的结果一起拿去训练（AMP 或短时 tracking）。比较：
- 跟踪误差（body 位置、朝向）；
- 摔倒率；
- 轮子侧滑。

GMR 论文的结论正是"重定向质量会直接影响 tracking"，所以要用下游指标来选，不看肉眼观感。

**验收**：一张对照表，定下正式用的重定向方案。

### Phase 6：交给训练（接口）

把导出的 pkl、`clips.csv` 和 5.4 节的注意事项交给 booster_mjlab：

1. **先扩充现有的 AMP 速度任务**：用 **`_hw` 版**的 A 级片段加进 AMP 数据集，看风格是否更丰富、速度跟踪有没有变差；
2. **再考虑做 nemov3 的 tracking 任务**，让它逐段跟踪花滑编排（像本仓库 SuperMimic 那样一个策略跟多段动作）。用 **`_full` 版**当参考：焊死的机器人照样能跟，会自己用髋去补偿腰；将来硬件放开腰和腕以后，同一批参考动作可以直接沿用。
   - anchor body 建议用根 `link_leg2pelvis`，而不是躯干：腰焊死时，躯干朝向跟不上参考，用躯干当 anchor 会把这个误差传到轮子的位置上；
   - policy 观测里如果有参考关节角，焊死机器人要去掉腰和腕 pitch 那几维。

实验记录按 booster_mjlab 的习惯写进 `tools/nemov3_finetune.md`。

---

## 7. 视频采集规范（自己拍）

| 项目 | 要求 | 原因 |
|---|---|---|
| 机位 | **三脚架固定**，侧面或斜 45°，能拍到整个动作区域 | 静止相机可以加 `-s` / `--static_cam`，避开冰面纹理太少导致的 VO 失败 |
| 背景 | 画面里要有场边围板、柱子之类的纹理 | 万一需要估计相机运动，纯白冰面没法估 |
| 帧率 | ≥ 60 fps；旋转片段 120–240 fps（手机慢动作也行，但要记下**真实帧率**） | 减少动作模糊；慢动作文件的 fps 标签可能是 30，要手动改正 |
| 构图 | 全身始终在画面内，脚不能出框；人占画面高度 1/3 以上 | 脚是重定向最关键的部位 |
| 服装 | 贴身、和冰面颜色有对比；不穿长裙摆 | 减少 2D 关键点的歧义 |
| 每个动作 | 重复 3–5 次，片段之间留 2 秒静止 | 挑最好的一次；静止段方便对齐地面高度 |
| 动作幅度 | **请滑行者把速度放慢**（目标 ≤ 3 m/s），双脚间距保持在肩宽左右 | 机器人缩放到 40% 后最高只有 1.3–1.9 m/s，轮距也窄 |
| 元数据 | 记录拍摄者身高、冰鞋 / 轮滑鞋类型、场地 | 用于 `human_height_assumption` 和偏置 |
| 授权 | 拍摄者签肖像使用同意书；**视频不进公开仓库** | 隐私和肖像权 |

**冰上 vs 花样轮滑**：
- 冰刀可以在冰面上转得很快，有些动作轮子做不到；
- 花样轮滑本身就是"轮子滚动 + 花滑动作"，力学上和 nemo_v3 更接近。

建议两种都拍，Phase 4 统计哪一种的 `slip` 更低、训练效果更好。

**比赛转播视频**：镜头一直在移动、有剪辑、慢动作回放的帧率混在一起，而且有版权。只用来**做动作清单参考**或少量研究用途，不作为主要数据源。

**现成的 3D 数据**：FS-Jump3D（名古屋大学，12 机位无标记动捕，花滑**跳跃**，86 个关键点）。跳跃对轮足用处不大，但可以用来**检验人体估计方案在花滑上的精度**；许可要另外查。

---

## 8. 动作清单和可行性分级（按实测关节限位修订）

| 等级 | 动作 | 依据 |
|---|---|---|
| **A：第一批** | 前进蹬冰（stroking）加上肢编排、弓步滑行（lunge，一腿前一腿后）、双足滑行时的手臂造型、双轮慢速原地转（two-foot spin，放慢）、蛇形滑（slalom）、开合脚滑行（swizzle） | 双轮着地；用到的是 hip yaw（±135°）、hip roll 外展（最多 60°）和手臂，都在限位内 |
| **B：第二批** | 大一字（spread eagle）、Ina Bauer、后退滑行、单足滑行加上肢、燕式（spiral） | 大一字靠两侧 hip yaw 外转，限位够，但两轮要沿同一条线反向摆，需要验证平衡；单足滑行和燕式要像独轮车那样做侧向平衡，**训练难度高**；后退滑行方向反过来，要确认 AMP 速度任务支持后退 |
| **C：暂不做** | 压步（crossovers）、单足高速旋转（scratch / camel / sit spin）、跳跃、托举和双人动作 | 压步：髋只能内收 10°，轮距 17–25 cm，两腿交叉会撞；sit spin：单腿深蹲加单轮自转；跳跃：空中转体。如果一定要做旋转，先放慢 3–5 倍做"双轮展示版" |

**根速度**：花滑运动员滑行可以达到每秒好几米。超过 nemo_v3 的速度范围（训练 1.3 m/s，硬件约 1.9 m/s）时，**只缩放根的水平位移**，身体姿态不变。

---

## 9. 质量检查指标（`retarget/scripts/quality_check.py`，待写）

| 指标 | 阈值（初值，可调） |
|---|---|
| 关节限位违规（裁剪前） | 记录比例；裁剪后为 0（按 0.9 软限位） |
| 腿关节速度 | ≤ 10 rad/s（超了就时间拉伸） |
| 轮速 | ≤ 25 rad/s（≈ 1.9 m/s）；进 AMP 速度任务时建议 ≤ 17 rad/s（1.3 m/s） |
| 手臂关节速度 | 肩 ≤ 23.9 rad/s，其余 ≤ 40.2 rad/s；考虑到硬件 kp 只有 1.5，实际要求更严 |
| 轮子穿地 | < 1 cm |
| 着地帧轮子悬空 | < 2 cm |
| 着地帧侧滑比 `slip` | 平均 < 0.1，峰值 < 0.3；以旧 skating 片段的数值为基准 |
| 两轮 / 两腿之间的碰撞 | 0（MuJoCo 机器人 body 之间的接触） |
| 根高度 | 在 [0.30 m，0.52 m] 之内（腿伸直是 0.51 m，HOME 蹲姿是 0.50 m） |
| 下游（Phase 5） | AMP / tracking 不摔、跟踪误差小 |

---

## 10. 目录结构

```
retarget/
├── README.md              # 安装依赖与常用指令
├── MILESTONE.md           # 本文件：计划书 + 开发记录
├── configs/               # smplx_to_nemo_v3.json（GMR IK 配置）
├── patches/               # gvhmr_skating.patch
├── scripts/               # 主流程（产出数据）：make_retarget_mjcf、gvhmr_camera_traj、gvhmr_to_nemov3、run_video_to_nemov3.sh
├── tools/                 # 检查与可视化：check_retarget_mjcf、render_compare、viser_play_motion（以后的 quality_check 也放这里）
├── assets/nemo_v3/        # 随仓库提交：booster_mjlab_nemov3/、nemo_v3_2_fullbody/、retarget_mjcf/
├── third_party/           # submodule：GVHMR、GMR
└── data/                  # ⚠ 不提交：视频、人体估计结果、重定向结果
```

---

## 11. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 人体估计的后处理把滑行钉成"踩地" | 滑行全部丢失 | 关掉或对比后处理（Phase 3 第 4 条）；确认 GMR 的代码不钉脚；soma-retargeter 和 robot_retargeter 要关掉 contact 相关选项 |
| 冰面纹理太少，相机运动估计失败 | 全局轨迹漂移 | 固定机位加 `-s`；画面里留围板等纹理 |
| 旋转时动作模糊、自遮挡 | 关节翻转、圈数错 | 高帧率拍摄；放慢；这类片段单独人工检查 |
| 没有踝关节 | 轮子外倾跟不上；hip roll 一动，轮子的位置也跟着变 | 外倾不作为跟踪目标；降低对应权重；接受误差 |
| 腰焊死 | 躯干扭转、侧弯丢失，上身"线条"变僵 | 同时出 full / hw 两版（5.1 节）；hw 版里 `link_torso` 的旋转权重偏向胸口；硬件放开腰以后直接换用 full 版 |
| 手臂电机弱（kp 1.5、1.27 Nm） | 快速甩臂在真机上跟不上 | 手臂单独低通；选慢节奏的编排 |
| 缩放后速度太快 | 超出 1.3 m/s 的训练范围 | 只缩放根的水平位移，或时间拉伸 |
| 人体动作要求轮子侧滑（`slip` 高） | 训练学不会，或者学出奇怪的动作 | 过滤掉；或者在重定向里调大 heading 点的权重 |
| GEM-X、soma-retargeter 还在快速迭代（beta） | API 变动 | 记录 commit hash；基线用 GMR |
| 许可：SMPL-X 模型、GVHMR、GEM-X 权重 | 不能商用或再分发 | 研究用途；用之前逐个确认 LICENSE |
| nemo_v3 资产公开 | 公开仓库包含内部机器人模型 | 用户已决定随仓库提交（2026-10-24）；文档中的内网 IP、路径、私有仓库地址已去敏 |

---

## 12. 时间估计（一个人）

| 阶段 | 时间 |
|---|---|
| Phase 0 剩余：生成两份 retarget MJCF | 0.5 天 |
| Phase 1 环境 | 1 天 |
| Phase 2 nemo_v3 接入 GMR + 后处理 | 2–3 天 |
| Phase 3 拍摄 + 人体估计对比 | 3–4 天（可以和 Phase 2 并行） |
| Phase 4 批量和质量过滤 | 2–3 天 |
| Phase 5 方案对照（可选） | 3–5 天 |
| **合计到第一批可训练的动作** | **约 1.5 周**（不含 Phase 5） |

---

## 13. 参考资料

- GVHMR：https://github.com/zju3dv/GVHMR
- GMR（General Motion Retargeting）：https://github.com/YanjieZe/GMR ，论文 *Retargeting Matters*：https://arxiv.org/abs/2510.02252
- GEM-X：https://github.com/NVlabs/GEM-X ，模型卡：https://huggingface.co/nvidia/GEM-X
- ROBOTIS 的 GEM-X / soma-retargeter 使用文档（视频 → SOMA BVH → 机器人 CSV 的完整例子）：
  - https://docs.robotis.com/docs/systems/aisapiens/motion_generation/gem-x/
  - https://docs.robotis.com/docs/systems/aisapiens/motion_generation_retargeting/soma_retargeter/
- soma-retargeter：https://github.com/NVIDIA/soma-retargeter （配置器：`documentation/robot-configurator.md`，脚接触：`documentation/foot-contact.md`）
- SOMA 人体模型：https://github.com/NVlabs/SOMA-X
- UMR：https://github.com/hanyang9/UMR ，论文 https://arxiv.org/abs/2609.02134 ，项目页 https://hanyang9.github.io/UMR/
- robot_retargeter：https://github.com/HighTorque-Locomotion/robot_retargeter （上游 https://github.com/ccrpRepo/robot_retargeter）
- 单排轮滑 humanoid RL（Booster T1 + 被动轮）：https://arxiv.org/abs/2606.31807
- 轮滑 humanoid AMP（动捕 → 重定向）：https://arxiv.org/abs/2607.10815
- 轮式 humanoid 的重定向（底盘和身体分开处理的思路，可借鉴）：https://arxiv.org/abs/2609.11357 、https://arxiv.org/abs/2610.08381
- FS-Jump3D（花滑跳跃 3D 数据）：https://arxiv.org/abs/2408.16638
- mjlab：https://github.com/mujocolab/mjlab
- booster_mjlab：nemov3 的训练仓库（私有，地址不在此公开）

---

## 14. 初步验证：1 分钟花滑视频 → nemo_v3（2026-10-24，已打通）

### 14.1 数据

- 视频：B 站《花样滑冰等级测试大纲教学视频》（av34903340）p06「第一级」，截取 56–116 s，正常速度的「1级步法完整示范」。
  - 单人，全身入镜；360p，30 fps；
  - **镜头会摇（pan）和变焦**；
  - 动作：单足滑行、弧线、压步、停止。
- 视频只放在 `retarget/data/videos/`（已 gitignore，有版权，**不进仓库**）。
- YouTube 在这台服务器上要求登录做人机验证，所以没用。
- 人体模型：直接链接本机已有的 `~/tasknpoint/submodules/PromptHMR/data/body_models/`（SMPL / SMPL-X，需要官方注册，不从网上另外下载）。

### 14.2 管线（一条命令，用法见 README.md）

| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 | GVHMR `tools/demo/demo.py`，环境变量 `GVHMR_NO_STATIC_PP=1` | 动态相机用 SimpleVO 估计；关掉"钉脚"后处理。需要先打 `retarget/patches/gvhmr_skating.patch` |
| 2 | `scripts/gvhmr_camera_traj.py` | 用相机旋转 + 相机坐标系下的位置**重建世界轨迹**（见 14.3） |
| 3 | `scripts/gvhmr_to_nemov3.py` | 主流程，依次做：<br>① 按本片段实测的腿长 / 臂长算缩放；<br>② 轮子朝向目标按速度方向融合；<br>③ 手臂从机器人肩部按段重建；<br>④ 运行时把 nemo_v3 注册进 GMR（不改 GMR 源码），做 IK（full 版 23 个关节、hw 版 18 个）；<br>⑤ 对齐地面、按滚动约束积分轮角、统计侧滑、平滑；<br>⑥ 扩成 32 列 |
| 4 | `tools/render_compare.py` | 生成 1280×480 左右对比视频（左：原视频，右：MuJoCo 渲染，视角跟随原相机的方位角） |
| 5 | `tools/viser_play_motion.py` | 浏览器交互播放，带原视频画面同步 |

**环境安装**：见 [README.md](README.md)（conda 环境、权重下载、补丁、常用指令）。

### 14.3 遇到的问题和解决

| 问题 | 现象 | 解决 |
|---|---|---|
| SimpleVO 崩溃 | `answer.cam2_from_cam1` 为 `None`：pycolmap 4.x 对纯旋转（摇镜头）的帧对不给相对位姿 | patch：改用单应矩阵 `R = K⁻¹HK` 恢复旋转（SVD 正交化），没有 H 时用单位阵 |
| **滑行被抹掉** | GVHMR 世界轨迹平均速度只有 0.22–0.24 m/s，60 s 只走了 13 m。原因有两个：<br>① 脚静止置信度 > 0.5 的帧约占一半，`pp_static_joint` 把这些脚钉住；<br>② 更根本的是网络的根速度先验来自"迈步"数据：腿不迈步就判断身体不动，加上摇镜头让人一直在画面中间 | ① 加了 `GVHMR_NO_STATIC_PP` 开关；<br>② **用相机重建轨迹**：`pelvis_world = R_c2w · pelvis_cam`，其中 `R_c2w = R_global · R_incam⁻¹`，按约 1 s 平滑。结果平均 1.16 m/s、60 s 走了 70 m（人的尺度），轨迹是连续的弧线 |
| 轮子侧滑大 | 人自身"脚尖朝向 vs 脚的速度方向"的侧滑就有 0.55–0.62：360p 下脚只有约 10 像素，脚的 yaw 估不准 | 轮子朝向目标在脚速 0.3 → 0.8 m/s 之间从"脚尖点"渐变到"速度方向"。侧滑从 0.51 / 0.58 降到 0.33 / 0.37（左 / 右） |
| **手臂垂下、肘部顶到限位** | GMR 按"到骨盆的距离"统一缩放，人和机器人的躯干比例不同，肩的目标高出 8 cm；再加上手臂和膝盖的**旋转目标**用了 T1 的偏置，跟 SMPL-X 的肢体坐标系不匹配，和位置目标互相拉扯。前臂方向余弦只有 0.15，手的误差 0.24 m | ① 肩改成"胸口朝向 × 机器人实际的肩部偏移"，肘和手按机器人的段长、沿人的方向逐段连上；<br>② 肢体旋转权重降到 1e-3。结果方向余弦 1.00，手误差 4 mm，轮心误差 9 mm |
| 新建 conda 环境 | `conda create -y` 遇到同名环境会**直接覆盖** | 建之前先 `conda env list` 确认没有同名环境（这次新建的 `gvhmr`、`gmr` 都确认过不存在） |

### 14.4 结果（`retarget/data/retarget_out/`，已 gitignore）

| 文件 | 内容 |
|---|---|
| `skate60_full.csv` / `skate60_hw.csv` | **给 SuperMimic 用**：格式和本仓库 `data/lafan1_g1/*.csv` 一样（无表头，30 fps，root pos 3 + root quat xyzw 4 + 关节角）。关节是 32 列，按 `NEMOV3_JOINT_ORDER`，含两个轮子。tracking 用 full 版 |
| `skate60_{full,hw}.pkl` | booster_mjlab 的 `MotionFile`（AMP 用 hw 版） |
| `skate60_{full,hw}.json` | 统计数据 |
| `skate60_compare_{full,hw}_480p.mp4` | 原视频和重定向结果的对比视频（1280×480，60 s） |

full 版的主要数字：

| 指标 | 值 |
|---|---|
| 缩放 | 腿 / 躯干 0.458（人腿 0.747 m，偏小，像青少年）；手臂按机器人段长 0.103 / 0.131 m 重建 |
| 根速度 | 平均 0.53 m/s，p95 1.1 m/s（在训练范围 1.3 m/s 以内） |
| 轮速 | p95 约 15–16 rad/s（上限 25） |
| 腿关节速度 | p99 6.8 rad/s（上限 10） |
| 着地帧侧滑（运动时平均） | 左 0.33 / 右 0.37 |
| 顶到限位的关节 | 髋 roll 26–32%（基本是内收的 −0.17：一级动作里有压步，而这台机器人的髋只能内收 10°，**和第 8 节的预判一致**）；腰 pitch 43% 停在 0（人在后仰，而腰只能前倾）；calf 7–8% |
| 格式检查 | CSV 和 pkl 逐项一致，四元数已归一化，关节顺序 = `NEMOV3_JOINT_ORDER` |

### 14.5 还没解决 / 下一步

1. **SuperMimic 本身还是 G1 专用**：`supermimic/robot.py`、`kinematics.py`、`motion.py`（读 CSV 的 `raw[:, 7:36]` 写死了 29 列）、`mdp.py` 都要移植到 nemov3。移植时要注意：
   - 轮关节用速度控制，不跟踪关节角；
   - 腰、腕 pitch 焊死时，tracking 用 full 版参考，anchor 用根。
2. 侧滑还有 0.33–0.37，主要受 360p 视频里脚部估计的限制。正式数据建议按第 7 节**自己拍**（固定机位、高帧率、更高分辨率），侧滑和轨迹都会明显变好。
3. 这段视频会变焦，深度估计会跳，所以世界轨迹的速度曲线有尖峰（已平滑，但没有完全消除）。用固定机位、不变焦的视频就没有这个问题。
4. 压步那几段在这台机器人上本来就做不到（髋内收限位），训练时可以把这些帧剔除，或者降低权重。
