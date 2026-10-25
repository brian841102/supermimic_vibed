# nemo_v3 花式滑冰动作重定向：计划书与开发记录（MILESTONE）

> 本文件是**计划书 + 开发过程中遇到的坑 + 因此做的调整**。安装依赖和常用指令见 [README.md](README.md)。

- 日期：2026-10-24（v2：已读取 nemo_v3 模型和 booster_mjlab 的现有做法，第 2 节的数据都是实测）
- 目标机器人：nemo_v3（轮足 humanoid，**两条腿末端各一个驱动轮，没有踝关节**），全身版 `nemo_v3_2_fullbody`
- 模型来源（已拷到 `retarget/assets/nemo_v3/`，随仓库提交）：
  - `booster_mjlab_nemov3/xmls/nemov3.xml`：booster_mjlab 训练用的 MJCF，**以它为准**（关节顺序、限位、actuator）
  - `nemo_v3_2_fullbody/urdf/train_v2.urdf` 等 6 个 URDF 变体（见 2.3 节）

---

## 当前优选：roller32 滑冰 = exp6b（2026-10-25）

| 项目 | 内容 |
|---|---|
| **优选的训练** | `logs/nemov3_roller32/2026-10-25_16-16-18_exp6b_elbow_limit_seed1`，用第 6000 轮的 checkpoint（`ckpt_006000.pt`） |
| 训练动作 | `retarget/data/supermimic_motions/roller32/roller32.csv`（原目录名 `nemov3`，2026-10-26 改名；来自 `roller40_hw`：权重 20、单轮平衡 + 离地余量 4.5 cm、人脚判定单轮时段、修复左右腿互换、手肘不往后折） |
| 奖励 / 模型 | 轮心高度奖励（两轮平均、σ = 3 cm）加单边线性惩罚（-10 / m）；仿真里手肘范围 [−1.57, +0.1] rad；轮子凸包 472 个顶点 |
| 结果 | 回报 **0.902**（0/1 失败）；中段点脚 **0 次**；抬起轮高度 p10 / 中位数 3.8 / 4.6 cm；单轮时段 ZMP 偏离中位数 0.4–0.7 cm；HOME 静止时手肘 −1.1° / −1.1°；根部角加速度 4.2（参考 2.7）、垂直加速度 0.97（参考 0.53） |
| 为什么是它 | exp5（回报 0.924，点脚 3 次）HOME 静止时手肘往后折 7–10°，不符合实体机"手肘略向后弓、前臂朝前"的预热；exp6（同设定、种子 0）点脚 9 次、抖动大，换种子（exp6b）后点脚降到 0 次，说明那是单次训练的运气。exp6b 是目前点脚最少、抖动最小、手肘修正后的版本 |
| 已知的缺点 | 根部水平误差平均 0.21 m、最大 0.39 m，比 exp5（0.11 / 0.25 m）大，策略的根部位置慢慢飘开，不影响奖励（参考按机器人当下位置重新锚定）；回报比 exp5 低约 0.02，在种子起伏范围内 |
| 怎么看 | 见 [README.md](README.md) §3⑤；详情见第 19、20 节（20.10 是 exp5 / exp6 / exp6b 的对比） |

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

- ✅ 已从内部机器拷贝 `nemo_v3_2_fullbody/`（171 MB）和 booster_mjlab 的 `robots/nemov3/`（85 MB）到 `retarget/assets/nemo_v3/`（后来改为随仓库提交，见 2.4 节）；
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

---

## 15. 第二轮：换简单动作 + 轮子朝向平顺化（2026-10-24）

用户反馈两点：一是第 14 节那段视频里有压步（两腿交叉），太复杂；二是**轮子朝向转得太突然，真机做不到**。

### 15.1 换成简单动作

- 来源：同一合集的 p01「基础级」，只取「基础级步法完整示范」里**没有交叉步**的两段，中间在远处绕弧线的 43–59 秒有交叉步，跳过：
  - `basic_a`：28–41.5 秒（13.5 秒），朝镜头方向左右蹬冰、单足滑行；
  - `basic_b`：60–76 秒（16 秒），左右蹬冰后双足滑行。
- p02「左右侧蹬冰滑行」只有前 6 秒是全身，后面都是腿部特写，不能用。
- 用一条命令 `scripts/run_video_to_nemov3.sh` 跑完这两段，这个脚本也是第一次完整跑通。

### 15.2 问题与调整（按发现顺序）

| # | 问题 | 量化 | 原因 | 调整 |
|---|---|---|---|---|
| 1 | 轮子朝向转得太突然 | 着地时朝向转速最大 21.8 rad/s；髋 yaw 关节速度最大 18 rad/s（URDF 上限 10） | 速度方向反转时朝向会翻 180°；低速时速度方向很杂；目标没有平滑 | 朝向改用角度处理：解缠绕 → 高斯平滑（σ = 5 帧）→ **转向速度限制 3 rad/s**（前后各扫一遍再平均，避免滞后）。IK 之后，腿部关节按 URDF 限速（10 rad/s），**髋 yaw 限速 5 rad/s**。朝向点的 IK 权重 20 → 50 |
| 2 | 新片段里髋 yaw 打到 ±135° | 12–35% 的帧顶到限位 | 正负号参考上一帧，开头选错就一直错，轮子朝向身体后方 | 正负号改为**参考骨盆的正面方向**（和脚尖方向的中位差只有 9°）；轮子相对骨盆的偏转限制在 ±75° 以内 |
| 3 | 髋 roll 顶在内收下限 | 35–72% | 缩放后人的两脚几乎落在身体正下方，约 4 cm；机器人轮心在零位时就离中线 8.6 cm，髋只能内收 10° | 踝 / 脚尖目标沿骨盆左右方向向外移，膝移一半，补上差距 → 降到 2–8% |
| 4 | 膝（calf）不是一直伸直就是弯到底 | 0 rad 占 58–100%，或 1.57 rad 占 12–21% | (a) **骨盆到髋的高度差对不上**：机器人髋 pitch 关节在根下方 9.7 cm，缩放后人的髋只在骨盆下方约 3.4 cm，腿被吃掉约 6 cm；(b) **直腿奇异点**：GMR 从直腿零位开始求解，膝关节在直腿时没有竖直方向的梯度；(c) 朝向点距离不一致（人脚尖缩放后约 6 cm，模型里是 12 cm）把轮子往前拉；(d) 朝向点固定在小腿坐标系，小腿一倾斜它就上下移动，IK 为了对齐它会让小腿保持竖直 | (a) 腿部目标沿骨盆竖直轴往下补上这段差（改成按髋到脚的长度对应）；(b) IK 从 HOME 蹲姿开始，并且 **IK 内部的 calf 下限设为 0.15 rad**（导出时还是真实限位）；(c) 朝向目标距离改成 `0.12 / s_leg`，缩放后正好 12 cm，只表示方向；(d) 每帧把朝向目标的高度设成上一帧朝向点相对轮心的高度差，只约束水平朝向；`--leg_reach` 默认 0.95 |

### 15.3 结果（full 版；hw 版接近）

| 指标 | basic_a | basic_b |
|---|---|---|
| 着地时轮子朝向转速 p99 / 最大 | 3.1–3.3 / 3.2–3.4 rad/s | 2.9–3.0 / 3.0–3.5 rad/s |
| 髋 yaw 关节速度最大 | 4.2 rad/s | 4.1 rad/s |
| 侧滑（运动时平均，左 / 右） | 0.31 / 0.31 | 0.15 / 0.12 |
| 髋 roll 顶到内收限位 | 约 8% | 约 8% |
| 膝 | 不再出现弯到底；多数帧接近伸直（人本来就站直滑行），IK 保留 0.15 rad 余量 | 同左 |

和第 14 节相比，着地时的朝向转速最大值从 21.8 降到约 3.4 rad/s，髋 yaw 不再超过硬件上限。
basic_b 的侧滑降到 0.12–0.15，因为这段的人体估计和相机轨迹更稳。

### 15.4 遗留

- 膝的姿态主要跟着人站直。如果下游希望机器人以 HOME 那样的蹲姿滑行（现有 AMP 数据就是蹲姿），可以调小 `--leg_reach`。但降到 0.9 时又会偶尔出现弯到底，要配合更好的 IK 正则（例如 mink 的 PostureTask）再调。
- basic_a 的侧滑（0.31）比 basic_b 高：这段人一直朝镜头滑过来，深度变化大，靠检测框估计的深度不准。

---

## 16. 第三轮：有上半身表现、又简单的轮滑动作（2026-10-24）

用户反馈：第 15 节的动作都是双手平举，想要上半身有表现、脚不交叉、在限位内；试过的舞蹈太剧烈；**要轮滑**。

### 16.1 选片过程

| 候选 | 结果 |
|---|---|
| B 站舞蹈室视频（DM 舞室 Into You） | 单人，但离镜头太近，脚会出框，还有弯腰到地面的动作（腰只能前倾 30°）；另一段有三个人 → 不用 |
| B 站 EVERGLOW 镜面教学（0.5 倍速，固定机位） | 画面质量很好：用 `-s` 静止相机模式，2 倍速还原后截 27 秒（`dance27`）。但用户觉得**太剧烈**，放弃 |
| B 站轮滑教学（LORENZO GUSLANDI 绕桩） | 单轮花式动作，镜头切换多 → 不用 |
| B 站站内搜索 | API 不登录返回 412。**不绕过 B 站的反爬机制**（被拦下后已停止，匿名 cookie 也删了） |
| **Wikimedia Commons**（公开 API，CC 授权） | `EVD-patinajeartistico-007.ogv`「花样轮滑基础图形讲解」，CC BY-SA 2.5 co：四轮鞋、户外、单人、动作很慢，手臂摆出不同姿势 → **采用** |

最终片段：`roller40` = 原片 10–50 秒（40 秒）。原片只有 320×224，先用 lanczos 放大到 640×448、30 fps 再处理。镜头是跟拍，所以走动态相机流程（`run_video_to_nemov3.sh`）。

### 16.2 这一轮发现的问题与调整

| 问题 | 原因 | 调整 |
|---|---|---|
| 舞蹈片段里左臂一直顶在限位：肘 +1.57 占 21–26%，肩 yaw +2.79 占 20% | 左臂从第 9 秒起掉进了"肩 yaw 转 180° + 肘反弯"的**等价翻转解**（左肘中位数 +0.63，右肘 −0.49），之后 IK 每帧从上一帧接着算，就一直卡在里面 | IK 内部把两肘的上限设为 `ELBOW_MAX_IK = 0.2`（正常弯曲方向是负值），导出时仍用真实限位 → 左肘中位数变成 −0.55 |
| 修正看起来"没生效" | 新加的循环变量用了 `name`，覆盖了片段名，输出被写成 `joint_rElbow_*`，原文件没被覆盖 | 循环变量改名为 `elbow`，误生成的文件已删除 |
| 固定机位的舞蹈视频怎么处理 | 跟滑冰不同：脚踩地时本来就该钉住 | 用 GVHMR `-s`、**保留**它的静止关节后处理，跳过相机轨迹重建，直接 `gvhmr_to_nemov3.py` |

### 16.3 roller40 结果

| 指标 | full | hw |
|---|---|---|
| 根速度（机器人，平均） | 0.31 m/s | 0.31 m/s |
| 侧滑（运动时平均，左 / 右） | 0.28 / 0.27 | 0.29 / 0.28 |
| 着地时轮子朝向转速 p99 | 3.1 rad/s | 3.1–3.7 rad/s |
| 顶到限位 > 3% 的关节 | 腰 pitch 15%、腰 roll 10–16%（人后仰、侧倾）、左肘弯到底 8% | hip roll 内收 10–14%、左肘 12%、右肘 4% |

画面：手臂一前一后、交替伸展，单足滑行，没有交叉腿。

---

## 17. SuperMimic 适配 nemov3 并训练（2026-10-24）

### 17.1 训练动作

`retarget/data/supermimic_motions/roller32/roller32.csv`（当时叫 `supermimic_motions/nemov3/`，后来改名；生成脚本 `scripts/make_supermimic_clip.py`）：

| 段 | 时长 | 内容 |
|---|---|---|
| HOME 静止 | 1 s | booster_mjlab `HOME_KEYFRAME`（AMP 的默认姿势；根高 0.50，朝向用片段第一帧的 yaw） |
| 过渡 | 1 s | 余弦插值到片段第一帧，关节最大 1.5 rad/s |
| 动作 | 30.6 s | roller40 的 hw 版，截到 **30.57 s**：30–34 s 里最静止的一帧（根速度 0.018 m/s）；32–34 s 又开始动，关节速度 11–28 rad/s |
| 回 HOME | 1 s | 余弦插值回 HOME（位置和朝向停在终点），关节最大 2.1 rad/s |
| HOME 静止 | 1 s | |

共 34.6 s，30 fps。关节按 SuperMimic 的 nemov3 顺序排列（= Isaac Lab 的 articulation 顺序，和 MJCF 声明顺序不同）。

### 17.2 代码适配（G1 和 nemov3 共用一套代码）

- `supermimic/robot_spec.py`：机型规格，用 `SUPERMIMIC_ROBOT=g1|nemov3` 或 `--robot` 选择，默认 g1。内容包括关节顺序、跟踪的 body、MJCF、默认姿势、动作缩放、速度控制关节、CoM 随机化的 body 和范围、终止高度、尺寸缩放。
- `supermimic/robots/nemov3.py`：nemov3 硬件版（20 关节）的 Isaac Lab 配置。增益用实机部署值（腿 50 / 0.5，肩 pitch/roll 18 / 0.8，其余臂关节 1.5 / 0.1）；力矩和速度上限取自 URDF。**轮子是速度控制**（只有阻尼的隐式驱动，kv 0.5，6 Nm）。
- `kinematics.py`：只保留被跟踪 body 的祖先链；支持焊死的固定 body；连续关节（轮子）的限位设成 ±1e4。旧名 `G1Kinematics` 保留成别名。
- `mdp.py`、`motion.py`、`buffer.py`、`env.py`、`train.py`、`evaluate.py`：维度改为从机型规格推导（状态 13 + 2J、观测 3J + 12、ξ = J + 4、目标每帧 15 + 2J）。轮子的处理：
  - 动作是速度目标（10 rad/s 每单位）；
  - **轮角不进入观测、世界模型输入、目标和 critic**（`POS_MASK`），只用轮速；
  - 关节偏置随机化不作用在轮子上。
- nemov3 的奖励 / 终止参数：位置和线速度奖励的宽度 × 0.5（机器人约 G1 的一半大）；终止高度 0.15 m；CoM 随机化范围减半。
- 仿真资产：`scripts/make_sim_mjcf.py` 生成 `assets/nemo_v3/sim/nemov3_sim_hw.xml`（焊死 12 个关节，**删掉 MJCF 自带的 actuator**，否则会和 Isaac Lab 的 PD 叠加），再用 `tools/mjcf_to_usd.py <xml> <dir>` 转成 USD。`mjcf_to_usd.py` 改成可以传参数，不带参数时仍然是转 G1。
- `tools/test_env.py`、`tools/viser_play.py` 改为两种机型通用（viser 按关节名称填 qpos）。

**检查**：
- **G1 回归**：`tools/test_env.py` 的输出和改动前完全一致（FK 6.497e-7、各帧回报相同）。
- **nemov3**：FK 位置误差 6e-6、速度误差 4–7e-5（float32 精度）；状态写入、读回一致。
- **冒烟测试**：256 个环境、600 轮跑通。

### 17.3 训练（`logs/nemov3_roller32`，单张 RTX 4090，4096 个环境，其余超参数和 G1 相同）

| iteration | 训练时间 | 评估回报 | 失败 |
|---|---|---|---|
| 1000 | 0.07 h | 0.717 | 0/1 |
| 2000 | 0.18 h | 0.848 | 0/1 |
| 3000 | 0.29 h | 0.883 | 0/1 |
| **4000** | **0.39 h** | **0.902** | **0/1** |
| 5000 | 0.49 h | 0.849 | 0/1 |
| 6000 | 0.58 h | 0.858 | 0/1 |
| 7000 | 0.69 h | 0.877 | 0/1 |

- 速度 2.86 it/s，显存约 15 GB。训练回合回报从第 3000 轮到第 6000 轮只从 0.786 升到 0.807，终止率接近 0。
- **饱和判定**：连续 3 次评估（5000、6000、7000 轮）都没超过最好值 + 0.005，在第 7200 轮停止。最好的是 `ckpt_004000`。
- ckpt 4000 的分段检查：

| 段 | 根速度 | 关节速度 p95 | 根高度误差 |
|---|---|---|---|
| 开头 HOME 静止 | 0.05 m/s | 1.7 rad/s | 0.1 cm |
| 动作 | 0.27 m/s | — | 0.7 cm |
| 结尾 HOME 静止 | **0.006 m/s** | **0.11 rad/s** | 0.0 cm |

  结尾完全静止；开头从参考状态重置后，有一点调整的晃动。

### 17.4 遗留

- 评估只有 1 段动作、跑 1 次，单个 checkpoint 之间的波动有 0.85–0.90。要更稳的判断，可以对每个 checkpoint 用不同摩擦或随机种子多评估几次。
- 开头静止段有轻微晃动（关节速度 p95 1.7 rad/s），可能是手臂的 PD 太弱（kp 1.5）加上刚重置。可以考虑在静止段加大动作变化率的惩罚。
- 现在是"hw 机器人跟 hw 参考"。"焊死的机器人跟 full 参考"需要对参考动作另外用 full 模型做 FK，还没做。

---

## 18. 机器人几乎没有位移：原因与修正（2026-10-24）

用户反馈：姿势差不多，但机器人位置没有明显移动；视频里的人滑出去的位移很大。

### 18.1 逐段定位

| 环节 | 路程（截到 30.57 s） | 活动范围 |
|---|---|---|
| GVHMR 原始世界轨迹 | 3.4 m | 1.1 × 0.4 m |
| 相机重建（固定焦距，第 14 节的方法） | 21 m | 3.7 × 1.8 m |
| 重定向参考（× 0.41） | 8.8 m | 1.5 × 0.7 m |
| 训练出的策略（ckpt 4000） | 8.4 m | 1.4 × 0.8 m，和参考平均差 0.10 m |

**训练没有问题**：策略几乎完全跟上了参考。位移是在人体轨迹这一步丢掉的。

### 18.2 原因：视频一直在变焦

- 用背景的相似变换累计出来的缩放在 **×0.58 到 ×1.43** 之间来回变：摄影师一直推拉镜头，让人在画面里保持 200–280 px 高。
- GVHMR 假设焦距固定，从人的大小估深度，所以深度一直被估在约 5 m（4.7–6.6 m）；朝向和远离镜头的位移、摇镜头时的弧线半径都被压掉了。
- 另外检查过焦距本身：用纯旋转时的单应矩阵做自标定（`tools/estimate_focal.py`），得到 658 px、视角 52°，和 GVHMR 默认的 781 px（44.5°）差不多。所以**不是长焦的问题，是变焦的问题**。

### 18.3 修正：考虑变焦的相机轨迹（`scripts/gvhmr_camera_traj.py --video`，一键脚本已默认打开）

1. 先遮掉人，再用 SIFT 估计相邻帧背景的相似变换，得到每帧的累计缩放 Z(t)、水平和竖直平移。
2. **深度 z = z_GVHMR × Z / median(Z)**。人在相机坐标系里的 x / y 偏移和焦距无关，不用改。
3. **相机转角**：背景平移 ÷ 当时的焦距 f(t) 逐帧累加，取代 GVHMR 的 R_c2w。符号推导：背景往左移 = 相机向右转，yaw 为正；背景往下移 = 抬头。
4. **身体的世界朝向**也用同一个相机旋转重算（`R_c2w_新 · R_incam`）。只换位置不换朝向，会让"面朝方向"和"移动方向"对不上。

**验证**：用一个独立的指标，滑行时身体朝向和移动方向应该一致。

| 轨迹 | 移动帧比例 | 朝向和移动方向对齐 |cos| | 路程 / 40 s | 活动范围 |
|---|---|---|---|---|
| GVHMR 原始 | 6% | 0.93 | 4 m | 1.2 × 0.5 m |
| 固定焦距 | 83% | 0.77 | 29 m | 6.3 × 4.0 m |
| **考虑变焦** | **96%** | **0.97** | **53 m** | **10.6 × 5.3 m** |

重定向之后（`roller40z`）：
- **轮子侧滑从 0.27–0.29 降到 0.05–0.07**：轮子朝向终于和移动方向一致；
- 机器人路程从 8.8 m 变成 **16.8 m**，活动范围 4.4 × 2.2 m；
- 平均速度 0.56 m/s，p95 0.82 m/s。

### 18.4 截断点的速度

现在看得出来，30.57 s 时人虽然是站姿，但**还在以 0.51 m/s 滑行**。原来的过渡段只插值姿势，根部会在一帧之内从 0.5 m/s 停到 0。

`make_supermimic_clip.py` 改成：过渡段的根部从片段终点的速度**匀减速到 0**（开头则从 0 匀加速到起始速度），姿势照旧用余弦插值回 HOME。新的训练动作是 `data/supermimic_motions/nemov3_z/roller32z.csv`，重新训练写在 `logs/nemov3_roller32z`。

### 18.5 13–16 s 的突然转动：人体朝向前后翻转（用户检视时发现）

- **现象**：13.3 s 附近，机器人根部的 yaw 以 238°/s 转过去，再以 −130°/s 转回来。
- **分解**：同一时间相机几乎没在转（2–9°/s）；但 GVHMR 在**相机坐标系里估出的人体朝向**出现 −312°/s、+180°/s 的尖峰，而视频里她只是从正面侧转约 45° 并伸腿滑行。这是低分辨率下单目估计常见的**前后翻转**，GVHMR 叠加的人体网格在 13.5 s 也明显没对上。
- **修正**：`gvhmr_camera_traj.py` 新增 `repair_orientation_flips`，默认开启（`--no_flip_repair` 可关闭）：
  1. 在相机坐标系里，把人体朝向和它的滑动中位数比较（21 帧窗口），偏离超过 35° 的帧判定为异常，前后再各多标 3 帧；
  2. 异常段的根部朝向和每个身体关节，都用前后正常帧做 slerp 补上。

  全片只找到一段：13.40–13.67 s（最大偏离 66°）。修正后 13.5 s 的根部 yaw 转速是 89°/s。
- **剩下的快速转向**：4.5 s、25.4 s 约 130°/s，持续约 0.25 s、转 30°。人体在世界坐标和相机坐标里都一致，判断是真实的快速转身，保留。33.0–33.7 s 有明显抖动，但在截断点 30.57 s 之后，不在训练动作里。

### 18.6 用修正后的动作重新训练（`logs/nemov3_roller32z`）

| 迭代 | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| 归一化回报 | 0.695 | **0.896** | 0.886 | 0.882 | 0.861 |
| 失败段数 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |

- 连续 3 次评估都没有超过「最好成绩 + 0.005」，判定为饱和，在第 5000 轮停止（约 0.5 h）。最好的 checkpoint 是第 2000 轮，比旧动作训练的最好成绩（0.902，第 4000 轮）收敛得更快。
- **位移已经跟上**（`traj_best.npz`，34.6 s）：策略根部的水平路程是 17.7 m（参考 17.2 m），范围 4.6 × 2.2 m（参考 4.4 × 2.2 m）；水平误差平均 0.09 m、p95 0.22 m、最大 0.23 m，结束时 0.09 m。
- 训练只写 CSV。要用 TensorBoard 看曲线，用 `tools/csv_to_tensorboard.py` 转出 `logs/<run>/tb/`（见 README）。

## 19. 抖动与抬脚消失（用户检视训练结果时提出，2026-10-24）

### 19.1 抖动：Newton 把轮子碰撞体简化成多边形
- **现象**：`logs/nemov3_roller32z` 最好的 checkpoint 里，髋 roll、根部 roll 和根部高度有约 10 Hz 的振荡（髋 roll 4–24 Hz 频段 0.47° rms，参考只有 0.07°），手臂正常；静止时几乎没有，轮子转起来才出现。
- **原因**：Isaac Lab / Newton 从 USD 读入网格碰撞体时，默认凸包最多 64 个顶点（`newton.Mesh.MAX_HULL_VERTICES`）。轮子因此成了 22–24 边的不规则多边形，滚动半径起伏 4 mm（右）/ 12 mm（左），每转一圈颠二十几下。booster_mjlab 直接用 MuJoCo 编译 MJCF，凸包不设上限（472 个顶点），所以没有这个问题。
- **先试了圆柱，后来放弃**：一开始把碰撞体换成圆柱（半径 0.075 m，宽 2.64 cm）。但真实胎面是圆弧的，圆柱和它差太多，侧倾时的支撑完全不同。用圆柱训练的 run 跑到第 2000 轮（回报 0.843）就停了，存档为 `logs/nemov3_roller32zb_cylinder`。
- **其他做法的调查**：
  - arXiv 2607.10815（轮滑 humanoid AMP，PhysX）：用沿轮宽排列的 9 片窄圆柱（切圆片）。原因是 PhysX 的网格 cooking 会让接触法线跳动；全球形太宽，侧倾超过约 21° 就会给出不真实的侧向支撑。
  - arXiv 2606.31807（Booster T1 被动轮）：训练和验证用球形 / 椭球形轮子。
  - flamingo_rl_train（PhysX，上过实体机）和 booster_mjlab（MuJoCo）：都直接用**轮子网格的凸包**。
- **最后的修正（照 flamingo / booster 的做法）**：保留轮子网格碰撞体，只把 Newton 的凸包上限调高。
  - `tools/mjcf_to_usd.py` 新增 `--hull_vertices '.*_wheel_collision=1024'`，在 instance prototype 的碰撞 prim 上写入 `newton:maxHullVertices`（同时写 PhysX 的 `hullVertexLimit`，上限 255）。
  - 写在外层 stage 的 over 没有用：碰撞 prim 在 instanceable prim 里面，外层的意见会被忽略。
  - 结果：凸包 472 个顶点，滚动半径起伏 0.13 mm，和 MuJoCo 原生编译的一致；训练速度不变（约 6.8 it/s）。
  - `make_sim_mjcf.py` 改回原来的网格碰撞体，生成的 MJCF 和改之前逐字相同。

### 19.2 抬脚消失：参考动作的单轮时段物理上站不住
- **现象**：参考里左轮离地超过 3 cm 合计约 11.7 s、右轮约 5.1 s，策略始终两轮着地。
- **原因 1：参考不可行**：单轮时段 ZMP 偏离支撑轮的横向距离中位数 3.4–4.8 cm、p90 8–11 cm，而胎面是圆弧的，接地近似一个点。重定向时左髋 roll 卡在 −10° 限位、小腿卡在 IK 下限，脚是靠整个身体侧倾约 20° 才抬起来的，重心因此落在支撑轮外面。
- **原因 2：奖励不在乎**：5 cm 的抬脚误差平均到 11 个身体上，位置奖励只少不到 1%。
- **修正 1（重定向）**：`gvhmr_to_nemov3.py` 新增单轮平衡（默认开启，`--no_balance` 关闭）：
  1. 找出单轮时段（≥ 0.2 s），前后各留 0.4 s 过渡；
  2. 用稀疏最小二乘一次解出整段的水平重心轨迹：ZMP = c − (z/g)·c'' 横向落在支撑轮上，其余时候贴近原轨迹，并保持平滑；
  3. mink IK 去追这条重心轨迹：支撑轮不动，抬起的轮和躯干朝向软约束，手臂冻结。

  逐次"按 ZMP 误差平移重心"的迭代会发散：c'' 项会放大约 0.7 Hz 以上的成分。实测结果：
  - ZMP 横向偏离：中位数 5.2 → 0.6 cm，p90 11.9 → 0.9 cm；
  - 抬脚：高度和时长都保住了；
  - 改动幅度：骨盆平移中位数 3.4 cm（p95 10.5 cm），手臂不变。

  新动作是 `data/retarget_out/roller40zb_hw.*` → `data/supermimic_motions/nemov3_zb/roller32zb.csv`。
- **修正 2（奖励）**：`RobotSpec.lift_points` 列出要追踪高度的点（nemov3 是两个轮心，在小腿坐标系里的偏移），奖励多一项 `exp(-mean(Δz²)/0.03²)`，权重 1。单步最大奖励从 0.10 变成 0.12，所以 nemov3 的归一化回报和之前的 run **不能直接比较**。G1 没有 `lift_points`，奖励和以前完全一样。
- **重新训练**：`logs/nemov3_roller32zb`（网格轮子，每 500 轮存一个 checkpoint）。

### 19.3 结果（`logs/nemov3_roller32zb`：网格轮子凸包 472 顶点 + 单轮平衡动作 + 轮心高度奖励）

| 迭代 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 归一化回报 | 0.651 | 0.812 | 0.472 | 0.859 | 0.890 | 0.900 | 0.907 | 0.910 | **0.920** | 0.906 | 0.887 | 0.901 |
| 失败 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- 第 5000 轮之后，连续 3 次评估都没有超过「最好成绩 + 0.005」，判定为饱和，在第 6500 轮停止（约 0.65 h）。最好的 checkpoint 是第 5000 轮。第 2000 轮摔过一次，之后都没再摔。
- **抬脚恢复**（第 5000 轮的轨迹）：
  - 离地超过 4 cm 的帧数：左 406 / 右 171（参考 370 / 177）；
  - 最高抬起：6.7 / 9.3 cm（参考 7.0 / 9.3 cm）；
  - 单轮时段策略自己的 ZMP 横向偏离：中位数 0.4–0.8 cm，确实是站在一个轮子上滑行。
- **抖动大幅减少**（括号里依次是上一版 `logs/nemov3_roller32z` → 这次 → 参考）：
  - 髋 roll 加速度均方根：24–25 → 8–9 → 5–6 rad/s²；
  - 根部角加速度：11.3 → 4.7 → 2.9；
  - 根部垂直加速度：2.32 → 0.92 → 0.53 m/s²。

  仍比参考略高。

### 19.4 轮子朝向权重改回 20，正式版本改用干净的名字

- **背景**：`configs/smplx_to_nemo_v3.json` 里轮子朝向标记点的 IK 位置权重，在挑选片段时从 20 调到了 50，但一直没有单独做过对照（`roller40zb` 和 `nemov3_roller32zb` 用的都是 50）。
- **对照**：只改这个权重、其余流程完全相同，比较 0–30.57 s：

| | 权重 20 | 权重 50 |
|---|---|---|
| 轮子朝向和行进方向的夹角，中位数 / p90 | 3.9° / 12° | 2.7° / 8° |
| 侧滑比例 | 0.09 | 0.08 |
| 膝盖弯曲中位数（左 / 右） | 0.37 / 0.62 rad | 0.15 / 0.15 rad |
| 膝盖贴在 IK 下限的比例 | 34% / 13% | 68% / 71% |

  朝向标记点固定在小腿上，权重越高，IK 越倾向用伸直小腿来满足它。用户在 viser 里比较后，选择**权重 20**：膝盖弯曲更像滑冰时蹲着的姿势，朝向误差稍大。配置因此回到已提交的版本（20）。
- **改名**：正式版本不再带 z / zb 这类后缀，换成以下名字：
  - 重定向结果：`data/retarget_out/roller40_{hw,full}.*`；
  - 训练片段：`data/supermimic_motions/roller32/roller32.csv`（当时叫 `nemov3/`，是 `robots/nemov3.py` 的默认动作目录；后来改名为 `roller32/`，默认改成 `nemov3_multi`）；
  - 训练 run：`logs/nemov3_roller32`（权重 20，3000 轮之前每 1000 轮评估、之后每 500 轮）。
- **旧产物归档**：搬走时保留原名，本文件前面提到的名字都还能在这里找到：
  - 训练 run：`logs/archive/`（`nemov3_roller32`、`nemov3_roller32z`、`nemov3_roller32z_prefix_unrepaired`、`nemov3_roller32zb`、`nemov3_roller32zb_cylinder`）；
  - 重定向结果：`data/archive/retarget_out/`；
  - 训练片段：`data/archive/supermimic_motions/`；
  - 相机轨迹：`data/archive/gvhmr_cam/`、`data/archive/gvhmr_camzoom/`。

  从头重跑相机轨迹和重定向，结果和归档版本逐字相同：权重 20 的 hw 版等于 `roller40zb_h20_hw`。

### 19.5 正式版结果（`logs/nemov3_roller32`：权重 20 + 单轮平衡 + 网格轮子 + 轮心高度奖励）

| 迭代 | 1000 | 2000 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 |
|---|---|---|---|---|---|---|---|---|
| 归一化回报 | 0.723 | 0.862 | 0.901 | 0.907 | **0.944** | 0.897 | 0.909 | 0.893 |
| 失败 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- 第 4000 轮之后，连续 3 次评估都没有超过「最好成绩 + 0.005」，判定为饱和，在第 5500 轮停止（约 0.53 h）。最好的 checkpoint 是第 4000 轮（0.944）。
- 和权重 50 的 run（`logs/archive/nemov3_roller32zb`，最好 0.920）比：两者奖励的算法相同，这次收敛更快、最好成绩更高，而且全程没有摔倒。
- **抬脚**：权重 20 的参考动作本身抬得比较低（离地 > 4 cm 的帧数：左 83 / 右 68，最高 5.0 / 6.8 cm）。策略跟得上：左 94 / 右 63 帧，最高 5.5 / 6.8 cm；单轮时段 ZMP 横向偏离中位数 0.6–1.1 cm。
- **抖动**（加速度均方根，策略 / 参考）：
  - 髋 roll：8–9 / 8–9 rad/s²；
  - 膝：14–18 / 8 rad/s²；
  - 根部角加速度：5.0 / 2.8；
  - 根部垂直加速度：1.16 / 0.51 m/s²。

  根部水平位置误差平均 0.07 m，最大 0.19 m。

### 19.6 待办（用户决定以后再处理）
- `supermimic/train.py` 的 `sim_rew` / `imag_rew` 写死除以 0.1（G1 的单步最大奖励），nemov3 加了轮心高度项之后上限变成 1.2。改成除以 `MAX_STEP_REWARD` 只影响记录，G1 数值不变；代价是新旧 nemov3 run 的曲线要先换算才能放在一起比。
- 轮心高度奖励推广成通用的末端点追踪：每个点可选"只比高度"或"比 3D 相对位置"，类似 DeepMimic 的末端执行器奖励。G1 默认不开，以保持 LAFAN1 复现。
- 消融实验：用现在的动作关掉轮心高度奖励再训练一次，确认这一项单独的贡献（目前它和单轮平衡、轮子凸包是一起改的）。

## 20. 单轮滑行时偶尔点脚（2026-10-25）

### 20.1 现象与原因
- `logs/nemov3_roller32/2026-10-24_23-10-25_exp1_heading20`（即 19.5 的正式版）在长的单轮时段中段会让抬起轮碰地约 0.1 s，再抬起来，间隔 0.5–0.9 s：左轮 6.9–10.6 s（训练片段时间轴）碰了 6 次，右轮 17.7–20.7 s 碰了 5 次。
- 那两段参考动作只抬 2–4 cm。左右轮距约 17 cm，身体侧倾约 7° 抬起轮就会碰地，余量太小。另外轮心高度奖励是两轮平均、宽度 3 cm，碰地 0.1 s 几乎不扣分，反而能借地面修正平衡。对照：权重 50（膝盖较直、抬得较高）的 run，同样的时段中段只碰了 0–1 次。

### 20.2 离地余量（第一版）与踩到的坑
- 在单轮平衡里先加一步：只动抬起那条腿的 hip pitch / roll 和膝盖（3 个关节对 3D 目标，逐帧有唯一解），把抬起轮抬到至少 4.5 cm，时段头尾各用 0.2 s 的余弦过渡；IK 的高度权重 50、水平权重 5，迭代 30 次。
- **坑 1**：一开始把抬高目标直接加进全身 IK，结果在关节限位附近跳到另一组解（支撑腿膝盖一帧多弯 0.45 rad，关节速度 13.5 rad/s，超过硬件上限 10 rad/s），所以改成单独抬腿。
- **坑 2**：IK 的水平和高度权重一样时，往后伸的腿被 hip roll 限位卡住，只抬到 3.4 cm；把高度设为优先后才到位。

### 20.3 假抬脚：用人的脚来判定单轮时段（用户检视时发现 20 s 前后有换脚，视频里没有）
- 单轮时段原本按重定向后的轮子高度（> 2 cm）判定。GVHMR 的脚有 2–4 cm 的误差，脚跟翘起（脚尖还着地）也会抬高脚踝，这些都被当成抬脚，离地余量再把它们抬到 4.5 cm。
- **修正 1**：`human_lift_phases` 改用人的脚判定：取脚踝和脚尖中较低的点，看它比另一只脚高多少。高过 5 cm 算抬起，降到 3.5 cm 以下才算落地；先用 5 帧中值滤波，而且至少持续 0.5 s。
- **修正 2**：`ground_unlifted_feet` 在 IK 之前，把没抬的脚的脚踝、脚尖目标压到另一只脚的高度，时段头尾平滑过渡，由 GMR 的全身 IK 让轮子贴地。
- **坑 3**：一开始是在重定向之后才把轮子往下压。膝盖已经接近伸直、伸不下去，IK 只能大幅甩动 hip pitch，关节速度冲到 44.7 rad/s。所以改成在 IK 之前修正人的目标。

### 20.4 左右腿互换（19 s 的交替脚）
- 视频里 18.3–21 s 一直是同一个剪刀步，但 GVHMR 估出的两脚前后距离在 18.6–19.2 s 从 −28 cm 翻到 +38 cm 又翻回来：左右腿认反了约 0.5 s。之前看到的 12 cm"抬脚尖峰"也是这次互换造成的。和 18.5 的人体朝向翻转是同一类单目估计误差。
- **修正**：`gvhmr_camera_traj.py` 新增 `repair_leg_swaps`，默认开启，`--no_leg_swap_repair` 关闭。用 SMPL-X 算出骨盆坐标系里两脚踝的前后距离，偏离 31 帧滑动中位数超过 20 cm 的片段（前后各多标 3 帧），腿部 8 个关节用前后正常帧 slerp 补上。
- **保护条件**：要同时满足才修，避免误删真实的交错：
  - 不超过 0.7 s；
  - 回到原位：片段前后的前后距离相差小于 20 cm；
  - 孤立：1.5 s 内没有别的候选。连续出现的交替较可能是真实的蹬地节奏，会整串保留并印出提醒。
- roller40 修了 6 段：3.8、9.6、18.6、24.8、29.1、38.8 s。每一段都放大脚部截图核对过，视频里都没有互换。
- 修好左右腿后，右脚在 19–21 s 有 3–4 cm 的 GVHMR 误差，把 16.2–18.6 s 的真抬脚延长到 20.9 s，所以"落地"门槛从 2.5 cm 提高到 3.5 cm。

### 20.5 结果（`logs/nemov3_roller32/2026-10-25_01-07-29_exp2_clearance`）

| 迭代 | 1000 | 2000 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 归一化回报 | 0.716 | 0.853 | 0.873 | 0.899 | 0.872 | 0.893 | **0.927** | 0.891 | 0.892 | 0.900 |

- 全程 0/1 失败；第 6500 轮判定饱和，最好的 checkpoint 是第 5000 轮。
- **和 heading20 对照**（两者各自对照自己的参考动作，单轮时段中段的碰地，不计头尾 0.2 s）：

| | heading20 | clearance |
|---|---|---|
| 参考动作的单轮时间 | 15.5 s | 20.4 s |
| 中段点脚次数（总时长） | 14 次（1.58 s） | **7 次（0.88 s）** |
| 每秒点脚次数 | 0.90 | **0.34** |
| 策略抬起轮的高度 p10 / 中位数 | 0.5 / 2.9 cm | **2.6 / 4.4 cm** |
| 策略真正单轮（抬起 > 3 cm）的时间，左 / 右 | 3.8 / 2.4 s | **12.3 / 3.9 s** |
| 单轮时段 ZMP 偏离中位数 | 0.6–1.1 cm | 0.6–0.7 cm |
| 膝关节加速度均方根（策略 / 参考） | 14–18 / 8 | 23–24 / 8–14 |

  根部水平误差平均 0.09 m，最大 0.26 m。
- **剩下的 7 次点脚**：7.3、10.7、15.1、16.2、23.8 s 等，单次 0.04–0.28 s。下一步可以收紧轮心高度奖励：两轮分开算，或加一项单边惩罚。

### 20.6 训练目录
- `train.py` 新增 `--exp_name` / `--run_name`：实验目录 `logs/<exp_name>/<YYYY-mm-dd_HH-MM-SS>_<run_name>/`（和 mjlab 一样），`latest` 指向最新一次；`evaluate.py`、`csv_to_tensorboard.py` 会把 `latest` 解析成实际路径。
- `args.json` 改为记录实际用到的 `motion_dir`。`docker/run_container.sh` 会把主机时区传进容器，否则时间戳是 UTC。
- 原本平铺的 `logs/nemov3_roller32` 搬进 `logs/nemov3_roller32/2026-10-24_23-10-25_exp1_heading20/`。

### 20.7 方案 B（轮心高度奖励两轮分开算、σ 3 → 2 cm）：失败（`logs/nemov3_roller32/2026-10-25_01-56-07_exp3_lift_perwheel_std2`）
- 回报：1000 轮 0.696，2000 轮 **0.872**，3000 轮 0.865，3500 轮 0.860，4000 轮 0.848，在第 4000 轮判定饱和。训练端的 `ep_ret_norm` 和 clearance 几乎是同一条曲线。
- **策略反而不抬脚了**：
  - 第 2000 轮从头到尾两轮着地；
  - 第 4000 轮只有右轮单轮滑行 3.3 s、左轮 0.3 s，clearance 是 12.3 / 3.9 s；
  - 单轮时段里"该离地却贴地"的总时间：13–17 s，clearance 只有 0.9 s。
- **原因**：σ = 2 cm、两轮分开算时，抬起轮贴在地上、参考要求抬 4.5 cm，这一项只剩 exp(−(4.5/2)²) ≈ 0.007，落在指数平坦的尾巴上。SuperMimic 是靠奖励对动作的梯度在世界模型里学的，梯度几乎为零，策略从"两轮着地"这个起点学不会离地。原本两轮平均、σ = 3 cm 时，同样的情况还有 ≈ 0.32。
- **教训**：只收紧追踪奖励的宽度，离目标远时会失去学习信号。要加强"碰地扣分"，同时得保留远距离的梯度，例如窄、宽两项并用，或改用线性惩罚。

### 20.8 方案 A（两轮分开算、σ 维持 3 cm）：同样失败（`logs/nemov3_roller32/2026-10-25_02-30-16_exp4_lift_perwheel_std3`）
- 回报：1000 轮 0.674，2000 轮 0.876，3000 轮 0.855。第 2000、3000 轮录下的轨迹**都完全没有抬脚**；而 clearance 在第 2000 轮已经右轮单轮 1.8 s，第 3000 轮左 6.0 s、右 3.3 s。所以在第 3000 轮提前停止。
- **原因**：原本的写法 `exp(-mean(Δz²)/σ²)` 是两轮先平均，只有一个轮子有误差时等于 `exp(-Δz²/(2σ²))`，单轮的有效宽度是 σ√2 ≈ 4.2 cm。"分开算、σ = 3 cm"其实比原本更窄。抬起轮贴地、参考要求抬 4.5 cm 时，梯度只有原本的约 1/3，而这一项的奖励反而更高（0.55 对 0.32），"不抬"被罚得更轻。
- **结论**：`mdp.py` 改回原本的写法（两轮平均、σ = 3 cm）。要加强"碰地扣分"，得用不削弱远距离梯度的做法，例如在原本的奖励之外，再加一项单边线性惩罚。

### 20.9 单边线性惩罚：有效（`logs/nemov3_roller32/2026-10-25_02-56-45_exp5_lift_floor`）
- `mdp.py` 的轮心高度奖励照旧（两轮平均、σ = 3 cm），另加一项惩罚 `w_lift_floor * Σ relu(z_ref − z)`（-10 / m）。只有参考动作里该轮比另一轮高超过 2 cm（`lift_floor_on`，参考要求离地）时才扣分。线性项在任何距离梯度都一样大，不会像 20.7、20.8 那样消失。这一项是惩罚，不计入单步最大奖励，G1 也没有这一项。
- 回报：

| 迭代 | 1000 | 2000 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|
| 归一化回报 | 0.642 | 0.833 | 0.866 | 0.902 | 0.894 | 0.920 | 0.881 | 0.901 | **0.924** |

  全程 0/1 失败。第 6000 轮只比最好成绩多 0.004，判定饱和，最好的 checkpoint 是第 6000 轮。
- **学得更快**：第 2000 轮已经单轮滑行 左 8.8 / 右 3.0 s（exp2 当时 0 / 1.8 s）；第 3000 轮的点脚就降到 7 次、0.80 s，等于 exp2 训练到最后的水平。
- **最终对比**（各自对照自己的参考动作，单轮时段中段，不计头尾 0.2 s）：

| | exp1_heading20 | exp2_clearance | **exp5_lift_floor** |
|---|---|---|---|
| 回报 | 0.944 | 0.927 | 0.924 |
| 参考动作的单轮时间 | 15.5 s | 20.4 s | 20.4 s |
| 中段点脚次数（总时长） | 14 次（1.58 s） | 7 次（0.88 s） | **3 次（0.24 s）** |
| 抬起轮高度 p10 / 中位数 | 0.5 / 2.9 cm | 2.6 / 4.4 cm | **3.6 / 4.5 cm** |
| 真正单轮滑行时间，左 / 右 | 3.8 / 2.4 s | 12.3 / 3.9 s | **13.3 / 4.5 s** |
| 单轮时段 ZMP 偏离中位数 | 0.6–1.1 cm | 0.6–0.7 cm | 0.5–0.8 cm |
| 根部垂直加速度均方根（参考 0.54） | 1.16 | 1.31 | 1.10 |
| 膝关节加速度均方根 | 14–18 | 23–24 | 20–24 |

  根部水平误差平均 0.11 m，最大 0.25 m。
- **剩下的 3 次点脚**都在右轮 15.1–18.1 s（视频时间）那段 5 s 的长单轮时段里，每次 0.06–0.10 s。

### 20.10 手肘往后折与限位（exp6、exp6b）
- **现象**：exp5 的策略在 HOME 静止段手肘往后折（左 +7°、右 +10°，整段最大 +9° / +15°）；参考动作的手肘在 HOME 是 0°（和 booster_mjlab 的 `HOME_KEYFRAME`、flamingo 的 `env.yaml` 一致），做动作时则跟得很好。
- **原因**：手肘很软（kp 1.5、kd 0.1、1.27 Nm）；奖励里没有关节角度项，只追踪手肘 / 手腕 link 的位置和朝向，前臂只有 13 cm，往后折 10° 手腕只偏约 2 cm，在 15 cm 宽的位置奖励里几乎不扣分；URDF 的手肘范围是 ±1.57 rad，允许往后折。
- **修正**：仿真里手肘范围收紧到 [−1.57, +0.1] rad（`make_sim_mjcf.py` 的 `ELBOW_RANGE`，同时重新生成 USD）；重定向的 IK 里手肘上限从 0.2 改为 0（`ELBOW_MAX_IK`）。参考动作的手肘从最大 +4° / +12° 变成 +0.3° / +0.5°；右肩 yaw 在 22–25 s 的右臂近乎伸直时，会先转约 76° 再转回来，用户在 viser 里确认可以接受。
- **硬件限位检查**：整段手臂各关节都在 URDF 范围内（肘弯到 −90°，正好是下限；左肩 yaw 最大 +134°，限位 ±160°），速度都在 URDF 速度上限的 45% 以内。这些是 URDF 的数字，实体机的机构限位以后再核对。
- **回报**：

| 迭代 | 1000 | 2000 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|
| exp5 | 0.642 | 0.833 | 0.866 | 0.902 | 0.894 | 0.920 | 0.881 | 0.901 | **0.924** |
| exp6（种子 0） | 0.650 | 0.829 | 0.876 | 0.872 | **0.907** | 0.884 | 0.890 | 0.886 | — |
| exp6b（种子 1，训练到 6000 轮） | 0.666 | 0.817 | 0.895 | 0.884 | 0.892 | 0.884 | 0.898 | 0.902 | **0.902** |

  全部 0/1 失败。exp6 照饱和规则在第 5500 轮停止，最好是第 4000 轮；exp6b 照 exp5 的轮数固定训练到第 6000 轮。
- **最终对比**（各自最好的 checkpoint，单轮时段中段的碰地，不计头尾 0.2 s）：

| | exp5（第 6000 轮） | exp6（第 4000 轮） | exp6b（第 6000 轮） |
|---|---|---|---|
| HOME 静止时手肘（左 / 右） | +7.0 / +10.1° | −0.7 / −0.4° | **−1.1 / −1.1°** |
| 整段手肘最大值 | +9.3 / +14.9° | 0.0 / +1.1° | 0.0 / +1.7° |
| 回报 | 0.924 | 0.907 | 0.902 |
| 中段点脚次数（总时长） | 3 次（0.24 s） | 9 次（0.74 s） | **0 次** |
| 抬起轮高度 p10 / 中位数 | 3.6 / 4.5 cm | 2.8 / 4.5 cm | **3.8 / 4.6 cm** |
| 单轮时段 ZMP 偏离中位数 | 0.5–0.8 cm | 0.6 cm | 0.4–0.7 cm |
| 膝关节加速度（左 / 右，参考 14 / 8） | 24 / 20 | 32 / 33 | 26 / 17 |
| 根部角加速度（参考 2.7） | 4.9 | 6.6 | **4.2** |
| 根部垂直加速度（参考 0.53） | 1.10 | 1.66 | **0.97** |
| 根部水平误差 平均 / 最大 | 0.11 / 0.25 m | 0.14 / 0.32 m | 0.21 / 0.39 m |

- **结论**：
  - 手肘限位有效：HOME 静止时手肘伸直，整段不再往后折。
  - exp6 比 exp5 多的点脚和抖动，是单次训练的随机差异：同样的设定换种子（exp6b）后，点脚降到 0 次，抖动回到 exp5 的水平以内。以后比较两个设定，至少要多一个种子。
  - exp6b 的根部水平误差明显大（平均 0.21 m）。策略和参考的根部位置会慢慢飘开，不影响奖励（奖励是以机器人当下的位置重新锚定参考），需要时可以另外看。

### 20.11 经验小结（点脚、手肘、播放器）
**训练与实验**
- **比较两个设定要多一个种子**：exp6 和 exp6b 设定完全相同，只差种子，点脚 9 次对 0 次，膝关节加速度 32 / 33 对 26 / 17，回报 0.907 对 0.902。单次训练的差异不能直接归因给设定。
- **加强某一项追踪，要保住远距离的梯度**：SuperMimic 靠奖励对动作的梯度在世界模型里学，`exp(-d²/σ²)` 离目标远时梯度几乎为零。收紧 σ 或改成两轮分开算（exp3、exp4）都让策略不抬脚；两轮先平均，单轮的有效宽度是 σ√2，分开算反而更窄。要多扣"碰地"，另加一项线性惩罚（exp5），不要动原本的指数项。
- **奖励只追踪 link 位置时，关节角度可能悄悄偏掉**：手肘（前臂 13 cm）往后折 10°，手腕只偏 2 cm，奖励几乎不扣分（exp5 在 HOME 静止时手肘 +7° / +10°）。要从仿真关节限位上限制，不要指望奖励。
- **评估中途录轨迹**：回报在训练早期差异很小，看不出策略有没有抬脚。第 2000、3000 轮各录一次轨迹，才能提早发现 exp3、exp4 这类"根本没学会"的情况。
- **中途发现的小问题**：`sim_rew` / `imag_rew` 的除数写死 0.1（G1 的单步最大奖励），nemo_v3 加了轮心高度项后上限是 1.2；`ep_ret_norm` 用的是 `MAX_STEP_REWARD`，上限是 1。尚未修改（只影响记录）。
- **run 命名**：`--exp_name` 底下并列放同一组实验，`--run_name` 写这次改了什么，编号（exp1…exp6b）只是惯例，写在 `--run_name` 里，不写进代码。

**重定向与检视**
- **用户检视是不可少的一步**：20 s 前后的假换脚、19 s 的左右腿互换，都是用户看出来，我再放大视频的脚部截图确认的。每次改动重定向后，都要放进 viser 并排比较，再训练。
- **补丁要在 IK 之前做**：事后压轮子或抬轮子，会把腿推到关节限位附近的另一组解，关节速度冲到 13–45 rad/s（硬件上限 10 rad/s）。要改轮子高度，就改人的目标，或者只动那条腿的 3 个关节。

**播放器（`tools/viser_play.py`）**
- **载入慢不是卡死**：nemo_v3 的视觉网格是 CAD 直接导出的，两份（机器人加参考影子）共约 205 万面，每次打开页面要传约 73 MB；浏览器反复断线重连会一直卡在载入。简化网格会让外观很丑，用户选择还原，请只留一个分页，等它载完。
- **速度控制**：每轮循环都把播放时间四舍五入到 0.02 s 再存回滑杆，循环快（27 ms）时，速度 0.5 到 1.0 都被量化成"每轮一格"，实测 0.5 倍速其实是 0.75 倍。改成用一个浮点数累积播放时间，只在检测到用户拖动滑杆时才改用滑杆的值，0.1–2.0 倍速实测都对。之前正常，是因为训练在后台跑，循环比较慢。
- **地面轨迹**：机器人实际轨迹（橘色）默认显示，参考轨迹（蓝色）默认不显示，两个开关分开；长度默认 5 秒，0.5–10 秒可调。

**待办**（用户决定以后再处理）
- exp6b 的根部水平误差平均 0.21 m、最大 0.39 m，比 exp5（0.11 / 0.25）大，要不要查；
- `sim_rew` / `imag_rew` 的除数改成 `MAX_STEP_REWARD`；
- 轮心高度奖励推广成通用的末端点追踪；
- 实体机上手肘的机构限位和预热姿势，等用户提供实机数据后再核对。

## 21. 第二段动作 roller_house 与两段一起训练（2026-10-25）

### 21.1 视频和选片
- 想再找几段约 30 s、不交叉、抬脚不超过 2 s 的轮滑片段。YouTube 在这台服务器上会弹人机验证，**不绕过**，改由用户在自己的电脑下载，放到服务器 `retarget/data/videos/incoming/`。
- 先去 Wikimedia Commons 找了一轮：荷兰的两段老新闻片（剪辑多、选手很小）、摩托领骑的竞速、EVD 系列其余各段（片头 / 历史 / 比赛合辑 / 绿幕教学）都不合用；只有 EVD-patinajeartistico-005 的 3–33 s 勉强可用，没有采用。
- 用户提供 `Roller skating inside the house.mp4`（608×1080 竖屏，9.7 s，一个连续镜头，单人，室内四轮鞋：滑远 → 转身 → 滑回）。只有约 10 s，蓬裙会遮腿。

### 21.2 重定向
- 流程不变（`run_video_to_nemov3.sh`）：考虑变焦的相机轨迹（只有 7° 的镜头转动）、左右腿互换修复 1 段（1.2–1.5 s）、单轮平衡。
- 整段指标：两轮左右间距最小 0.062 m（轮距 0.172 m），**没有交叉**；右轮单轮最长 1.9 s、其余 ≤ 1.3 s；根部朝向累计转了约 650°（最高 277°/s，是一个原地转身加迈步）。
- 用户澄清："脚的朝向不能剧烈变化"指的是**髋 yaw 关节**的角速度，不是根部相对世界的转速；**根部在世界坐标里转多快都可以**。所以不限制 277°/s，只检查髋 yaw：整段最大 5.3–5.5 rad/s（出现在 7.4 s 之后），截取后的片段只有 2.0 rad/s（硬件上限 10；roller40 是 3.0）。

### 21.3 训练片段
- `make_supermimic_clip.py` 新增 `--start_s`（裁掉开头）。起点 **1.1 s**：用 GVHMR 的人物框判断，从这一刻起头和脚都完整在画面内；终点 **7.4 s**：她转完身站定，根部速度 0.07 m/s、转速降到 28°/s，再往后她又迈步朝镜头走。
- 用户一开始要"前后各 2 s 的衔接"，做出来后觉得太长。roller32 的做法是 1 s 静止加 1 s 衔接（占动作加衔接的 6%），这段只有 6.3 s，2 s 的衔接占 39%（1 s 版本占 24%）。改成 **1 s**，和 roller32 一致；衔接段关节最大速度进 1.08 / 出 1.99 rad/s（roller32 是 1.22 / 2.02）。结果：1 s 静止 + 1 s 衔接 + 6.3 s 动作 + 1 s 衔接 + 1 s 静止，共 10.3 s。
- 新工具 `retarget/tools/clip_to_pkl.py`：把训练片段转成 `viser_play_motion.py` 能播的 pkl（含 HOME 和衔接段）。

### 21.4 两段一起训练（`logs/nemov3_multi/2026-10-25_22-58-12_exp1_roller32_roller_house`）
- 动作目录 `retarget/data/supermimic_motions/nemov3_multi/`：`roller32.csv`（34.6 s）加 `roller_house.csv`（10.3 s）。SuperMimic 本来就是在目录里的多段动作上训练（G1 是 40 段），策略靠目标里的未来参考帧区分是哪一段，采样按回报分箱。设定同 exp6b（手肘限位、轮心高度奖励加单边线性惩罚、网格轮子），种子 0。
- 回报：

| 迭代 | 1000 | 2000 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 | 6500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 平均 | 0.693 | 0.860 | 0.883 | 0.886 | 0.899 | 0.907 | 0.912 | 0.912 | **0.9125** | 0.910 |

  全程 0/2 失败；第 6500 轮判定饱和，最好的 checkpoint 是第 6000 轮。第 6000 轮分段：roller32 **0.888**、roller_house **0.937**。
- 各段指标（第 6000 轮，单轮时段中段的碰地不计头尾 0.2 s）：

| | roller32 | roller_house |
|---|---|---|
| 中段点脚 | 1 次（0.08 s） | 1 次（0.14 s） |
| 抬起轮高度 p10 / 中位数 | 3.7 / 4.8 cm | 4.8 / 6.7 cm |
| 单轮时段 ZMP 偏离中位数 | 0.5–0.7 cm | 0.2 cm |
| 髋 yaw 最大速度 | 3.4 rad/s | 2.1 rad/s |
| HOME 静止时手肘（左 / 右） | +0.2 / −0.5° | +0.2 / −0.5° |
| 根部水平误差 平均 / 最大 | 0.24 / 0.37 m | 0.09 / 0.16 m |

- 对照单独训练的 roller32（exp6b：回报 0.902，点脚 0 次）：两段混合训练后 roller32 的回报 0.888，略低 0.014，在种子起伏范围内；roller32 的根部水平误差（0.24 m）比 exp6b（0.21 m）略大，仍是前面说的"位置慢慢飘开、不影响奖励"的现象。

### 21.5 动作目录改名（2026-10-26）
- 原来的 `supermimic_motions/nemov3/`（只有 roller32）改名为 `supermimic_motions/roller32/`；`robots/nemov3.py` 的默认动作目录改成 `nemov3_multi/`（roller32 + roller_house）。不带 `--motion_dir` 训练 nemov3，就是这两段一起训练；要只训练一段，用 `--motion_dir` 指向 `roller32/` 或 `roller_house/`。
- 旧 run 的 `args.json` 里记录的是当时的路径（`nemov3`、`nemov3_z`、`nemov3_zb`、`nemov3_multi`）。用旧 run 评估旧动作时，要自己指定 `--motion_dir`（旧版动作在 `retarget/data/archive/supermimic_motions/`）。
