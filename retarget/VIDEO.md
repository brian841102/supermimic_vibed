# 训练素材（VIDEO）

nemo_v3 的训练动作都来自视频：视频 → GVHMR → 重定向 → 训练片段。这份文档记录每一段**用到的素材**：来源、原视频放在哪、做过哪些调整（裁剪、放大、取舍的时间段）、产物在哪、动作是什么类型。

- 流程和指令见 [README.md](README.md)，选片过程和遇到的问题见 [MILESTONE.md](MILESTONE.md)。
- 视频和产物都在 `retarget/data/`（不提交，视频有版权、体积大），所以这份文档是唯一的记录。
- 新增动作：由 Claude 先问你"怎么描述这段动作"，再把你的描述写进下面的表和"描述"栏（见文末"新增一段动作"）。

## 动作类型

粗略分成三类（以后有不适合的再加）：

| 类型 | 意思 |
|---|---|
| 舞蹈 | 成套的舞蹈动作，有节奏、有编排 |
| 情绪动作 | 表达情绪的动作，例如开心、难过 |
| 特技 | 比较短的花活，例如原地回转 |

## 一览

| 名称 | 类型 | 动作描述 | 来源 | 训练片段 | 长度 |
|---|---|---|---|---|---|
| `roller32` | **舞蹈** | 四轮鞋花样轮滑基础图形：户外、缓慢滑行、双脚着地为主，手臂摆出不同姿势，结尾站立不动 | Wikimedia Commons | `supermimic_motions/roller32/roller32.csv` | 34.6 s |
| `roller_house` | 特技 | 室内四轮鞋：背对镜头滑远，转身（根部在世界里转约 1.5 圈），站定 | 用户提供 | `supermimic_motions/roller_house/roller_house.csv` | 10.3 s |
| `three_turn` | 特技 | 直排轮三字转示范：向前滑 → 三字转 → 倒滑（单脚，右脚拖在身后） | 用户提供 | `supermimic_motions/three_turn/three_turn.csv` | 11.7 s |

> "类型"由你决定（`roller32` 归舞蹈，`roller_house` 归特技）；"动作描述"目前是 Claude 根据画面写的，**待你修改成自己的描述**。单轮滑行的时长是从训练片段里的"动作"段量的（轮子离地超过 3 cm、持续 0.5 s 以上）。

nemo_v3 默认的训练目录是 `retarget/data/supermimic_motions/nemov3_multi/`（上面三段 csv 的复制，目录里放几段就一起训练几段）。

G1 的训练数据是 LAFAN1（40 段，在仓库根目录的 `data/lafan1_g1/`），不在这份文档里；LAFAN1 的授权是 CC BY-NC-ND，不能再分发。

---

## roller32

| 项目 | 内容 |
|---|---|
| 类型 / 动作描述 | **舞蹈**。四轮鞋（quad）花样轮滑基础图形讲解：户外、单人、动作很慢，手臂摆出不同姿势；单轮滑行（轮子离地超过 3 cm、持续 0.5 s 以上）累计约 17 s、共 7 次，**最长一次约 6.8 s**（左轮，视频第 4.8–11.6 s）；没有交叉步（两轮左右间距最小 0.093 m，轮距 0.172 m）；结尾站立不动 |
| 来源 | Wikimedia Commons：`EVD-patinajeartistico-007.ogv`「花样轮滑基础图形讲解」，授权 CC BY-SA 2.5 co（需要署名、相同方式共享）。页面：<https://commons.wikimedia.org/wiki/File:EVD-patinajeartistico-007.ogv> |
| 原视频位置 | `retarget/data/videos/commons/EVD-patinajeartistico-007.ogv`（320×224，25 fps，56.8 s） |
| 调整一：截取和放大 | 取原片 **10–50 s**（40 s），用 lanczos 放大到 640×448、30 fps，得到 `retarget/data/videos/roller40.mp4`。等价命令：`ffmpeg -ss 10 -t 40 -i EVD-patinajeartistico-007.ogv -vf "scale=640:448:flags=lanczos,fps=30" roller40.mp4` |
| 流程 | `retarget/scripts/run_video_to_nemov3.sh retarget/data/videos/roller40.mp4`：GVHMR（移动镜头，不做静止关节后处理）→ 相机轨迹（考虑变焦、修复朝向翻转 13.40–13.67 s、修复左右腿互换 6 段）→ 重定向（`hw` 版：腰、头、推杆、腕 pitch 焊死；单轮平衡加离地余量 4.5 cm） |
| 中间产物位置 | `retarget/data/gvhmr_nopp/roller40/`、`gvhmr_cam/roller40/`、`retarget_out/roller40_{full,hw}.{csv,pkl,json}` |
| 调整二：训练片段 | 取重定向结果 **0–30.57 s**（原视频 10–40.57 s）：`roller40` 的第 32 s 附近她站立不动，取其中最静止的一帧 30.57 s 截断；前后各加 1 s HOME 静止和 1 s 余弦衔接（根部匀加速 / 匀减速到静止）。命令：`make_supermimic_clip.py --src retarget_out/roller40_hw.csv --end_s 30.57 --out supermimic_motions/roller32/roller32.csv`，共 1038 帧（34.6 s） |
| 训练片段位置 | `retarget/data/supermimic_motions/roller32/roller32.csv`（和 `.json`：分段、帧数） |
| 训练结果 | 单独训练（exp6b）回报 0.902、0/1 失败；和另两段一起训练（`logs/nemov3_multi/…exp5_roller_house_slim`，第 11000 轮）：这一段 0.9227、点脚 0 次 |
| 备注 | 画面分辨率只有 320×224（放大后处理）；镜头是跟拍，所以走动态相机流程；原片没有声音和剪辑 |

## roller_house

| 项目 | 内容 |
|---|---|
| 类型 / 动作描述 | **特技**（原地回转）。室内四轮鞋：背对镜头滑远 → 走几步并转身（根部在世界坐标里累计转约 650°，最高 277°/s；髋 yaw 关节最高 2.0 rad/s）→ 站定；单人，穿蓬裙（会遮住部分腿）；没有交叉步（两轮左右间距最小 0.062 m）；单轮滑行共 2 次、累计 2.8 s，最长一次 1.8 s |
| 来源 | 用户提供的文件 `Roller skating inside the house.mp4`（从用户另一台电脑的下载文件夹拷贝），**原始出处和授权不详** |
| 原视频位置 | `retarget/data/videos/incoming/roller_house.mp4`（608×1080 竖屏，30 fps，9.7 s，h264） |
| 调整一：取舍的时间段 | 只用视频 **1.1–7.4 s**：1.1 s 起她的头和脚才完整在画面内（更早只有手臂或脚被切掉）；7.4 s 她转完身站定（根部速度 0.07 m/s、转速降到 28°/s），再往后她又迈步朝镜头走，不算立定 |
| 流程 | `run_video_to_nemov3.sh retarget/data/videos/incoming/roller_house.mp4`：同 roller32；左右腿互换修复 1 段（1.2–1.5 s）；重定向用 `hw` 版；单轮平衡之后又更新了两处（把轮子悬空的时段并入单轮阶段、换支撑脚时支撑轮的位置成本平滑过渡，MILESTONE 23.4–23.5），训练片段已用新流程重新生成 |
| 中间产物位置 | `retarget/data/gvhmr_nopp/roller_house/`、`gvhmr_cam/roller_house/`、`retarget_out/roller_house_{full,hw}.{csv,pkl,json}`、`roller_house_clip.pkl`（训练片段的播放版） |
| 调整二：训练片段 | 重定向结果 **1.1–7.4 s**（6.3 s）；前后各 **1 s** HOME 静止和 **1 s** 衔接（和 roller32 一致；一开始用 2 s，占动作的 39%，太长）。命令：`make_supermimic_clip.py --src retarget_out/roller_house_hw.csv --start_s 1.1 --end_s 7.4 --out supermimic_motions/roller_house/roller_house.csv`，共 310 帧（10.3 s） |
| 训练片段位置 | `retarget/data/supermimic_motions/roller_house/roller_house.csv`（和 `.json`） |
| 训练结果 | 和 roller32、three_turn 一起训练（`logs/nemov3_multi/…exp5_roller_house_slim`，第 11000 轮）：这一段 0.9186、点脚 0 次（第 8000 轮起点脚都是 0；3000–7000 轮每次评估点脚 1 次、0.12–0.24 s） |
| 备注 | 只有 9.7 s（要求是约 30 s）；人由远到近，大小变化大，GVHMR 深度估计受影响；来源和授权不详，**不要提交视频和产物**（`retarget/data/` 已在 `.gitignore`） |

## three_turn

| 项目 | 内容 |
|---|---|
| 类型 / 动作描述 | **特技**（三字转）。直排轮、室内轮滑场、单人、摄像机固定：她从远处面朝镜头向前滑，约视频 9.5–11.0 s 做一次三字转（朝向相对行进方向从 +30° 转到 −150°），之后背对镜头倒滑，**单脚滑行约 2.8 s**（视频 12.2–15 s，右脚拖在身后、脚尖朝下，脚跟抬高）。*（动作描述是 Claude 写的草稿，待你修改。）* |
| 来源 | 用户提供的文件 `3 Turns On Roller Skates Inlines Example.mp4`（从用户另一台电脑的下载文件夹拷贝），**原始出处和授权不详** |
| 原视频位置 | `retarget/data/videos/incoming/three_turn_orig.mp4`（1920×1080，59.94 fps，34.6 s，h264）；片头有"3-Turn Example"字样 |
| 调整一：截取 | 先取原片 **5–17 s** 转成 30 fps，得到 `retarget/data/videos/three_turn.mp4`（流程输入）和 854×480 的预览 `three_turn_480p.mp4`。等价命令：`ffmpeg -ss 5 -t 12 -i three_turn_orig.mp4 -vf fps=30 -c:v libx264 -crf 18 -an three_turn.mp4` |
| 流程 | `run_video_to_nemov3.sh retarget/data/videos/three_turn.mp4`（得到 `gvhmr_nopp`、`retarget_out/three_turn_*`），再用 `gvhmr_camera_traj.py --floor_fit` 重做相机轨迹、重新重定向（GVHMR 的重力方向有 11.8° 误差，不修的话抬起的变成左脚）；左右腿互换修复 1 段（视频 16.3–16.5 s，在取用范围之外）；平衡之后再限速一次（MILESTONE 22.2–22.4）；单轮阶段把轮子悬空的时段也算进去（MILESTONE 23.4），右轮 6.6–10.6 s 的离地高度最低点 3.0 → 4.4 cm |
| 中间产物位置 | `retarget/data/gvhmr_nopp/three_turn/`、`gvhmr_cam/three_turn/`、`retarget_out/three_turn_{full,hw}.{csv,pkl,json}`、`three_turn_clip.pkl`（训练片段的播放版） |
| 调整二：训练片段 | 取重定向结果 cut 时间 3.0–10.0 s（**原视频 8.0–15.0 s**，7.0 s）：7 s 之前 GVHMR 没追到人，8.0 s 她站着展开双手；15.5 s 起快碰到画面右边缘（16.0 s 被切）。前后各 1 s HOME 静止，衔接用 `--smooth`（根部按最小急动度过渡，a_max 取这段自己根部加速度的 p75 = 0.48 m/s²；开头 1.30 s、结尾 1.43 s）。命令：`make_supermimic_clip.py --src retarget_out/three_turn_hw.csv --start_s 3.0 --end_s 10.0 --smooth --out supermimic_motions/three_turn/three_turn.csv`，共 351 帧（11.7 s） |
| 训练片段位置 | `retarget/data/supermimic_motions/three_turn/three_turn.csv`（和 `.json`）；已复制到 `nemov3_multi/` 一起训练 |
| 训练结果 | 和 roller32、roller_house 一起训练（`logs/nemov3_multi/…exp5_roller_house_slim`，第 11000 轮）：这一段 0.9357、点脚 0 次 |
| 备注 | ① 单轮滑行：右轮视频 10.0–15.4 s 持续抬起约 5 s（3–5 cm，三字转时最高 11 cm），超过"抬脚不超过 2 s"，**你说可以接受**（这是视频本身的动作）；② 交叉：视频 9.9–10.5 s（`three_turn_clip` 的 4.13–4.80 s）两轮中心的横向间距降到 −0.4 cm，累计约 0.7 s 低于轮宽 6 cm；**你检查过没有碰撞，决定照现状训练**；`three_turn_hw` 在视频 16 s 之后（人走到画面右边缘、追踪失效）的数据不能用，不在训练片段里；③ 髋 yaw 最大 5.0 rad/s（限速），其他关节 ≤ 9.0 rad/s；④ 倒滑时髋前倾平均约 20°；⑤ 来源和授权不详，**不要提交视频和产物** |

---

## 新增一段动作

1. 视频放到 `retarget/data/videos/incoming/`（或 `videos/`），告诉 Claude 文件名和要用的时间段。
2. **Claude 先问你"这段动作怎么描述"**：属于哪一类（舞蹈 / 情绪动作 / 特技，或新增一类）、动作内容、有没有交叉步、单轮滑行的时间、转身和静止的位置、是否有特别要注意的地方。
3. 跑流程：`run_video_to_nemov3.sh <视频>`，检查（交叉、单轮时长、髋 yaw 速度、关节限位），用 `viser_play_motion.py` 看结果。
4. 生成训练片段：`make_supermimic_clip.py --src <视频名>_hw.csv --start_s <起> --end_s <止> --out supermimic_motions/<名称>/<名称>.csv`。
5. 想一起训练，就把这段 csv 复制到 `supermimic_motions/nemov3_multi/`。
6. 把这段加到本文档：一览表加一行，再照 roller32 / roller_house 的格式写一节（来源、原视频位置、调整、产物位置、动作描述、训练结果、备注）。

### 空白模板

| 项目 | 内容 |
|---|---|
| 类型 / 动作描述 | （舞蹈 / 情绪动作 / 特技 / 其他：你的描述） |
| 来源 | （网站 / 提供者、授权） |
| 原视频位置 | `retarget/data/videos/...`（分辨率、帧率、长度） |
| 调整一：截取 / 放大 | （取视频的哪一段，是否放大） |
| 流程 | （用的指令，修复了哪些段） |
| 中间产物位置 | （`gvhmr_nopp/`、`gvhmr_cam/`、`retarget_out/`） |
| 调整二：训练片段 | （`--start_s` / `--end_s` / `--blend_s`、总长度） |
| 训练片段位置 | `retarget/data/supermimic_motions/<名称>/<名称>.csv` |
| 训练结果 | （run、回报、失败） |
| 备注 | （限制、授权、注意事项） |
