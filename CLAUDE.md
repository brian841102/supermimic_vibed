# CLAUDE.md

本仓库：在 Isaac Lab 3（kit-less，Newton + MuJoCo-Warp）上复现 SuperMimic，并把它适配到 nemo_v3 轮足机器人
（`retarget/`：视频 → GVHMR → GMR → nemo_v3 参考动作）。G1 和 nemo_v3 必须**同时可用**，不可只顾一种机型。

## 沟通
- 回复和进度回报一律用**简体中文**，英文只留专有名词、代码和命令。

## 文档分工
- `README.md` / `README_CN.md`：G1 复现的安装与指令；`MILESTONES.md`：G1 复现的里程碑。
- `retarget/README.md`：重定向与 nemov3 训练的安装依赖、常用指令。
- `retarget/MILESTONE.md`：计划书、开发中遇到的坑、因此做的调整（按章节追加）。
- `retarget/VIDEO.md`：训练素材的来源、原视频位置、调整（裁剪等）和动作类型。**用户每新增一段动作，先问用户怎么描述这段动作**：属于哪一类（舞蹈 / 情绪动作 / 特技，或新增一类）、动作内容、有没有交叉步、单轮滑行的时间等，再写进 VIDEO.md。

## 机型选择
- 环境变量 `SUPERMIMIC_ROBOT=g1|nemov3`（或 `--robot`），必须在 import `supermimic.mdp` 之前设定；默认 `g1`。
- 机型相关常数集中在 `supermimic/robot_spec.py` 与 `supermimic/robots/<robot>.py`。改通用代码时，G1 的行为（奖励、超参数、
  执行器、仿真设定）必须保持不变；新功能用 spec 字段开关（例如 `lift_points`，G1 为空）。

## 训练与评估
- 在容器 `supermimic` 里跑（`docker/run_container.sh <GPU> 128g`），训练、评估各开一个 tmux。
- 训练目录：`--exp_name <实验> --run_name <这次改了什么>` -> `logs/<实验>/<YYYY-mm-dd_HH-MM-SS>_<run_name>/`（同 mjlab），
  `logs/<实验>/latest` 指向最新一次；同一组实验的多次尝试都放在同一个 `<实验>` 底下。
- checkpoint / 评估频率：**3000 轮之前每 1000 轮，之后每 500 轮**：
  `supermimic/train.py ... --ckpt_every 1000 --ckpt_every_late 500 --ckpt_late_from 3000`，
  另开 `supermimic/evaluate.py --run_dir logs/<实验>/latest --watch` 评估每个 checkpoint。
- 每次评估都回报（回报、失败段数）；回报不上升就立即停下来查原因并修正。
- 饱和判定：连续 3 次评估都没有超过「最好成绩 + 0.005」即停止，然后把最好的 checkpoint 放到 viser 并给出网址。
- 曲线：训练只写 CSV；`tools/csv_to_tensorboard.py logs/<run> --watch`（在容器里跑，`logs/` 归容器的 root 所有），
  主机上 `tensorboard --logdir logs --bind_all --port 6006`。

## nemo_v3 仿真资产
- `retarget/scripts/make_sim_mjcf.py` 生成 MJCF，`tools/mjcf_to_usd.py` 转 USD。转换 nemov3 时必须带
  `--hull_vertices '.*_wheel_collision=1024'`：Newton 默认只留 64 个凸包顶点，轮子会变成不规则多边形而抖动
  （`retarget/MILESTONE.md` 19.1）。轮子碰撞体保持真实网格（与上过实体机的 flamingo、booster_mjlab 一致）。

## Git
- `main` 只用来同步上游（`upstream`）；日常开发在 `develop`。
- 提交走 `/commit` 流程（`.claude/skills/commit/`），只在用户要求时提交、推送。
- 本仓库是**公开 fork**，不要提交：`data/`（LAFAN1 是 CC BY-NC-ND）、`logs/`、checkpoint（`*.pt`）、轨迹（`*.npz`）、
  视频、`retarget/data/`、`.cache/`、论文 PDF、密钥。文档里的内网 IP、主机路径、私有仓库地址要去敏。

## 已知的坑
- `pkill -f <pattern>` 不要和含该 pattern 的命令写在同一行（会杀掉自己的 shell），改用 PID 文件。
- `conda create -y` 会覆盖同名环境，先 `conda env list` 确认。
- 容器里生成的文件归 root，主机进程写不进 `logs/`；需要写入时在容器里执行。
