# SuperMimic 复现 —— 里程碑记录

本文件记录两次完整复现：原作者的首次实现（RTX PRO 6000），以及按 [README_CN.md](README_CN.md) 在一台全新机器上从零重跑的结果（RTX 4090）。
成功标准（两次相同）：评估 normalized return ≥ 0.82 **且** 40/40 段 clip 都没有触发 termination。

## 原作者首次复现（2026-10-21）

时间为当地时间（CEST，UTC+2），标注 UTC 的除外。单 GPU（RTX PRO 6000 Blackwell），48 个 CPU 线程。

| 时间 | 里程碑 |
|---|---|
| 2026-10-21 16:55 | 任务开始；读完论文（没有公开代码：项目主页显示 "Coming soon"） |
| 2026-10-21 17:00 | Clone IsaacLab main @ 46d6cc3（VERSION 3.0.0）；开始构建 kit-less 镜像（`Dockerfile.kitless`） |
| 2026-10-21 17:01 | kit-less 镜像构建完成（`isaac-lab-kitless:main-46d6cc3`，加上 tmux/ffmpeg → `supermimic:latest`）；容器 `supermimic` 启动 |
| 2026-10-21 17:07 | Task 1 完成：Isaac Lab 自带的 Isaac-Velocity-Flat-G1 PPO 能在 kit-less 环境下用 Newton/MuJoCo-Warp 训练（`physics=newton_mjwarp`），4096 envs，约 224k steps/s |
| 2026-10-21 17:16 | G1 29-DoF asset 在 MJWarp 上跑通（URDF → MJCF → USD，BeyondMimic actuators）：4096 envs 下约 640k control steps/s |
| 2026-10-21 17:26 | 完整 SuperMimic pipeline（sim collect → world model（WM）fit → BPTT policy + critic）端到端跑通；排查 WM loss 尺度和速度问题（0.66 it/s） |
| 2026-10-21 17:49 | 速度优化（`torch.compile`、reward / critic / WM-error 批量计算、bf16 MLP）：0.66 → 3.1 it/s。**主训练开始**：`logs/lafan1_main`（容器内 tmux 会话 `lafan1_main`） |
| 2026-10-21 17:50 | 以 `njmax=500` / `nconmax=128` 重启主训练（摔倒时偶尔会出现 MJWarp constraint overflow） |
| 2026-10-21 18:11 | 第一个 checkpoint（it 2000，训练 0.15 h）：评估 normalized return 0.541，12/40 段失败。启动评估监视器（tmux `eval_main`） |
| 2026-10-21 18:19 | it 4000（训练 0.41 h）：评估 normalized return 0.685，4/40 段失败 |
| 2026-10-21 18:32 | it 6000（训练 0.62 h）：评估 normalized return 0.781，1/40 段失败 |
| 2026-10-21 18:39 | 确定成功标准：评估 return ≥ 0.82 **且** 40/40 段 clip 都没有 termination |
| 2026-10-21 18:47 | it 8000（训练 0.82 h）：评估 normalized return 0.779，2/40 段失败。分析 fallAndGetUp1_subject5（it 6000）的失败：是爬起动作中真实的跟踪失败（机器人翻滚），论文的 tilt termination 正确触发 |
| 2026-10-21 18:57 | it 10000（训练 1.04 h）：评估 normalized return 0.811，1/40 段失败（fallAndGetUp2_subject3） |
| 2026-10-21 19:10 | **达到 return 目标**：it 12000（训练 1.24 h，checkpoint 保存于 2026-10-21 17:05:52 UTC）：评估 normalized return 0.824 ≥ 0.82（论文：H100 上 1.2 h 达到 0.82）；1/40 段失败（jumps1_subject2） |
| 2026-10-21 19:23 | **复现成功（两项标准都满足）**：it 14000，训练 1.47 h（checkpoint 保存于 2026-10-21 17:19:24 UTC，即 19:19 CEST）：评估 normalized return 0.829 ≥ 0.82 **且** 0/40 段失败 |
| 2026-10-21 19:31 | 按要求在 it ≈ 15700（训练 1.67 h）停止训练；最后使用的 checkpoint 是 `ckpt_014000`（40/40 的那个） |
| 2026-10-21 19:44 | 渲染复现视频（`results/supermimic_lafan1_it14000_reproduced.mp4`）；删除容器，释放 GPU |

## 新机器从零重跑（2026-10-22，RTX 4090）

时间为当地时间（CST，UTC+8），标注 UTC 的除外（`eval.csv` 里的 `saved_at` / `eval_at` 是 UTC）。
2× RTX 4090 24 GB（训练只用 GPU 0），32 个 CPU 线程，188 GB 内存，国内网络。流程与踩坑细节见 [README_CN.md](README_CN.md)。

| 时间 | 里程碑 |
|---|---|
| 2026-10-22 17:56 | 任务开始：检查宿主机（driver 580.173 / CUDA 13.0，Docker 29.4，`docker run --gpus` 可用）；两张 GPU 上都有其他进程占用显存（6.5 GB / 12.7 GB） |
| 2026-10-22 17:58 | Clone IsaacLab main @ 46d6cc3；在宿主机上下载 LAFAN1 G1 retarget 数据和 G1 description（40 个 CSV、38 个 STL，共 113 MB，约 2.5 分钟） |
| 2026-10-22 17:59 | kit-less 镜像第一次构建失败：Docker Hub 不通（`# syntax=docker/dockerfile:1` 和 `ubuntu:24.04` 都拉不到） |
| 2026-10-22 18:01 | 通过 build-arg（`BUILDKIT_SYNTAX`、`BASE_IMAGE_ARG`）改走 daocloud 镜像源后重新构建 |
| 2026-10-22 18:06 | 第二次构建失败：`uv sync` 从 GitHub 拉 `rl_games` 时断线（`curl 92 HTTP/2 stream ... CANCEL`），属于网络问题 |
| 2026-10-22 18:37 | 重试后 kit-less 镜像构建成功（耗时 29 min 56 s，主要花在从 PyPI 下载 wheel 上；镜像 18.7 GB）。`tools/setup.sh` 加入 `DOCKER_MIRROR` 和自动重试 |
| 2026-10-22 18:42 | `supermimic:latest` 构建完成（约 5 分钟）；用户已释放两张 GPU 上的显存 |
| 2026-10-22 18:43 | 容器 `supermimic` 启动（GPU 0）；URDF → MJCF → USD 转换完成（`nq 36 nv 35 nbody 40 ngeom 71 mass 33.34`） |
| 2026-10-22 18:44 | `tools/test_env.py` 通过：FK pos err 6.5e-7，linvel / angvel err ~1e-6。**主训练开始**：`logs/lafan1_main`，默认参数 |
| 2026-10-22 18:51 | 训练速度：WM-only 阶段约 6.9 it/s，it 500 开始 policy 更新后约 2.66 it/s（原作者约 3.1 it/s）；显存 17.3 GB。启动评估监视器（tmux `eval_main`，同卡，合计约 19 GB） |
| 2026-10-22 19:01 | it 2000（训练 0.21 h，checkpoint 保存于 10:57:59 UTC）：评估 normalized return 0.5466，12/40 段失败（原作者 0.541，12/40） |
| 2026-10-22 19:06 | 新增 `tools/viser_play.py`：用 ckpt 2000 录制的轨迹在浏览器里播放，headless Chromium 截图确认渲染正常；`docker/run_container.sh` 加入 8080 端口映射 |
| 2026-10-22 19:16 | it 4000（训练 0.47 h）：评估 normalized return 0.7105，4/40 段失败（原作者 0.685，4/40） |
| 2026-10-22 19:30 | it 6000（训练 0.70 h）：评估 normalized return 0.7827，1/40 段失败（原作者 0.781，1/40） |
| 2026-10-22 19:44 | it 8000（训练 0.94 h）：评估 normalized return 0.7758，4/40 段失败（原作者 0.779，2/40） |
| 2026-10-22 19:58 | it 10000（训练 1.17 h）：评估 normalized return 0.7912，3/40 段失败（dance2_subject4、fallAndGetUp2_subject2、walk3_subject4） |
| 2026-10-22 20:12 | **复现成功（两项标准都满足）**：it 12000，训练 1.40 h（checkpoint 保存于 12:09:15 UTC，即 20:09 CST）：评估 normalized return 0.8308 ≥ 0.82 **且** 0/40 段失败 |
| 2026-10-22 20:17 | 录制 ckpt 12000 的评估轨迹 `logs/lafan1_main/traj_012000.npz`（单独评估得到 0.8315，0/40；与 0.8308 的差异来自 GPU 仿真的非确定性） |
| 2026-10-22 20:27 | it 14000（训练 1.66 h）：评估 normalized return 0.8273，1/40 段失败（fallAndGetUp2_subject3），所以最佳 checkpoint 仍是 `ckpt_012000` |
| 2026-10-22 20:28 | 在 it ≈ 14300（训练 1.71 h）停止训练；成功曲线保存为 `results/success_curve_rtx4090.png` |
| 2026-10-22 20:29 | `tools/render.py` 渲染 mp4 验证通过；viser 加载 ckpt 12000 的轨迹（dance1_subject1：131.5 s 完整播完，return 0.819） |
| 2026-10-23 | 用户从另一台电脑通过 `http://<服务器IP>:8080` 打开 viser；浏览器报 "WebGL is unavailable"，在 `chrome://flags` 启用 *Override software rendering list* 后正常显示。删除容器、释放 GPU；新建 `develop` 分支，加入 `/commit` skill |
