---
name: commit
description: 整理并提交代码的四步流程——把目前的更动按"对目标有效 / 无效 / 待验证"分类，由用户决定去留，给出按开发逻辑排序、符合命名规范的提交计划，用户批准后自动提交。用户说"整理代码""提交""/commit"时使用。
---

# commit：整理代码并提交

四个步骤依次进行，**每一步结束都停下来等用户**，不要跳步，也不要替用户做第 2、4 步的决定。回复用简体中文。

本仓库的目标：在 LAFAN1（40 段）上复现 SuperMimic。成功标准是评估归一化回报 ≥ 0.82 **且** 0/40 段失败（见 `MILESTONES.md`）。
所有"有效 / 无效"的判断都以这个目标为准。

## 第 1 步：把目前的更动分类

1. 收集全部更动：`git status --short`、`git diff`（已跟踪）、未跟踪文件逐个看内容；与上一次提交（`git log --oneline -10`）以及上游 `main` 对照，确认范围。
2. 按"功能 / 实验"把 hunk 分组（一个文件可以拆到多组），每组写清楚：做了什么、涉及哪些文件、服务于哪个训练 run（`logs/<run>/`）或哪个目标（复现指标、训练速度、环境搭建、可视化等）。
3. 分三类，**每条都要给依据**。依据来源有：
   - `MILESTONES.md` 里的里程碑和结论；
   - 对应 run 的 `logs/<run>/eval.csv`（各 checkpoint 的 `norm_return` / `n_failed`）、`train_log.csv`（`it_per_s`、`wm_loss` 等）；
   - `tools/test_env.py` 的输出；
   - 用户原话。

   没有依据的不能归为"有效"：
   - **对目标有效**：有实验证明它带来了想要的结果。写出对照的 run 和数字，例如"run A vs run B：it 12000 回报 0.83 vs 0.79，失败 0/40 vs 3/40"，或"训练速度 2.1 → 2.7 it/s"。
   - **对目标无效**：实验证明没用、有害（回报下降、失败段数增加、变慢、显存爆掉），或者被后来的做法取代。
   - **是否有效待验证**：还没做对照实验、结果混合（例如回报变高但失败段数也变多），或只是工具 / 基础设施、无法用实验判断。
4. 另外单独列出：
   - 评估 / 可视化工具（`supermimic/evaluate.py`、`tools/render.py`、`tools/viser_play.py`、`tools/plot.py` 等），说明未来是否可能还要用；
   - 环境搭建脚本（`docker/`、`tools/setup.sh`、`tools/download_data.sh`）；
   - 文档（`README.md`、`README_CN.md`、`MILESTONES.md`、`results/`）；
   - skill / 配置（`.claude/`）；
   - 只在 scratchpad、`.cache/` 或 `logs/` 下的临时脚本（不提交）。
5. 用表格呈现：`| # | 更动 | 文件 | 相关 run / 目标 | 依据 | 分类 |`，最后给出我的建议（保留 / 移除），然后**停下来等用户决定**。

## 第 2 步：用户决定去留

1. 用户逐项决定。有疑问的项先问清楚，不要自己猜。
2. **移除前先备份**：在 `logs/<主题>_backup_<日期>/` 下存 `uncommitted.diff`（`git diff`）和未跟踪文件打包成的 `untracked.tgz`。`logs/` 已被 gitignore，不会提交。
3. 按决定移除代码后检查：
   - 所有 Python 文件能编译：`python3 -m py_compile supermimic/*.py tools/*.py`；
   - 容器里能跑通（需要先 `docker/run_container.sh <GPU> 128g`）：
     - `docker exec supermimic python tools/test_env.py`：`FK pos err`、`FK linvel/angvel err` 仍在 1e-6 量级；
     - 训练能启动：`docker exec supermimic python supermimic/train.py --num_envs 256 --iterations 600 --log_every 50 --log_dir logs/_smoke_<日期>`，要跑过 it 500（策略更新开始）且不报错；
     - 评估能载入已有 checkpoint：`docker exec supermimic python supermimic/evaluate.py --run_dir logs/<run> --ckpt logs/<run>/ckpt_XXXXXX.pt`；
   - 保留实验的设定与对应 run 一致：`supermimic/train.py` 的参数默认值和 `logs/<run>/args.json` 逐项一致，只允许执行期字段不同（`log_dir`、`resume`、launcher 参数）；
   - 之前已提交的设定没有被改到：`git diff main -- supermimic/ run_train.sh` 里只有计划内的改动，超参数（`train.py` 参数默认值）、MDP 常数（`mdp.py`）、执行器参数（`robot.py`）、仿真设置（`env.py` 的 `make_sim`）都没有被顺手改掉；
   - 有改动的辅助脚本能跑（例如改了 `tools/viser_play.py` 就要起一次 viser、确认能打开页面；改了 `tools/setup.sh` 就要重跑一次，命中缓存很快）。
4. 被移除的实验只在 `MILESTONES.md` 里留记录，并注明"代码已移除、备份位置"。
5. 报告检查结果，进入第 3 步。

## 第 3 步：提交计划

1. **按开发逻辑排序**：后面的提交可以依赖前面的，反之不行；每个提交单独检出时都应能导入、不报错。顺序如下：
   1. 环境搭建 / 基础设施：`docker/`、`tools/setup.sh`、`tools/download_data.sh`、`run_train.sh`；
   2. 机器人模型与资产：`supermimic/robot.py`、`supermimic/kinematics.py`、`supermimic/rotations.py`、`tools/urdf_to_mjcf.py`、`tools/mjcf_to_usd.py`；
   3. 动作库、MDP 与仿真环境：`supermimic/motion.py`、`supermimic/mdp.py`、`supermimic/env.py`、`supermimic/buffer.py`；
   4. 算法与训练：`supermimic/networks.py`、`supermimic/train.py`；
   5. 评估与可视化工具：`supermimic/evaluate.py`、`tools/render.py`、`tools/viser_play.py`、`tools/plot.py`、`tools/analyze_failure.py`、`tools/test_*.py`、`tools/bench.py`；
   6. 文档与结果图：`README.md`、`README_CN.md`、`MILESTONES.md`、`results/`；
   7. skill / 配置：`.claude/`。
2. **命名规范（Conventional Commits）**：标题用英文，正文用简体中文。格式为 `<type>(<scope>): <subject>`。
   - type：`feat` 新功能、`fix` 修 bug、`refactor` 不改行为的重构、`chore` 工具 / 杂项、`docs` 文档、`test` 测试、`perf` 性能；
   - scope：模块名，例如 `docker`、`setup`、`robot`、`motion`、`mdp`、`env`、`train`、`eval`、`tools`、`viser`、`docs`、`skills`；
   - subject：祈使句、小写开头、不加句号、≤ 72 字符；
   - 正文（简体中文）：说明为什么改、改了什么，引用对应的 run（`logs/<run>`）和关键结果（回报、失败段数、it/s、显存）。
3. 每个提交列出：提交信息（标题 + 正文）、包含的文件（一个文件拆到多个提交时，写明各自包含哪些 hunk）。
4. 检查不该提交的东西：论文 PDF、`IsaacLab/`、`data/`（LAFAN1 是 CC BY-NC-ND，不能再分发）、`logs/`、checkpoint（`*.pt`）、录制轨迹（`*.npz`）、视频（`*.mp4`）、`.cache/`、`build_*.log`、scratchpad、密钥。
   大文件也不要提交：`results/` 下的新图片控制在几百 KB 以内。本仓库是公开 fork（`origin` 是自己的 fork，`upstream` 是 `zita-ch/supermimic_vibed`），不用 Git LFS。
5. 呈现计划后**停下来讨论**。用户要求修改时，改完重新呈现整个计划。

## 第 4 步：批准后自动提交

1. 只有用户**明确批准**（例如"提交吧""批准"）才执行。
2. 在当前分支上按计划顺序逐个提交（一般是 `develop`；如果在 `main` 上，先问用户要不要开新分支）：
   - 只 `git add` 计划列出的路径；一个文件分到多个提交时，用 `git add -p` 的等价做法（生成对应 hunk 的 patch，再 `git apply --cached`）；
   - 提交信息先写到 scratchpad 的文件里（没有 scratchpad 就写到已被 gitignore 的 `.cache/`），再用 `git commit -F <文件>`；
   - 提交信息结尾加上当前会话 system reminder 里给出的署名行（例如 Co-Authored-By）；
   - 每个提交后确认 `git status` 与预期一致。
3. 全部完成后给出 `git log --oneline`（本次新增的提交）和剩余未提交的更动（应为空，或只剩用户决定不提交的）。
4. **不推送（push）**，除非用户另外要求。
