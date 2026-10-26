# retarget：视频 → nemo_v3 动作重定向

把单目视频里的人体动作（目前是花式滑冰）重定向到轮足机器人 nemo_v3，输出给 SuperMimic（CSV）和 booster_mjlab（`MotionFile` pkl）用的参考动作。

流程：**GVHMR**（视频 → SMPL-X）→ **相机轨迹重建**（保留滑行）→ **GMR**（IK，nemo_v3 的 full / hw 两个版本）→ **轮子后处理** → 对比视频 / 浏览器播放。

> 计划书、开发过程中踩到的坑、以及因此做的调整，都在 [MILESTONE.md](MILESTONE.md)。本文件只讲安装和常用指令。

所有命令都在**仓库根目录**（`supermimic_vibed/`）下执行。

---

## 目录

```
retarget/
├── README.md                  # 本文件：安装与常用指令
├── MILESTONE.md               # 计划书 + 开发记录（坑与调整）
├── VIDEO.md                   # 训练素材：来源、原视频位置、调整（裁剪等）、动作类型
├── configs/smplx_to_nemo_v3.json   # GMR 的 IK 配置（full / hw 共用）
├── patches/gvhmr_skating.patch     # 对 GVHMR 的补丁
├── scripts/                   # 主流程（产出数据）：生成模型、相机轨迹、重定向、一键脚本
├── tools/                     # 检查与可视化（不产出训练数据）：模型检查、对比视频、浏览器播放
├── assets/nemo_v3/            # nemo_v3 模型 + 生成的重定向专用 MJCF
├── third_party/               # submodule：GVHMR、GMR（权重和环境不在仓库里）
└── data/          ⚠ 不提交    # 视频、GVHMR 结果、重定向结果、对比视频
```

`retarget/data/` 和仓库根目录的 `logs/` 里面是这样（都不提交）：

```
retarget/data/
├── videos/                    # 输入视频：roller40.mp4、incoming/roller_house.mp4；commons/ 放 Commons 原片
├── gvhmr_nopp/<视频名>/       # GVHMR 输出（hmr4d_results.pt、渲染视频）
├── gvhmr_cam/<视频名>/        # 加上相机轨迹、修好朝向翻转和左右腿互换的 hmr4d_results.pt（重定向的输入）
├── retarget_out/              # <视频名>_{full,hw}.{csv,pkl,json}、对比视频、<视频名>_clip.pkl（训练片段的播放版）
├── supermimic_motions/        # 训练动作：roller32/、roller_house/、three_turn/ 各一段；nemov3_multi/ 是默认训练目录（这些段的复制）
├── archive/                   # 旧版动作（让旧 run 可以重现）
└── run_<视频名>.log、viser_8080.{log,pid}

logs/                          # 仓库根目录
├── lafan1_main/               # G1：--log_dir 指定，里面直接是 train_log.csv、eval.csv、ckpt_*.pt
├── nemov3_multi/              # nemov3：--exp_name 指定，同一组实验放在一起
│   ├── <YYYY-mm-dd_HH-MM-SS>_<run_name>/   # 每次训练一个目录：args.json、train_log.csv、eval.csv、ckpt_*.pt、traj_best.npz、tb/
│   └── latest -> 最新的一次
└── archive/                   # 旧的实验
```

`data/`（视频有版权、体积大）、`third_party/.ytdlp/`、`third_party/*.log` 在仓库根目录的 `.gitignore` 里。GVHMR 的权重放在 submodule 里面的 `inputs/checkpoints/`，不会被提交。

---

## 1. 安装依赖

### 1.0 前提

| 项目 | 要求 |
|---|---|
| 系统 | Linux，NVIDIA GPU（GVHMR 推理约需 6 GB 显存；GMR 只用 CPU） |
| conda | `~/miniconda3`（脚本里写的是 `source ~/miniconda3/etc/profile.d/conda.sh`） |
| 网络 | GitHub、PyPI、download.pytorch.org、huggingface.co |
| 硬盘 | 第三方代码 + 权重约 7 GB |
| 工具 | `ffmpeg`、`git`、`curl` |

> ⚠ **`conda create -y` 遇到同名环境会直接覆盖它**。建环境前先 `conda env list`，确认没有 `gvhmr`、`gmr` 这两个名字。

### 1.1 nemo_v3 模型资产

已经提交在 `retarget/assets/nemo_v3/` 里（约 256 MB），clone 本仓库后就有，不需要另外拷贝：

| 目录 | 内容 |
|---|---|
| `booster_mjlab_nemov3/` | booster_mjlab 的 `robots/nemov3/`：训练用的 `xmls/nemov3.xml` + meshes + `nemov3_constants.py`（**以它为准**） |
| `nemo_v3_2_fullbody/` | URDF 描述包（`train_v2.urdf` 等 6 个变体、xacro、meshes） |
| `retarget_mjcf/` | 1.2 节生成的 `nemov3_retarget_{full,hw}.xml` |

### 1.1b 第三方代码（submodule）

GVHMR 和 GMR 是 submodule，固定在验证过的版本上；GVHMR 还要再打本项目的补丁：

```bash
git submodule update --init retarget/third_party/GVHMR retarget/third_party/GMR
git -C retarget/third_party/GVHMR apply ../../patches/gvhmr_skating.patch
```

### 1.2 生成重定向专用 MJCF（full / hw）

需要 mujoco。本机是用 `supermimic:latest` 容器跑的，装好 1.4 节的 `gmr` 环境后也可以直接在里面跑。

```bash
python retarget/scripts/make_retarget_mjcf.py
MUJOCO_GL=egl python retarget/tools/check_retarget_mjcf.py --render retarget/data/retarget_mjcf_check.png
```

产出 `retarget/assets/nemo_v3/retarget_mjcf/nemov3_retarget_{full,hw}.xml`。检查脚本应显示：
- IK 关节数：full 23、hw 18；
- 和 `nemov3.xml` 的位姿误差约 1e-7；
- 轮心离地 0.075 m。

### 1.3 GVHMR（视频 → SMPL-X）

代码已经在 1.1b 节用 submodule 拉好并打好补丁，这里只建环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n gvhmr python=3.10
conda activate gvhmr
cd retarget/third_party/GVHMR
pip install numpy==1.23.5 setuptools wheel
pip install --no-build-isolation chumpy      # 不加 --no-build-isolation 会编译失败
pip install -r requirements.txt
pip install -e .
cd -
```

补丁 `gvhmr_skating.patch` 的两处修改：
- `solver_two_view.py`：摇镜头（纯旋转）时，pycolmap 4.x 不返回相对位姿。改为从单应矩阵恢复旋转。
- `gvhmr_pipeline.py`：新增环境变量 `GVHMR_NO_STATIC_PP=1`，用来关掉会把滑行的脚钉住的后处理。

**权重**（Google Drive 原链接国内不方便，用 HF 镜像 `camenduru/GVHMR`）：

```bash
mkdir -p retarget/third_party/GVHMR/inputs/checkpoints && cd retarget/third_party/GVHMR/inputs/checkpoints
HF=https://huggingface.co/camenduru/GVHMR/resolve/main
for f in gvhmr/gvhmr_siga24_release.ckpt dpvo/dpvo.pth yolo/yolov8x.pt vitpose/vitpose-h-multi-coco.pth "hmr2/epoch=10-step=25000.ckpt"; do
  mkdir -p "$(dirname "$f")"
  for i in 1 2 3 4 5; do curl -fL --retry 5 -C - -o "$f" "$HF/$f" && break; done
done
ls -l */      # 核对大小，见下表
cd -
```

| 文件 | 大小（字节） |
|---|---|
| `gvhmr/gvhmr_siga24_release.ckpt` | 163,508,011 |
| `vitpose/vitpose-h-multi-coco.pth` | 2,549,075,546（第一次下载曾经断在 1.79 GB，**一定要核对**） |
| `hmr2/epoch=10-step=25000.ckpt` | 2,709,494,041 |
| `yolo/yolov8x.pt` | 136,867,539 |
| `dpvo/dpvo.pth` | 14,167,743 |

**人体模型**（SMPL / SMPL-X 需要在官网注册，不能从网上随便下）：本机已有一份，链接过来即可。

```bash
B=~/tasknpoint/submodules/PromptHMR/data/body_models
G=retarget/third_party/GVHMR/inputs/checkpoints/body_models
mkdir -p $G/smplx $G/smpl
ln -sf $B/smplx/SMPLX_NEUTRAL.npz $G/smplx/
ln -sf $B/smpl/SMPL_NEUTRAL.pkl $G/smpl/
```

### 1.4 GMR（SMPL-X → nemo_v3）

```bash
conda create -y -n gmr python=3.10
conda activate gmr
pip install -e retarget/third_party/GMR
pip install viser==1.0.30            # 浏览器播放器用

# SMPL-X 模型（GMR 读 assets/body_models/smplx/）
B=~/tasknpoint/submodules/PromptHMR/data/body_models/smplx
G=retarget/third_party/GMR/assets/body_models/smplx
mkdir -p $G
for f in SMPLX_NEUTRAL.npz SMPLX_NEUTRAL.pkl SMPLX_MALE.npz SMPLX_MALE.pkl SMPLX_FEMALE.npz SMPLX_FEMALE.pkl; do ln -sf $B/$f $G/; done
```

GMR 源码**不用改**：`gvhmr_to_nemov3.py` 会在运行时把 `nemo_v3_full` / `nemo_v3_hw` 注册进 GMR。

---

## 2. 常用指令

### 2.1 一条命令：视频 → 重定向 → 对比视频

```bash
retarget/scripts/run_video_to_nemov3.sh retarget/data/videos/roller40.mp4 1     # 第二个参数是 GPU 编号
```

1 分钟左右的视频大约 3 分钟跑完（GVHMR 约 1 分钟，重定向约 10 秒，两段对比视频约 2 分钟）。产出在 `retarget/data/retarget_out/`：

| 文件 | 用途 |
|---|---|
| `<name>_full.csv` | **SuperMimic 用**：无表头，30 fps，每行 = root pos 3 + root quat xyzw 4 + 32 个关节角（`NEMOV3_JOINT_ORDER`，含轮子） |
| `<name>_hw.pkl` / `<name>_full.pkl` | booster_mjlab 的 `MotionFile`（AMP 用 hw 版） |
| `<name>_{full,hw}.json` | 统计数据：缩放、根速度、轮速、侧滑、限位 |
| `<name>_compare_{full,hw}_480p.mp4` | 原视频 vs 重定向结果，1280×480 |

full / hw 的区别：full 放开腰和腕 pitch（给 tracking 和将来的硬件用）；hw 和当前硬件一样焊死（给 AMP 用）。

### 2.2 分步执行

```bash
source ~/miniconda3/etc/profile.d/conda.sh

# ① GVHMR：动态相机（默认 SimpleVO）；固定机位的视频可以加 -s
conda activate gvhmr
cd retarget/third_party/GVHMR
CUDA_VISIBLE_DEVICES=1 GVHMR_NO_STATIC_PP=1 python tools/demo/demo.py \
    --video=$PWD/../../data/videos/roller40.mp4 --output_root $PWD/../../data/gvhmr_nopp
cd -

conda activate gmr
# ② 用相机重建世界轨迹（摇镜头跟拍的滑行必做；固定机位、不跟拍时可以跳过，直接用 ① 的结果）
python retarget/scripts/gvhmr_camera_traj.py --pred retarget/data/gvhmr_nopp/roller40/hmr4d_results.pt \
    --out retarget/data/gvhmr_cam/roller40/hmr4d_results.pt --pos_sigma 8 --video retarget/data/videos/roller40.mp4
#   --video                 考虑变焦：用背景估计镜头的缩放和转动（跟拍时一定要加，否则人看起来几乎不动，MILESTONE 18）
#   --no_flip_repair        不修复人体朝向的短暂前后翻转（默认会修，MILESTONE 18.5）

# ③ 重定向（默认同时出 full 和 hw）
python retarget/scripts/gvhmr_to_nemov3.py --gvhmr_pred_file retarget/data/gvhmr_cam/roller40/hmr4d_results.pt \
    --out_dir retarget/data/retarget_out
#   --variant full          只出 full 版
#   --raw_heading           轮子朝向直接用估计的脚尖点（不按速度方向融合）
#   --leg_reach 0.95        人腿伸直时对应机器人腿长的比例（默认 0.95；调小会更蹲，太小会让膝盖弯到底）
#   --no_balance            不做单轮平衡（默认会把单轮时段的重心移到支撑轮上方，MILESTONE 19.2）

# ④ 对比视频
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python retarget/tools/render_compare.py \
    --motion retarget/data/retarget_out/roller40_full.pkl --video retarget/data/videos/roller40.mp4 \
    --gvhmr_pred retarget/data/gvhmr_cam/roller40/hmr4d_results.pt \
    --out retarget/data/retarget_out/roller40_compare_full_480p.mp4 --title "(full)"
#   --seconds 20            只渲染前 20 秒（调参时用）
```

### 2.3 播放重定向结果

**看对比视频**：

```bash
ffplay retarget/data/retarget_out/roller40_compare_full_480p.mp4
```

**浏览器交互播放**（可以切换 full / hw，原视频画面同步显示）：

```bash
conda activate gmr
python retarget/tools/viser_play_motion.py retarget/data/retarget_out/roller40_hw.pkl \
    --video retarget/data/videos/roller40.mp4 --port 8080
```

然后浏览器打开 `http://<服务器IP>:8080`（服务器 IP 可用 `hostname -I` 查）。浏览器提示 WebGL 不可用时，到 `chrome://flags` 打开 *Override software rendering list*。

后台常驻和停止：

```bash
setsid nohup python retarget/tools/viser_play_motion.py retarget/data/retarget_out/roller40_full.pkl \
    retarget/data/retarget_out/roller40_hw.pkl --video retarget/data/videos/roller40.mp4 --port 8080 \
    > retarget/data/viser_8080.log 2>&1 < /dev/null &
echo $! > retarget/data/viser_8080.pid
kill $(cat retarget/data/viser_8080.pid)      # 停止（不要用 pkill -f：同一行命令里出现这个名字时会连自己的 shell 一起结束）
```

### 2.4 找视频

**Wikimedia Commons**（公开 API，CC 授权，推荐）。例如本次用的花样轮滑片段：

```bash
mkdir -p retarget/data/videos/commons && cd retarget/data/videos/commons
UA="nemo-retarget-research/0.1 (research script)"      # 不要在 User-Agent 里放个人信息
F="EVD-patinajeartistico-007.ogv"
U=$(curl -s -A "$UA" "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url&titles=File:$F" \
    | python3 -c "import json,sys; print(list(json.load(sys.stdin)['query']['pages'].values())[0]['imageinfo'][0]['url'])")
curl -s -A "$UA" -L -o $F "$U"
cd ..
ffmpeg -ss 10 -t 40 -i commons/$F -vf "scale=640:448:flags=lanczos,fps=30" -an -c:v libx264 -crf 16 roller40.mp4   # 原片 320x224，放大 2 倍
cd -
retarget/scripts/run_video_to_nemov3.sh retarget/data/videos/roller40.mp4 1
```

**固定机位**的视频（例如舞蹈室）：GVHMR 用 `-s`，并且**保留**它的静止关节后处理；跳过 `gvhmr_camera_traj.py`，直接把 `retarget/data/gvhmr/<name>/hmr4d_results.pt` 交给 `gvhmr_to_nemov3.py`。

**B 站**：

YouTube 在本服务器上要求登录做人机验证，所以改用 B 站。不登录最高只有 360p。

```bash
python3 -m venv retarget/third_party/.ytdlp && retarget/third_party/.ytdlp/bin/pip install yt-dlp
Y=retarget/third_party/.ytdlp/bin/yt-dlp
mkdir -p retarget/data/videos && cd retarget/data/videos
$Y -f "bv*[height<=720]+ba/b" --merge-output-format mp4 -o "bili_p6.%(ext)s" "https://www.bilibili.com/video/av34903340?p=6"
ffmpeg -ss 56 -t 60 -i bili_p6.mp4 -c:v libx264 -crf 18 -an roller40.mp4      # 第 14 节：一级 60 秒（有压步）
$Y -f "bv*[height<=720]+ba/b" --merge-output-format mp4 -o "bili_p1.%(ext)s" "https://www.bilibili.com/video/av34903340?p=1"
ffmpeg -ss 28 -t 13.5 -i bili_p1.mp4 -c:v libx264 -crf 18 -an basic_a.mp4     # 第 15 节：基础级蹬冰（无交叉）
ffmpeg -ss 60 -t 16 -i bili_p1.mp4 -c:v libx264 -crf 18 -an basic_b.mp4       # 第 15 节：基础级蹬冰 + 双足滑行
ffmpeg -i bili_p6.mp4 -vf "fps=1/20,scale=320:-1,tile=6x6" -frames:v 1 sheet.jpg   # 先拼一张缩略图挑片段
cd -
```

选片段时注意：避开**慢动作**（速度会算错；如果是整段固定倍速的教学视频，可以用 `setpts` 还原）、**剪辑切换**和**特写**。B 站站内搜索不登录会返回 412，**不要去绕过它**，改用网页搜索拿到视频号，或者用 Commons。

---

## 3. 用 SuperMimic 训练 nemov3

SuperMimic 的代码支持 G1 和 nemov3 两种机型，用 `SUPERMIMIC_ROBOT=g1|nemov3`（或 `--robot`）选择，默认是 G1。

```bash
# ① 仿真资产（一次即可）：焊死当前硬件不动的关节，删掉 MJCF 自带的 actuator，再转成 USD
docker/run_container.sh 1 128g                      # 主机 8080 被占用时：VISER_PORT=8090 docker/run_container.sh 1 128g
docker exec supermimic bash -c "cd /workspace/supermimic && python retarget/scripts/make_sim_mjcf.py && \
    python tools/mjcf_to_usd.py retarget/assets/nemo_v3/sim/nemov3_sim_hw.xml retarget/assets/nemo_v3/sim/usd \
    --hull_vertices '.*_wheel_collision=1024'"   # 轮子凸包上限，否则 Newton 只留 64 个顶点（MILESTONE 19.1）

# ② 训练动作（以一段视频 roller40 为例）：HOME 静止 1 s → 过渡 1 s → 动作 → 过渡 1 s → HOME 静止 1 s
#    --end_s 是动作取到视频的第几秒；视频开头人没有全身入镜时，用 --start_s 裁掉；--blend_s 改过渡长度
conda activate gmr
python retarget/scripts/make_supermimic_clip.py --src retarget/data/retarget_out/roller40_hw.csv --end_s 30.57 \
    --out retarget/data/supermimic_motions/roller32/roller32.csv

# 训练用的动作目录：一个目录里放几段 csv，就同时训练几段（策略靠目标里的未来参考帧区分是哪一段）。
# 多段视频可以各自按上面的流程单独导出，再把它们的 csv 都复制到同一个目录一起训练，例如：
mkdir -p retarget/data/supermimic_motions/nemov3_multi
cp retarget/data/supermimic_motions/roller32/roller32.csv retarget/data/supermimic_motions/nemov3_multi/
#   cp retarget/data/supermimic_motions/<另一段>/<另一段>.csv retarget/data/supermimic_motions/nemov3_multi/
# robots/nemov3.py 的默认动作目录就是 nemov3_multi；想只训练其中一段，用 --motion_dir 指向那一段所在的目录。

# ③ 自检（FK 与仿真器应一致到约 1e-5）
docker exec -e SUPERMIMIC_ROBOT=nemov3 supermimic python tools/test_env.py --num_envs 16

# ④ 训练 + 评估监视器（不带 --motion_dir 就用默认的 nemov3_multi，目录里放几段就一起训练几段；3000 轮之前每 1000 轮、之后每 500 轮存一个 checkpoint，每个都会被评估）
#    --exp_name 同一组实验的目录；每次训练在底下新建 <YYYY-mm-dd_HH-MM-SS>_<run_name>/（和 mjlab 一样），
#    logs/<exp_name>/latest 指向最新的一次。--run_name 写这次改了什么，例如 exp1_roller32_roller_house；编号只是自己的惯例
docker exec supermimic tmux new -d -s nemov3_train "cd /workspace/supermimic && SUPERMIMIC_ROBOT=nemov3 python supermimic/train.py \
    --exp_name nemov3_multi --run_name exp1_roller32_roller_house --ckpt_every 1000 --ckpt_every_late 500 --ckpt_late_from 3000 2>&1 \
    | tee -a logs/nemov3_multi_train.log"
docker exec supermimic tmux new -d -s nemov3_eval "cd /workspace/supermimic && sleep 60 && SUPERMIMIC_ROBOT=nemov3 python supermimic/evaluate.py \
    --run_dir logs/nemov3_multi/latest --watch 2>&1 | tee -a logs/nemov3_multi/latest/eval_stdout.log"
cat logs/nemov3_multi/latest/eval.csv

# ⑤ 录制某个 checkpoint 的评估轨迹，用 viser 查看（容器内 8080 = 主机 VISER_PORT）
R=logs/nemov3_multi/2026-10-25_22-58-12_exp1_roller32_roller_house     # 或 logs/nemov3_multi/latest
docker exec -e SUPERMIMIC_ROBOT=nemov3 supermimic python supermimic/evaluate.py --run_dir $R \
    --ckpt $R/ckpt_006000.pt --record $R/traj_best.npz
docker exec -d supermimic bash -c "cd /workspace/supermimic && SUPERMIMIC_ROBOT=nemov3 python tools/viser_play.py \
    $R/traj_best.npz --eval $R/ckpt_006000_eval.json --port 8080"

# ⑥ TensorBoard 看训练曲线：训练只写 CSV，先在容器里转成事件文件（--watch 持续追加新行）
#    logs/ 归容器的 root 所有，所以转换在容器里跑；TensorBoard 在主机上用任一装了 tensorboard 的环境启动
docker exec -d supermimic bash -c "cd /workspace/supermimic && python tools/csv_to_tensorboard.py logs/nemov3_multi/latest --watch"
tensorboard --logdir logs --bind_all --port 6006     # 浏览器打开 http://<服务器 IP>:6006
```

仓库根目录 README.md 里 G1 的命令都不用改：不设 `SUPERMIMIC_ROBOT` 时就是 G1，行为和以前完全一样。

## 4. 已知限制

完整原因和数据见 [MILESTONE.md](MILESTONE.md) 第 14 节。

- **SuperMimic 同时支持 G1 和 nemov3**：机型相关的常数都在 `supermimic/robot_spec.py` 和 `supermimic/robots/`，`motion.py` 读 `7 + 关节数` 列（G1 是 29，nemov3 是 20，列数不对会直接报错）。nemov3 的默认动作目录是 `retarget/data/supermimic_motions/nemov3_multi/`（见 §3）。
- **视频质量决定重定向质量**：GVHMR 对低分辨率、会变焦的视频估计有限。最早的 360p 教学视频（skate60）侧滑平均 0.33–0.37、速度曲线有尖峰；现在变焦由考虑变焦的相机轨迹处理，侧滑平均：roller40 约 0.07–0.11、roller_house 约 0.14–0.22。GVHMR 还会偶尔把人体朝向前后翻转、把左右腿认反，流程里已自动修复（`gvhmr_camera_traj.py`，MILESTONE 第 18、20 节）。正式数据建议自己拍：固定机位或平稳跟拍、高帧率、高分辨率、一个连续镜头、单人全身入镜（MILESTONE 第 7 节）。
- **压步、后仰在 nemo_v3 上做不到**：髋 roll 范围是 −10° ~ +60°，只能内收 10°（腿长约 0.5 m，轮子最多靠到中线，交叉不过去）；腰（只有全身版 `full` 有）pitch 0 ~ 30°，只能前倾、不能后仰，roll ±10°，yaw ±80°。**当前硬件版 `hw` 的腰、头、推杆、腕 pitch 是焊死的**，腰完全不能动。
