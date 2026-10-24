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
├── configs/smplx_to_nemo_v3.json   # GMR 的 IK 配置（full / hw 共用）
├── patches/gvhmr_skating.patch     # 对 GVHMR 的补丁
├── scripts/                   # 主流程（产出数据）：生成模型、相机轨迹、重定向、一键脚本
├── tools/                     # 检查与可视化（不产出训练数据）：模型检查、对比视频、浏览器播放
├── assets/nemo_v3/            # nemo_v3 模型 + 生成的重定向专用 MJCF
├── third_party/               # submodule：GVHMR、GMR（权重和环境不在仓库里）
└── data/          ⚠ 不提交    # 视频、GVHMR 结果、重定向结果、对比视频
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
retarget/scripts/run_video_to_nemov3.sh retarget/data/videos/skate60.mp4 1     # 第二个参数是 GPU 编号
```

60 秒视频大约 3 分钟跑完（GVHMR 约 1 分钟，重定向约 10 秒，两段对比视频约 2 分钟）。产出在 `retarget/data/retarget_out/`：

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
    --video=$PWD/../../data/videos/skate60.mp4 --output_root $PWD/../../data/gvhmr_nopp
cd -

conda activate gmr
# ② 用相机重建世界轨迹（摇镜头跟拍的滑行必做；固定机位、不跟拍时可以跳过，直接用 ① 的结果）
python retarget/scripts/gvhmr_camera_traj.py --pred retarget/data/gvhmr_nopp/skate60/hmr4d_results.pt \
    --out retarget/data/gvhmr_cam/skate60/hmr4d_results.pt --pos_sigma 8

# ③ 重定向（默认同时出 full 和 hw）
python retarget/scripts/gvhmr_to_nemov3.py --gvhmr_pred_file retarget/data/gvhmr_cam/skate60/hmr4d_results.pt \
    --out_dir retarget/data/retarget_out
#   --variant full          只出 full 版
#   --raw_heading           轮子朝向直接用估计的脚尖点（不按速度方向融合）

# ④ 对比视频
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python retarget/tools/render_compare.py \
    --motion retarget/data/retarget_out/skate60_full.pkl --video retarget/data/videos/skate60.mp4 \
    --gvhmr_pred retarget/data/gvhmr_cam/skate60/hmr4d_results.pt \
    --out retarget/data/retarget_out/skate60_compare_full_480p.mp4 --title "(full)"
#   --seconds 20            只渲染前 20 秒（调参时用）
```

### 2.3 播放重定向结果

**看对比视频**：

```bash
ffplay retarget/data/retarget_out/skate60_compare_full_480p.mp4
```

**浏览器交互播放**（可以切换 full / hw，原视频画面同步显示）：

```bash
conda activate gmr
python retarget/tools/viser_play_motion.py retarget/data/retarget_out/skate60_full.pkl retarget/data/retarget_out/skate60_hw.pkl \
    --video retarget/data/videos/skate60.mp4 --port 8080
```

然后浏览器打开 `http://<服务器IP>:8080`（服务器 IP 可用 `hostname -I` 查）。浏览器提示 WebGL 不可用时，到 `chrome://flags` 打开 *Override software rendering list*。

后台常驻和停止：

```bash
setsid nohup python retarget/tools/viser_play_motion.py retarget/data/retarget_out/skate60_full.pkl \
    retarget/data/retarget_out/skate60_hw.pkl --video retarget/data/videos/skate60.mp4 --port 8080 \
    > retarget/data/viser_8080.log 2>&1 < /dev/null &
pkill -f viser_play_motion.py      # 停止
```

### 2.4 找视频（B 站）

YouTube 在本服务器上要求登录做人机验证，所以改用 B 站。不登录最高只有 360p。

```bash
python3 -m venv retarget/third_party/.ytdlp && retarget/third_party/.ytdlp/bin/pip install yt-dlp
Y=retarget/third_party/.ytdlp/bin/yt-dlp
mkdir -p retarget/data/videos && cd retarget/data/videos
$Y -f "bv*[height<=720]+ba/b" --merge-output-format mp4 -o "bili_p6.%(ext)s" "https://www.bilibili.com/video/av34903340?p=6"
ffmpeg -ss 56 -t 60 -i bili_p6.mp4 -c:v libx264 -crf 18 -an skate60.mp4      # 截取 60 秒正常速度的片段
ffmpeg -i bili_p6.mp4 -vf "fps=1/20,scale=320:-1,tile=6x6" -frames:v 1 sheet.jpg   # 先拼一张缩略图挑片段
cd -
```

选片段时注意：避开**慢动作**（速度会算错）、**剪辑切换**和**特写**。B 站的搜索 API 不登录会返回 412，要先用网页搜索拿到视频号。

---

## 3. 已知限制

完整原因和数据见 [MILESTONE.md](MILESTONE.md) 第 14 节。

- SuperMimic 代码还是 G1 专用的（`supermimic/motion.py` 写死读 29 列关节），要移植到 nemov3 才能直接训练这份 CSV。
- 360p、会变焦的教学视频估计质量有限：侧滑 0.33–0.37，速度曲线有尖峰。正式数据建议用固定机位、高帧率、更高分辨率自己拍（MILESTONE.md 第 7 节）。
- 压步、后仰在 nemo_v3 上本来就做不到：髋只能内收 10°，腰只能前倾。
