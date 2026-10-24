#!/bin/bash
# Video -> GVHMR -> camera-based trajectory -> GMR (nemo_v3 full + hw) -> comparison video. See retarget/MILESTONE.md section 14.
# Usage: retarget/scripts/run_video_to_nemov3.sh <video.mp4> [GPU index, default 1]
# Needs the conda envs `gvhmr` and `gmr` (retarget/README.md) and GVHMR with retarget/patches/gvhmr_skating.patch applied.
set -euo pipefail
VIDEO=$(realpath "$1")
GPU=${2:-1}
RT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=$(basename "${VIDEO%.*}")
D="$RT/data"
source ~/miniconda3/etc/profile.d/conda.sh

# 1. GVHMR (moving camera: SimpleVO; no static-joint post-processing, it pins gliding feet)
conda activate gvhmr
(cd "$RT/third_party/GVHMR" && CUDA_VISIBLE_DEVICES=$GPU GVHMR_NO_STATIC_PP=1 python tools/demo/demo.py --video="$VIDEO" --output_root "$D/gvhmr_nopp")

# 2. world trajectory from camera rotation + in-camera position (glides), 3. retarget, 4. comparison videos
conda activate gmr
python "$RT/scripts/gvhmr_camera_traj.py" --pred "$D/gvhmr_nopp/$NAME/hmr4d_results.pt" --out "$D/gvhmr_cam/$NAME/hmr4d_results.pt" --pos_sigma 8
python "$RT/scripts/gvhmr_to_nemov3.py" --gvhmr_pred_file "$D/gvhmr_cam/$NAME/hmr4d_results.pt" --out_dir "$D/retarget_out" --name "$NAME"
for V in full hw; do
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python "$RT/tools/render_compare.py" --motion "$D/retarget_out/${NAME}_$V.pkl" --video "$VIDEO" \
    --gvhmr_pred "$D/gvhmr_cam/$NAME/hmr4d_results.pt" --out "$D/retarget_out/${NAME}_compare_${V}_480p.mp4" --title "($V)"
done
echo "done: $D/retarget_out/${NAME}_{full,hw}.{csv,pkl,json} and ${NAME}_compare_{full,hw}_480p.mp4"
