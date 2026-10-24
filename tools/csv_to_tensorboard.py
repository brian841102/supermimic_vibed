"""Export a run's CSV logs (train_log.csv, eval.csv) to TensorBoard event files in <run_dir>/tb/.

Training only writes CSV; this converts it so TensorBoard can show the curves. With --watch it keeps polling and
appends new rows as training goes on. Runs on the host in any env with torch + tensorboard:
    python tools/csv_to_tensorboard.py logs/<run> [--watch]
    tensorboard --logdir logs --bind_all --port 6006
"""

import argparse
import csv
import os
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

SOURCES = (("train_log.csv", "train", "iteration"), ("eval.csv", "eval", "iteration"))


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--watch", action="store_true", help="keep appending new rows")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between polls with --watch")
    a = ap.parse_args()
    out = os.path.join(a.run_dir, "tb")
    shutil.rmtree(out, ignore_errors=True)  # rebuilt from the CSVs every start
    w = SummaryWriter(out)
    done = {name: 0 for name, _, _ in SOURCES}
    while True:
        for name, prefix, step_key in SOURCES:
            rows = read(os.path.join(a.run_dir, name))
            for row in rows[done[name] :]:
                if step_key not in row:
                    continue
                step = int(float(row[step_key]))
                for k, v in row.items():
                    if k == step_key:
                        continue
                    try:
                        w.add_scalar(f"{prefix}/{k}", float(v), step)
                    except (TypeError, ValueError):
                        pass
            done[name] = len(rows)
        w.flush()
        if not a.watch:
            break
        time.sleep(a.every)
    w.close()
    print("wrote", out, {k: v for k, v in done.items()})


if __name__ == "__main__":
    main()
