#!/usr/bin/env python3
"""Score checkpoints (each alone and averaged) on preprocessed cases and write the ensemble STLs.

python scripts/evaluate.py --records data/cache/records.json --models models/ --cases case05,case06 --out eval/
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arteryseg import mesh, models as M, preprocess as P  # noqa: E402
from arteryseg.inference import predict_prob  # noqa: E402
from arteryseg.metrics import scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--models", required=True, help="folder of .pt files or one .pt")
    ap.add_argument("--cases", required=True, help="comma-separated case ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--no-stl", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads(Path(a.records).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = M.find_checkpoints(a.models)
    nets = M.load_models(paths, dev)
    summary = {}
    for case in a.cases.split(","):
        t0 = time.time()
        r = records[case]
        x = np.load(r["image"], mmap_mode="r")
        y = np.load(r["label"], mmap_mode="r")
        probs = {p.stem: predict_prob([net], x, dev, overlap=a.overlap) for p, net in zip(paths, nets)}
        ens = np.mean(list(probs.values()), 0)
        row = {k: scores(v >= 0.5, y) for k, v in probs.items()}
        if len(probs) > 1:
            row["ensemble"] = scores(ens >= 0.5, y)
        summary[case] = row
        if not a.no_stl:
            mesh.stl_from_prob(ens, np.array(r["affine"]), out / f"{case}.stl", P.SPACING)
        print(f"[{case}] " + "  ".join(f"{k}={v['dice']:.4f}" for k, v in row.items()) +
              f" ({time.time() - t0:.0f}s)", flush=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
