#!/usr/bin/env python3
"""Train one network on a wall-clock budget. Warmup + cosine LR runs on the elapsed-time fraction,
so the run always finishes on time. All math is fp32 (no AMP, TF32 off).

python scripts/train.py --records data/cache/records.json --arch dynunet --train case01,case02 \
    --val case03 --minutes 90 --patch 160 --batch 2 --out runs/dynunet

Several runs can share one large GPU: start them as separate processes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arteryseg import models as M  # noqa: E402
from arteryseg.inference import predict_prob  # noqa: E402
from arteryseg.metrics import scores  # noqa: E402
from arteryseg.training import Crops, cosine_lr, crop_size, deep_supervision_loss, gpu_augment, seg_loss  # noqa: E402

DEFAULT_LR = dict(dynunet=1e-3, segresnet=1e-3, swinunetr=2e-4)


def evaluate(model, roi, records, cases, device, overlap):
    model.eval()
    res = {}
    for c in cases:
        r = records[c]
        x = np.load(r["image"], mmap_mode="r")
        y = np.load(r["label"], mmap_mode="r")
        res[c] = scores(predict_prob([(model, roi)], x, device, overlap=overlap) >= 0.5, y)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--arch", required=True, choices=M.ARCHS)
    ap.add_argument("--train", required=True, help="comma-separated case ids")
    ap.add_argument("--val", default="", help="comma-separated held-out case ids (optional)")
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--patch", type=int, default=160, help="cube edge; must be divisible by 32")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every-min", type=float, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for a quick smoke test")
    a = ap.parse_args()
    if a.patch % 32:
        sys.exit("--patch must be divisible by 32")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(a.records).read_text())
    train, val = a.train.split(","), [v for v in a.val.split(",") if v]
    lr_max = a.lr or DEFAULT_LR[a.arch]
    roi = [a.patch] * 3
    (out / "config.json").write_text(json.dumps(dict(vars(a), lr_max=lr_max), indent=2))

    loader = DataLoader(Crops([records[c] for c in train], crop_size(a.patch), a.seed), batch_size=a.batch,
                        num_workers=a.workers, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0,
                        prefetch_factor=4 if a.workers else None)
    model = M.build_model(a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4)
    print(f"{a.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, patch {a.patch}, "
          f"batch {a.batch}, train {train}, val {val}", flush=True)

    budget, t0 = a.minutes * 60, time.time()
    last_eval, tick, step, hist = t0, t0, 0, []
    log = open(out / "log.jsonl", "a")
    model.train()
    for xb, yb in loader:
        frac = (time.time() - t0) / budget
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(frac, lr_max)
        x, y = gpu_augment(xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True), a.patch)
        z = model(x)
        loss, dice = deep_supervision_loss(z, y) if z.dim() == 6 else seg_loss(z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            row = dict(step=step, min=round((time.time() - t0) / 60, 2), lr=cosine_lr(frac, lr_max),
                       loss=round(loss.item(), 4), soft_dice_loss=round(dice.item(), 4),
                       grad_norm=round(float(grad_norm), 3),
                       sec_per_step=round((time.time() - tick) / (1 if step == 1 else 50), 3),
                       vram_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None)
            tick = time.time()
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if val and time.time() - last_eval > a.eval_every_min * 60:
            res = evaluate(model, roi, records, val, dev, overlap=0.25)
            last_eval = time.time()
            row = dict(step=step, min=round((last_eval - t0) / 60, 2), val=res,
                       mean_dice=float(np.mean([m["dice"] for m in res.values()])))
            print("EVAL", json.dumps(row), flush=True)
            hist.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            M.save_checkpoint(out / "last.pt", model, a.arch, roi, steps=step)

    M.save_checkpoint(out / "final.pt", model, a.arch, roi, steps=step, n_train_cases=len(train))
    final = dict(arch=a.arch, steps=step, minutes=round((time.time() - t0) / 60, 1), history=hist)
    if val:
        final["final_val"] = evaluate(model, roi, records, val, dev, overlap=0.5)
        final["final_mean_dice"] = float(np.mean([m["dice"] for m in final["final_val"].values()]))
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print("DONE", json.dumps(final.get("final_val")), flush=True)


if __name__ == "__main__":
    main()
