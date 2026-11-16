# Training on HiPerGator

This trains the DynUNet with the same recipe as the released models, on every labeled case:
- fp32, with no AMP and TF32 off
- 160³ patches, batch 2, AdamW
- GPU augmentation
- warmup + cosine LR schedule over a fixed time budget

The chain is: download → preprocess → train one model per seed in parallel → export the models and
an STL per case.

## 1. One-time setup (login node)

```bash
ssh <gatorlink>@hpg.rc.ufl.edu
mkdir -p /blue/<group>/$USER/arteryseg && cd /blue/<group>/$USER/arteryseg
git clone -b hipergator https://github.com/ishaan1234/BrainBrainTrainTrain.git
cd BrainBrainTrainTrain
nano hpc/hipergator/config.sh          # set GROUP (run `id -gn` if unsure); pick the GPU type
bash hpc/hipergator/setup.sh           # conda env in /blue, PyTorch CUDA 12.8, this package
```

## 2. Data (login node)

```bash
bash hpc/hipergator/get_data.sh "https://drive.google.com/drive/folders/<folder-id>"
```

This downloads the folder to `$WORK/raw`, unzips any case zips, and writes `$WORK/cases.json`.

CT and label are told apart by their voxel values, not their names. Folders with no CT are listed as
skipped. Read that list before training.

## 3. Train (submits everything)

```bash
bash hpc/hipergator/run_all.sh
squeue -u $USER
tail -f /blue/<group>/$USER/arteryseg/artery-train-<jobid>_0.log
```

| Job | Resources | What it does |
|---|---|---|
| `artery-prep` | CPU, about 10 min | resamples each case to 0.5 mm and writes `$WORK/cache` |
| `artery-train` | 1 GPU per seed | trains `TRAIN_MINUTES` per model (default 24 h), with seeds 2026 and 7 in parallel |
| `artery-export` | 1 GPU | runs after training; writes `$WORK/models` and `$WORK/export` |

Export output:
- `$WORK/models`: checkpoints, split into GitHub-sized parts
- `$WORK/export`: one STL per case plus `metrics.json`

**Restarts.** Training saves its full state every 10 minutes. If a job hits its time limit, it
requeues itself and continues where it stopped, and the LR schedule picks up at the same point.
You can also cancel it and run `run_all.sh --skip-prep` later to resume.

**Speed.** In fp32, an A100 is much slower than a B200, roughly 4x. Use B200 if your group has it:
set `GPU_PARTITION`/`GPU_GRES` in `config.sh` and check `sinfo -s` for the partition names.

The time budget is fixed, so a slower GPU gets fewer steps in the same 24 h. Raise `TRAIN_MINUTES`
for more steps; each job runs 20 h and requeues itself to continue. Check the log's `sec_per_step` after
the first hour to see what a day buys you.

## 4. Use the new models

Segment a scan on HiPerGator:

```bash
source hpc/hipergator/config.sh
sbatch --account=$GROUP --qos=$QOS --partition=$GPU_PARTITION --gres=gpu:$GPU_GRES \
  hpc/hipergator/predict.sbatch /path/to/dicom_folder /path/to/output
```

To ship the new models in the repo, replace the old ones and push:

```bash
git rm models/*.part*
cp $WORK/models/*.part* models/
git add models && git commit -m "DynUNet trained on all cases" && git push
```

The `.part` files are under 50 MB each, so they push without Git LFS.

## Settings (`config.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `GROUP`, `QOS` | `gaitor_club` | your allocation; `QOS=$GROUP-b` uses burst |
| `WORK` | `/blue/$GROUP/$USER/arteryseg` | everything lives here (not `/home`, which is small) |
| `GPU_PARTITION`, `GPU_GRES` | `hpg-b200`, `b200:1` | GPU type |
| `TRAIN_MINUTES` | 1440 | training time per model |
| `SEEDS` | `2026 7` | one model per seed; more seeds give a bigger ensemble |
| `PATCH`, `BATCH` | 160, 2 | same as the released models |
