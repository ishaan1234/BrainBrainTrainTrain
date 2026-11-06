# How the released models were trained

## Hardware

Training ran on a Google Colab G4 runtime: one RTX PRO 6000 Blackwell (96 GB), 48 vCPUs and 176 GB RAM.
It was driven from a terminal with the [Colab CLI](https://github.com/googlecolab/google-colab-cli).
All math was fp32, with no AMP and TF32 off.

## Setup

```bash
pip install google-colab-cli
colab --auth oauth2 new -s train --gpu G4         # or A100 / H100 if your account has them
colab --auth oauth2 upload -s train arteryseg ... # or git clone the repo on the VM
colab --auth oauth2 install -s train monai==1.5.1 SimpleITK einops
```

`colab exec` blocks until the code returns and can time out. Long jobs should be started detached from a
short `exec`, and then polled:

```python
# launch.py, sent with: colab --auth oauth2 exec -s train -f launch.py
import subprocess
subprocess.Popen("cd /content/artery-seg && nohup python -u scripts/train.py ... > /content/train.log 2>&1 &", shell=True)
```

Stop the runtime with `colab --auth oauth2 stop -s train` when done; it bills until stopped.

## Runs

**Phase 1: architecture comparison.**
- Trained on 4 scans, validated on 2 held-out patients.
- Three processes shared the GPU, using about 84 of 96 GB at 100% SM utilization.
- 85 minutes each:

| Arch | Patch | Batch | VRAM | Steps |
|---|---|---|---|---|
| dynunet | 160³ | 2 | 19 GB | ~1900 |
| segresnet | 160³ | 2 | 20 GB | ~1650 |
| swinunetr | 128³ | 2 | 40 GB | ~1800 |

**Phase 2: release models.**
- Two DynUNets (seeds 2026 and 7) plus one SegResNet.
- All 6 scans, 90 minutes each, about 2100 steps for the DynUNets.
- Only the DynUNets are released.

Total cost was about 36 Colab compute units, including preprocessing and evaluation.

## Notes

- Preprocessing all 6 scans in parallel (`prepare_data.py`) took about 1 minute on 48 cores.
- In fp32 the GPU was compute-bound: about 2.5 s per step per run with three runs sharing it.
  Enabling TF32 or AMP would be several times faster, at reduced numeric precision.
