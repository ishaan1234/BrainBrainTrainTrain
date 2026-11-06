# arteryseg

Segments the brain arteries in a head CT angiography scan and exports them as an STL surface.
Input is a DICOM folder, a `.zip` of one, or a NIfTI file. Output is `arteries.stl` (mm, LPS
coordinates, so it opens aligned with the CT in 3D Slicer) and a binary mask on the original scan grid.

> **Research use only. Not a medical device and not validated for diagnosis.** The released models were
> trained on 6 labeled scans; see [Accuracy](#accuracy).

## Install

```bash
git clone https://github.com/ishaan1234/BrainBrainTrainTrain && cd BrainBrainTrainTrain
pip install -e .           # or: pip install -e ".[dev]" to run the tests
```

The weights in `models/` are split into parts under 50 MB (`*.pt.part00`, `part01`, ...) to stay under
GitHub's file size limit. They are reassembled in memory when loaded, so there is no extra step.

## Segment a scan

```bash
python scripts/predict.py /path/to/dicom_folder --out result/
```

Outputs in `result/`:
- `arteries.stl`
- `prediction.nii.gz`: binary mask on the original grid
- `probability.nii.gz`: soft probabilities, only with `--save-prob`

The two DynUNet checkpoints in `models/` are averaged. Point `--models` at one checkpoint (e.g.
`models/dynunet_s1.pt`, which also resolves its parts) to run one model at half the time.

| Option | Effect |
|---|---|
| `--overlap 0.25` | sliding-window overlap (default 0.5); 0.25 is about 2x faster |
| `--tta` | add a left-right flip (2x slower) |
| `--min-component-mm3 50` | drop disconnected specks smaller than this |
| `--threshold 0.5` | probability cutoff |
| `--cpu`, `--threads N` | force CPU inference |
| `--no-crop` | don't crop to the body (the crop only skips air) |

A GPU takes under a minute per scan. On CPU, expect several minutes per model and about 4 to 8 GB of RAM.

From Python:

```python
from arteryseg.pipeline import segment
segment("scan_dicom/", "models/", "result/")
```

## How it works

1. **Load and resample.**
   - DICOM is read with SimpleITK; the series with the most slices wins.
   - The scan is reoriented to RAS and resampled linearly to 0.5 mm isotropic.
2. **Normalize.** Three input channels come from HU windows:
   - vessel: −100 to 700 HU
   - bone: 0 to 2000 HU
   - wide: −1024 to 3000 HU

   The bone channel restores what the vessel window clips, so arteries next to the skull base stay
   separable from bone.
3. **Infer.** Sliding-window inference with 160³ windows and Gaussian blending. The network math is fp32.
4. **Export.**
   - Probabilities are resampled linearly to the original grid, then thresholded.
   - The STL comes from marching cubes on the smooth probability map, after removing small
     disconnected components.

## Accuracy

The architectures were compared by training on 4 scans and testing on 2 patients the models never saw
(Dice, higher is better):

| Model | Patient A | Patient B |
|---|---|---|
| **DynUNet** (nnU-Net style, deep supervision) | **0.77** | **0.66** |
| SegResNet | 0.75 | 0.63 |
| SwinUNETR | 0.71 | 0.57 |

The released models are two DynUNets trained on all 6 scans with different seeds. There is no held-out
scan left to measure them on, so treat the numbers above as the expected accuracy on a new scan.

Known limitations:
- The training labels cover intracranial arteries only. On scans that include the neck, neck vessels
  the model finds count as false positives.
- Thick-slice scans (≥1 mm) segment noticeably worse than 0.3 to 0.6 mm scans.

## Train your own

```bash
cp cases.example.json cases.json        # list your CTs and binary artery labels
python scripts/prepare_data.py --cases cases.json --out data/cache
python scripts/train.py --records data/cache/records.json --arch dynunet \
    --train case01,case02,case03,case04 --val case05,case06 --minutes 90 --out runs/dynunet
python scripts/evaluate.py --records data/cache/records.json --models runs/dynunet/final.pt \
    --cases case05,case06 --out eval/
```

- Labels may be on a cropped sub-grid, e.g. 3D Slicer segment exports; they are placed using their affines.
- `train.py` uses GPU-side augmentation: rotation, scaling, left-right flip, contrast, gamma, noise,
  and thick-slice simulation.
- It runs on a wall-clock budget with a time-based cosine LR schedule.
- Several runs can share one large GPU.
- [docs/training.md](docs/training.md) describes how the released models were trained on a Colab GPU
  from the command line.

## Tests

```bash
pytest
```

The tests use synthetic volumes only. They cover preprocessing, cropped-label placement, DICOM/NIfTI
equivalence, STL geometry, augmentation, all three architectures, and an end-to-end pipeline run.

## Layout

```
arteryseg/   io.py  preprocess.py  models.py  inference.py  mesh.py  metrics.py  training.py  pipeline.py
scripts/     predict.py  prepare_data.py  train.py  evaluate.py
models/      dynunet_s1.pt.part00-02  dynunet_s2.pt.part00-02
tests/       test_core.py
docs/        training.md
```
