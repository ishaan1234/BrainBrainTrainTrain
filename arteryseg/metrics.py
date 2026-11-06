import numpy as np


def scores(pred, label) -> dict:
    """Dice, precision and recall of two binary volumes."""
    pred = np.asarray(pred, bool)
    label = np.asarray(label, bool)
    tp = int(np.count_nonzero(pred & label))
    p, g = int(pred.sum()), int(label.sum())
    return dict(dice=2 * tp / (p + g) if p + g else 1.0, precision=tp / p if p else 0.0,
                recall=tp / g if g else 1.0, predicted_voxels=p, ground_truth_voxels=g)
