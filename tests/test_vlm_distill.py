"""The mixle structured classifier behind the VLM re-representation demo (tested without the heavy VLM).

The full ``examples/distill_vlm_to_structured.py`` needs CLIP + CIFAR; here we test the mixle-facing piece
-- the per-class diagonal-Gaussian generative classifier -- on synthetic separable data, so the logic
(recovery, calibration direction, size) is covered fast and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.distill_vlm_to_structured import StructuredClassifier, cheap_features


def _three_blobs(n, seed):
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    y = rng.randint(0, 3, n)
    x = centers[y] + rng.randn(n, 2) * 0.7
    return x, y


def test_structured_classifier_recovers_separable_classes():
    xtr, ytr = _three_blobs(600, 0)
    xte, yte = _three_blobs(300, 1)
    clf = StructuredClassifier(diagonal=True).fit(xtr, ytr, 3)
    acc = (clf.predict(xte) == yte).mean()
    assert acc > 0.95  # well-separated Gaussians are exactly what this structured model represents
    assert clf.n_params() == 3 * (2 * 2) + 3  # 3 classes x (mean+var over 2 dims) + priors


def test_confidence_is_calibrated_higher_where_correct():
    xtr, ytr = _three_blobs(600, 0)
    xte, yte = _three_blobs(400, 2)
    clf = StructuredClassifier(diagonal=True).fit(xtr, ytr, 3)
    conf, pred = clf.confidence(xte)
    correct = pred == yte
    # points the model is confident about are the ones it gets right -- the property the demo relies on
    if (~correct).any():
        assert conf[correct].mean() > conf[~correct].mean()
    order = np.argsort(-conf)
    top_half, bottom_half = order[: len(order) // 2], order[len(order) // 2 :]
    assert correct[top_half].mean() >= correct[bottom_half].mean()


def test_cheap_features_are_encoder_free_fixed_size():
    Image = pytest.importorskip("PIL.Image")

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    feats = cheap_features([img, img])
    assert feats.shape == (2, 12 * 12 * 3)  # downsampled RGB, no learned net
    assert feats.min() >= 0.0 and feats.max() <= 1.0
