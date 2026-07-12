"""Adapt a real VLM (CLIP) on a laptop with a structured bridge -- structure is what extracts the signal.

The core claim: structured architecture is what makes VLM training feasible on a laptop. You cannot train the
encoder (billions of FLOPs); you train the bridge on top, and the choice of bridge is the whole game. Measured
on real CLIP + CIFAR-100, few-shot (16 images/class), over several class splits:

  * ``mixle.reason.StructuredAdapter`` (diagonal + low-rank residual, ~9k params) reliably adapts the task
    (~+4.4%, tiny variance) and preserves CLIP's zero-shot transfer to classes it never trained on;
  * a full-matrix bridge (unstructured, ~260k params -- 30x larger) barely adapts at all (~+0.1%, high
    variance) under the same regularization -- it cannot extract a useful adaptation from laptop-scale data.

Both train in seconds on CPU over frozen CLIP embeddings; only the bridge trains. Structure is the inductive
bias that turns 1280 examples into a real adaptation; without it the same regularized capacity finds nothing.
Both preserve transfer here -- the honest headline is sample efficiency, not the (non-robust) claim that a big
bridge destroys generality.

Honest scope: this adapts a frozen frontier encoder; it is not training a VLM from scratch, and the gain is
modest. The point is that the structured bridge is the only one that learns anything at this data/compute scale
-- which is the whole game when the scale is a laptop.

Setup: needs cached CLIP embeddings. Run  examples/_embed_cifar100_clip.py  once to build them (~7 min CPU), then:
Run:  python examples/adapt_vlm_structured.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mixle.reason import StructuredAdapter

CACHE = Path(__file__).parent / "clip_c100.npz"


def raw_zero_shot_acc(X, y, anchors, cls_idx):
    """CLIP's own zero-shot accuracy on a class set (cosine to class-text anchors) -- the reference."""
    m = np.isin(y, cls_idx)
    a = anchors[cls_idx] / np.linalg.norm(anchors[cls_idx], axis=1, keepdims=True)
    xn = X[m] / np.linalg.norm(X[m], axis=1, keepdims=True)
    return (cls_idx[(xn @ a.T).argmax(1)] == y[m]).mean(), m


def adapted_acc(ad, X, y, anchors, cls_idx):
    m = np.isin(y, cls_idx)
    return (cls_idx[ad.predict(X[m], anchors[cls_idx])] == y[m]).mean()


def main():
    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE.name}; run examples/_embed_cifar100_clip.py first to cache CLIP features")
    d = np.load(CACHE, allow_pickle=True)
    Xtr, ytr, Xte, yte, Tcls = d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["Tcls"]
    dim = Xtr.shape[1]

    print("adapt frozen CLIP with a laptop-trained bridge; Δ = accuracy minus CLIP's own zero-shot on the same")
    print("class set (so the split's difficulty cancels). Mean of 5 class splits, 16 shots/seen-class.\n")
    print(f"{'bridge':<26}{'params':>9}{'seen Δ':>14}{'unseen Δ':>14}")

    import torch

    results = {}
    for full in (False, True):
        seen_d, unseen_d, npar = [], [], 0
        for sp in range(5):
            perm = np.random.RandomState(sp).permutation(100)
            seen, unseen = perm[:80], perm[80:]
            rng = np.random.RandomState(sp)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 16, replace=False) for c in seen])
            pos = -np.ones(100, int)
            pos[seen] = np.arange(len(seen))

            torch.manual_seed(sp)
            ad = StructuredAdapter(dim, rank=8, weight_decay=1.0, full=full)
            ad.fit(Xtr[idx], pos[ytr[idx]], Tcls[seen], epochs=300)
            npar = ad.n_params()

            r_seen, _ = raw_zero_shot_acc(Xte, yte, Tcls, seen)
            r_unseen, _ = raw_zero_shot_acc(Xte, yte, Tcls, unseen)
            seen_d.append(adapted_acc(ad, Xte, yte, Tcls, seen) - r_seen)
            unseen_d.append(adapted_acc(ad, Xte, yte, Tcls, unseen) - r_unseen)

        name = "full matrix (unstructured)" if full else "StructuredAdapter"
        results[name] = (np.mean(seen_d), np.mean(unseen_d))
        print(
            f"{name:<26}{npar:>9}"
            f"{np.mean(seen_d):>+9.3f}±{np.std(seen_d):.3f}"
            f"{np.mean(unseen_d):>+9.3f}±{np.std(unseen_d):.3f}"
        )

    s_seen, s_unseen = results["StructuredAdapter"]
    f_seen, f_unseen = results["full matrix (unstructured)"]
    print(
        f"\n-> the structured bridge (9k params) reliably adapts the task ({s_seen:+.1%}); the full matrix "
        f"(30x larger) gets {f_seen:+.1%}\n   from the same 1280 examples -- structure is the inductive bias "
        f"that extracts a useful adaptation at laptop\n   scale. Both preserve zero-shot transfer to unseen "
        f"classes ({s_unseen:+.1%} vs {f_unseen:+.1%}). The same recipe adapts\n   any frozen encoder "
        f"(Qwen-VL's vision tower) -- you train only the tiny structured bridge."
    )


if __name__ == "__main__":
    main()
