"""Cache CLIP CIFAR-100 image/text embeddings for examples/adapt_vlm_structured.py (run once, ~7 min CPU).

Writes examples/clip_c100.npz with Xtr/ytr (8000 train), Xte/yte (2000 test) image embeddings and Tcls
(100 class-text anchors). Kept separate so the adaptation experiment iterates without re-running CLIP.
"""

import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

torch.set_num_threads(6)
OUT = Path(__file__).parent / "clip_c100.npz"

ds_tr = load_dataset("cifar100", split="train[:8000]")
ds_te = load_dataset("cifar100", split="test[:2000]")
classes = ds_tr.features["fine_label"].names
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed(ds):
    imgs = [ex["img"] for ex in ds]
    y = np.array([ex["fine_label"] for ex in ds])
    E = []
    with torch.no_grad():
        for i in range(0, len(imgs), 256):
            E.append(model.get_image_features(**proc(images=imgs[i : i + 256], return_tensors="pt")).numpy())
            print(f"  {i + 256}/{len(imgs)}", flush=True)
    E = np.concatenate(E)
    return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32), y


with torch.no_grad():
    T = model.get_text_features(
        **proc(text=[f"a photo of a {c}" for c in classes], return_tensors="pt", padding=True)
    )
    Tcls = (T.numpy() / np.linalg.norm(T.numpy(), axis=1, keepdims=True)).astype(np.float32)

t0 = time.time()
Xtr, ytr = embed(ds_tr)
Xte, yte = embed(ds_te)
np.savez(OUT, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Tcls=Tcls, classes=np.array(classes))
print(f"cached {OUT.name} in {time.time() - t0:.0f}s", flush=True)
