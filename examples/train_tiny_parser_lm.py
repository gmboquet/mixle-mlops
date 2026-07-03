"""Close the loop for real: factory -> verified pairs -> from-scratch tiny LM -> EXECUTION accuracy.

This is the honest end-to-end run of the "train a tiny LLM to parse HTML by writing code" pipeline:

1. The verified data factory (:mod:`mixle_mlops.datasets.code_tasks`) manufactures tasks -- pages
   rendered FROM known records -- and harvests ``(page-snippet -> parser code)`` pairs whose code was
   EXECUTED in the sandbox and reproduced the truth. No pair enters training on anyone's word.
2. A from-scratch character-level causal Transformer (``mixle.models.language_model.LM``, ~0.7M params,
   no pretraining, no tokenizer) is SFT-trained on those pairs with ``fit_pairs`` (dense teacher
   forcing, loss masked to the code).
3. The student is scored the only way that matters -- by RUNNING its generated programs against pages
   from schema combinations it NEVER saw in training, with best-of-k sampling promoted by the same free
   execution verifier used at harvest time (the serving mode).

Scope, honestly: the model reads a capped snippet of the page's data region (template + field names +
one data row for value shapes), not arbitrary web HTML, and the field-name pool is fixed with one type
per name. Generalization tested = UNSEEN (field-combination x order x template) tasks. Scaling beyond
this is a capacity/data question (SmolLM2-class base via ``POST /v1/fine_tunes``), not a pipeline one:
every stage here is exactly what the bigger run reuses.

Run:  python examples/train_tiny_parser_lm.py            (~3-6 min on Apple Silicon; CPU works too)
"""

from __future__ import annotations

import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

from mixle.models.language_model import LM
from mixle_mlops.datasets.code_tasks import ReferenceTeacher, harvest, make_task, run_parser, verify

TEMPLATES = ("table", "divs", "list")
FIELD_POOL = {
    "name": "str",
    "price": "float",
    "qty": "int",
    "city": "str",
    "score": "float",
    "rank": "int",
    "code": "str",
    "mass": "float",
}
SNIPPET_CHARS = 176  # data region cap: header/first item + one data row (field names AND value shapes)
SEP, EOS, PAD = "\t", "\v", 0  # id 0 is reserved for padding; SEP/EOS never occur in pages or code


def snippet(html: str) -> str:
    """The page's data region: enough to read the template, the field names, and one row of values."""
    for pat in (
        r"<table.*?</tr>\s*(?:<!--.*?-->\s*)?<tr>.*?</tr>",  # header row + first data row
        r"<div class='item'>.*?</div>",  # first item block
        r"<ul class='records'>\s*<li>.*?</li>",  # first list entry
    ):
        m = re.search(pat, html, re.S)
        if m:
            return m.group(0)[:SNIPPET_CHARS]
    return html[:SNIPPET_CHARS]


def combo_tasks(combos, *, seeds, noise=0.5):
    return [
        make_task({k: FIELD_POOL[k] for k in combo}, 4, template=tpl, seed=s, noise=noise)
        for combo in combos
        for tpl in TEMPLATES
        for s in seeds
    ]


def main() -> dict:
    t0 = time.time()
    rng = np.random.RandomState(0)

    # --- 1. manufacture tasks; hold out COMBINATIONS the student will never see ------------------------
    combos = list(itertools.permutations(FIELD_POOL, 3))
    rng.shuffle(combos)
    train_combos, eval_combos = combos[25:135], combos[:25]
    train_tasks = combo_tasks(train_combos, seeds=(0, 1))
    eval_tasks = combo_tasks(eval_combos, seeds=(7,))
    print(
        f"tasks: {len(train_tasks)} train ({len(train_combos)} combos x 3 templates x 2 seeds), "
        f"{len(eval_tasks)} eval on {len(eval_combos)} UNSEEN combos"
    )

    # --- 2. harvest execution-verified pairs (the factory refuses anything that doesn't run true) ------
    ds = harvest(ReferenceTeacher(), train_tasks)
    assert ds.yield_rate == 1.0, f"reference teacher should verify everywhere, got {ds.yield_rate:.2%}"
    jsonl = Path(__file__).parent / "tiny_parser_sft.jsonl"
    ds.save_jsonl(str(jsonl))
    print(f"harvested: {len(ds.trajectories)} verified pairs (yield {ds.yield_rate:.0%}) -> {jsonl.name}")

    # --- 3. char-level corpus: snippet SEP code EOS ----------------------------------------------------
    # the vocab is the CLOSED charset the renderer/teacher can produce, not the observed corpus --
    # an eval page may legally contain a character (e.g. a rare word) no training snippet happened to
    import string

    pairs_text = [(snippet(t.task.html) + SEP, t.code + EOS) for t in ds.trajectories]
    charset = sorted(string.ascii_letters + string.digits + string.punctuation + " \n" + SEP + EOS)
    stoi = {c: i + 1 for i, c in enumerate(charset)}  # 0 = PAD

    def enc(s):
        return [stoi[c] for c in s]

    itos = {i: c for c, i in stoi.items()}
    longest = max(len(p) + len(c) for p, c in pairs_text)
    block = int(np.ceil(longest / 64) * 64)
    print(f"corpus: vocab {len(stoi) + 1} chars, longest pair {longest}, block {block}")

    # --- 4. from-scratch tiny LM, SFT on the verified pairs --------------------------------------------
    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.manual_seed(0)
    except ImportError as e:  # pragma: no cover - the demo needs torch
        raise SystemExit("this example trains a torch LM: pip install torch") from e
    lm = LM(vocab=len(stoi) + 1, d_model=128, n_layer=3, n_head=4, block=block, device=device)
    n_params = sum(p.numel() for p in lm.module.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, char-level, from scratch, device={device}")

    pairs_ids = [(enc(p), enc(c)) for p, c in pairs_text]
    epochs, t_fit = 60, time.time()
    lm.fit_pairs(
        pairs_ids,
        epochs=epochs,
        batch_size=24,
        lr=3e-3,
        seed=0,
        log=lambda e, x: (e % 10 == 9) and print(f"  epoch {e + 1:3d}/{epochs}  loss/char {x:.4f}"),
    )
    print(f"trained in {time.time() - t_fit:.0f}s")

    # --- 5. the student writes code for unseen tasks; EXECUTION judges it (best-of-k serving mode) -----
    # one fused pass: sample 0 is greedy (the plain metric), samples 1..7 are temperature draws the
    # free verifier may promote (the serving mode) -- stopping at the first program that verifies

    eos_id, max_new = stoi[EOS], 320

    def student(html, sample=0):
        prompt = enc(snippet(html) + SEP)
        out = lm.generate(prompt, n=max_new, greedy=(sample == 0), temperature=0.5, seed=sample, stop_id=eos_id)
        return "".join(itos[i] for i in out[len(prompt) :] if i != eos_id and i != PAD)

    rows = []
    for j, task in enumerate(eval_tasks):
        row = {"template": task.template, "greedy_ok": False, "greedy_f1": 0.0, "best8_ok": False, "ran": False}
        for k in range(8):
            code = student(task.html, sample=k)
            parsed, _err = run_parser(code, task.html)
            row["ran"] = row["ran"] or parsed is not None
            ok, f1 = verify(parsed, task.records)
            if k == 0:
                row.update(greedy_ok=ok, greedy_f1=f1, program=code)
            if ok:
                row["best8_ok"] = True
                break
        rows.append(row)
        if j % 15 == 14:
            print(f"  eval {j + 1}/{len(eval_tasks)}", flush=True)

    n = len(rows)
    greedy = {
        "exact": sum(r["greedy_ok"] for r in rows) / n,
        "mean_f1": sum(r["greedy_f1"] for r in rows) / n,
        "run_rate": sum(r["ran"] for r in rows) / n,
    }
    best8_exact = sum(r["best8_ok"] for r in rows) / n
    per_template = {
        tpl: (lambda rs: sum(r["best8_ok"] for r in rs) / max(1, len(rs)))([r for r in rows if r["template"] == tpl])
        for tpl in TEMPLATES
    }

    print("\n=== execution accuracy on UNSEEN schema combinations ===", flush=True)
    print(f"greedy    : exact {greedy['exact']:.2%}  mean-f1 {greedy['mean_f1']:.2%}  ran {greedy['run_rate']:.2%}")
    print(f"best-of-8 : exact {best8_exact:.2%}")
    print("per-template exact (best-of-8): " + ", ".join(f"{k} {v:.0%}" for k, v in per_template.items()))
    sample_task = eval_tasks[0]
    print(f"\n--- the greedy program for unseen combo {list(sample_task.schema)} on '{sample_task.template}' ---")
    print(rows[0].get("program", ""))
    print(f"total {time.time() - t0:.0f}s", flush=True)
    results = {"greedy": greedy, "best8_exact": best8_exact, "per_template": per_template, "params": n_params}
    (Path(__file__).parent / "tiny_parser_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
