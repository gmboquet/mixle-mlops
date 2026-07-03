"""Close the loop for real: factory -> verified pairs -> from-scratch tiny LM -> EXECUTION accuracy.

This is the honest end-to-end run of the "train a tiny LLM to parse HTML by writing code" pipeline:

1. The verified data factory (:mod:`mixle_mlops.datasets.code_tasks`) manufactures tasks -- pages
   rendered FROM known records -- and harvests ``(page-snippet -> parser code)`` pairs whose code was
   EXECUTED in the sandbox and reproduced the truth. No pair enters training on anyone's word.
2. A from-scratch causal Transformer (``mixle.models.language_model.LM``, ~0.65M params, no
   pretraining) is SFT-trained on those pairs with ``fit_pairs`` (dense teacher forcing, loss masked to
   the code). Tokens are identifier-level (a field name is one token), so "copy the fields from this
   page into the record constructor" is a single induction step -- the reason real code models tokenize.
3. The student is scored the only way that matters -- by RUNNING its generated programs against pages
   from schema combinations it NEVER saw in training. Crucially it runs its OWN code at inference: the
   write->run->fix repair loop (feed the real traceback back) and label-free self-consistency (sample
   several, keep what the programs AGREE on) -- no ground truth needed at serving time.

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
import os
import re
import time
from pathlib import Path

import numpy as np

from mixle.models.language_model import LM
from mixle_mlops.datasets.code_tasks import (
    ReferenceTeacher,
    harvest,
    make_task,
    repair_loop,
    run_parser,
    verify,
)

# TINY_PARSER_SMOKE=1 runs a fast, tiny version (few tasks, few epochs) to exercise every code path
# without the full ~20-minute train -- for CI / a quick "does the pipeline run end to end" check.
SMOKE = os.environ.get("TINY_PARSER_SMOKE") == "1"

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
# id 0 = PAD; the rest are separators that never occur in pages or code, so they stay atomic tokens.
SEP, EOS, ERRSEP, FIXSEP, PAD = "\t", "\v", "\x1e", "\x1d", 0


def _corrupt(code: str, variant: int) -> str:
    """A realistic near-correct bug -- the kind a model actually emits -- so repair data has real errors."""
    subs = [
        lambda c: c.replace("float(", "flot(", 1),  # NameError: flot
        lambda c: c.replace("re.findall", "re.findl", 1),  # AttributeError
        lambda c: re.sub(r"v\[(\d)\]", lambda m: f"v[{int(m.group(1)) + 7}]", c, count=1),  # IndexError
        lambda c: c.replace("int(", "int (", 1) if "int(" in c else c.replace("str", "strr", 1),
    ]
    return subs[variant % len(subs)](code)


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
    train_combos, eval_combos = (combos[25:33], combos[:3]) if SMOKE else (combos[25:135], combos[:25])
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

    # --- 3. tokenized corpus: fresh pairs + REPAIR pairs (so the model learns write->run->fix) ----------
    # Identifier-level tokens: each field name ("price") is ONE token in both the page snippet and the
    # code. That turns "copy the field names from this page into the record constructor" into a single
    # induction step a tiny model actually learns -- char-level makes copying a 5-char sequence and the
    # model instead memorizes global field->type stats and hallucinates the combo (verified: it did).
    # Whitespace runs also collapse to one token (a 4-space indent is 1 token, not 4) -- shorter sequences
    # mean fewer positions where one slip breaks the program. The regex tiles every char, so join is exact.
    tok_re = re.compile(r"[A-Za-z_]+|\n+| +|.", re.S)

    def tokenize(s):
        return tok_re.findall(s)

    def fresh(html):  # prompt for a first attempt
        return tokenize(snippet(html)) + [SEP]

    def repair(html, failed_code, error):  # prompt after a crash: broken code + the real traceback
        # no page needed -- the broken code already carries the field names, and the errors are structural,
        # fixable from the code + traceback; this also keeps the repair prompt short (block stays ~320)
        return tokenize(failed_code) + [ERRSEP] + tokenize(error) + [FIXSEP]

    # fresh (snippet -> correct code) plus repair (snippet + broken code + REAL traceback -> correct code).
    # The broken drafts are corruptions of the verified program, executed to capture their genuine error --
    # so the repair examples teach recovery from the errors the model will actually produce.
    pairs_tok = []
    n_repair = 0
    for i, t in enumerate(ds.trajectories):
        pairs_tok.append((fresh(t.task.html), tokenize(t.code) + [EOS]))
        bad = _corrupt(t.code, i)
        _out, err = run_parser(bad, t.task.html)
        if err is not None:  # a genuine failure with a genuine traceback -> a repair example
            pairs_tok.append((repair(t.task.html, bad, err), tokenize(t.code) + [EOS]))
            n_repair += 1

    vocab = sorted({tok for p, c in pairs_tok for tok in p + c})
    stoi = {t: i + 2 for i, t in enumerate(vocab)}  # 0 = PAD, 1 = UNK (an eval-only page token)
    stoi["<unk>"] = 1

    def enc(toks):
        return [stoi.get(t, 1) for t in toks]

    itos = {i: t for t, i in stoi.items()}
    pairs_text = [(enc(p), enc(c)) for p, c in pairs_tok]
    longest = max(len(p) + len(c) for p, c in pairs_text)
    block = int(np.ceil(longest / 64) * 64)
    print(f"corpus: {len(pairs_tok)} pairs ({n_repair} repair), vocab {len(stoi) + 1} tokens, block {block}")

    # --- 4. from-scratch tiny LM, SFT on the verified pairs --------------------------------------------
    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.manual_seed(0)
    except ImportError as e:  # pragma: no cover - the demo needs torch
        raise SystemExit("this example trains a torch LM: pip install torch") from e
    dim, layers, heads = (64, 2, 4) if SMOKE else (192, 4, 6)
    lm = LM(vocab=len(stoi) + 1, d_model=dim, n_layer=layers, n_head=heads, block=block, device=device)
    n_params = sum(p.numel() for p in lm.module.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, identifier-level, from scratch, device={device}")

    # two-phase SFT: a warm phase at 2e-3 to fit the structure, then a precision phase at 5e-4 to stop
    # the mechanical slips (dropped index, duplicated token) that a constant-LR small model leaves behind
    warm, fine = (6, 2) if SMOKE else (110, 30)
    t_fit = time.time()
    lm.fit_pairs(
        pairs_text,
        epochs=warm,
        batch_size=32,
        lr=2e-3,
        seed=0,
        log=lambda e, x: (e % 10 == 9) and print(f"  epoch {e + 1:3d}/{warm}  loss/tok {x:.4f}"),
    )
    lm.fit_pairs(pairs_text, epochs=fine, batch_size=32, lr=5e-4, seed=1)
    print(f"trained in {time.time() - t_fit:.0f}s")

    # --- 5. serving: the model runs its OWN code and judges it WITHOUT ground truth --------------------
    # On a real page there is no truth to check against, so we score the label-free serving modes and only
    # use the held-out truth to MEASURE them. One shared pool of samples keeps the eval cheap:
    #   greedy            sample 0 alone (the baseline).
    #   self-consistency  run all N, return the record-set the most programs AGREE on; agreement = a
    #                     confidence (does it predict correctness?).
    #   best-of-N oracle  any sample that matches truth -- an UPPER BOUND (uses labels; not deployable).
    #   repair-loop       write -> run -> feed the real traceback back -> fix (the "then run it" loop) --
    #                     its own sequential pass, since each turn depends on the previous error.
    eos_id, max_new, N = stoi[EOS], 200, 6

    def codegen(html, schema=None, *, sample=0, failed_code=None, error=None):
        toks = repair(html, failed_code, error) if failed_code is not None else fresh(html)
        prompt = enc(toks)
        greedy = failed_code is None and sample == 0  # explore on repairs and on samples > 0
        out = lm.generate(prompt, n=max_new, greedy=greedy, temperature=0.5, seed=sample + 1, stop_id=eos_id)
        return "".join(itos[i] for i in out[len(prompt) :] if i not in (eos_id, PAD, 1))

    rows = []
    for j, task in enumerate(eval_tasks):
        # one pool of N programs; greedy / self-consistency / oracle are all read off it (no re-sampling)
        cands = []
        for k in range(N):
            code = codegen(task.html, sample=k)
            parsed, _ = run_parser(code, task.html)
            cands.append((code, parsed))
        greedy_code, greedy_parsed = cands[0]
        g_ok = verify(greedy_parsed, task.records)[0]
        ran = [(c, r) for c, r in cands if r is not None]

        # self-consistency: the record-set the most run-programs agree on (verify == exact set match, so
        # its own vote count is inter-program agreement), with that agreement share as the confidence
        sc_ok, sc_agreement, sc_ran = False, 0.0, bool(ran)
        if ran:
            votes = [sum(verify(ri, rj)[0] for _c2, rj in ran) for _c1, ri in ran]
            best = max(range(len(ran)), key=lambda i: votes[i])
            sc_agreement = votes[best] / len(ran)
            sc_ok = verify(ran[best][1], task.records)[0]
        oracle_ok = any(verify(r, task.records)[0] for _c, r in ran)

        rep = repair_loop(codegen, task.html, task.schema, max_rounds=4)
        rep_ok = verify(rep.records, task.records)[0] if rep.records is not None else False

        rows.append(
            {
                "template": task.template,
                "greedy_ok": g_ok,
                "greedy_ran": greedy_parsed is not None,
                "repair_ok": rep_ok,
                "repair_ran": rep.ran,
                "repair_rounds": rep.rounds,
                "sc_ok": sc_ok,
                "sc_ran": sc_ran,
                "sc_agreement": sc_agreement,
                "oracle_ok": oracle_ok,
                "program": greedy_code if j == 0 else None,
            }
        )
        if j % 15 == 14:
            print(f"  eval {j + 1}/{len(eval_tasks)}", flush=True)

    n = len(rows)
    frac = lambda key: sum(r[key] for r in rows) / n  # noqa: E731

    print("\n=== serving on UNSEEN schema combinations (label-free unless noted) ===", flush=True)
    print(f"greedy          : exact {frac('greedy_ok'):.2%}  ran {frac('greedy_ran'):.2%}")
    repaired = frac("repair_ok") - frac("greedy_ok")
    print(
        f"repair-loop     : exact {frac('repair_ok'):.2%}  ran {frac('repair_ran'):.2%}  "
        f"(+{repaired:.2%} over greedy from write->run->fix)"
    )
    print(f"self-consistency: exact {frac('sc_ok'):.2%}  answered {frac('sc_ran'):.2%}  (n={N}, label-free)")
    print(f"best-of-{N} oracle: exact {frac('oracle_ok'):.2%}  (upper bound, uses labels)")

    # calibration: does inter-program agreement predict correctness? (the label-free confidence)
    hi = [r for r in rows if r["sc_ran"] and r["sc_agreement"] >= 0.99]
    lo = [r for r in rows if r["sc_ran"] and r["sc_agreement"] < 0.99]
    acc = lambda rs: (sum(r["sc_ok"] for r in rs) / len(rs)) if rs else float("nan")  # noqa: E731
    print(
        f"calibration     : full-agreement {acc(hi):.0%} correct ({len(hi)} tasks)  vs  "
        f"split {acc(lo):.0%} correct ({len(lo)} tasks)"
    )

    print(f"\n--- greedy program, unseen combo {list(eval_tasks[0].schema)} on '{eval_tasks[0].template}' ---")
    print(rows[0]["program"])
    print(f"total {time.time() - t0:.0f}s", flush=True)
    results = {
        "greedy_exact": frac("greedy_ok"),
        "repair_exact": frac("repair_ok"),
        "self_consistency_exact": frac("sc_ok"),
        "oracle_exact": frac("oracle_ok"),
        "calibration": {"high_agreement_acc": acc(hi), "low_agreement_acc": acc(lo)},
        "params": n_params,
        "repair_pairs": n_repair,
    }
    (Path(__file__).parent / "tiny_parser_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
