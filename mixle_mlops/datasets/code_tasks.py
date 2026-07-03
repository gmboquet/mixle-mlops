"""Verified training data for code-writing tasks: render pages from known records, keep only code that runs.

Training a (tiny) LLM to *parse HTML by writing code* needs data no one wants to hand-label. The way out
is that this task family has two gifts:

1. **Labels by construction (inverse rendering).** Sample structured records, then *render* them into
   HTML with varied templates and injected noise. The ground-truth extraction is known exactly -- the
   page was built from it. Unlimited volume, and difficulty (template family, noise level, schema
   width) is a curriculum knob you control.
2. **Execution is a free, perfect verifier.** A teacher writes parsing code; a sandbox runs it against
   the page; the output either matches the known records or it does not. Rejection-sampling on that
   verifier yields clean ``(html -> code)`` pairs with no human in the loop -- and failed attempts are
   not waste: ``(html, bad code, error, fixed code)`` becomes a *repair* trajectory, the
   write->run->fix loop the model must learn.

The pipeline: :func:`make_task` (records + rendered page) -> teacher writes code -> :func:`run_parser`
(subprocess sandbox) -> :func:`verify` (record-set match) -> :func:`harvest` (verified pairs + repair
turns, ``save_jsonl`` in prompt/completion form for ``/v1/fine_tunes`` or any SFT loop) ->
:func:`evaluate` (a model's EXECUTION accuracy -- programs judged by running them, never token match).

The subprocess isolates hangs and crashes; it is **not** a security boundary against a malicious
teacher. Synthetic templates cover what the renderer can express -- real-web coverage grows by adding
template families (or folding in scraped pages with verified parses), not by trusting the model more.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CodeTask",
    "Trajectory",
    "VerifiedDataset",
    "ReferenceTeacher",
    "LLMTeacher",
    "extract_code",
    "TEMPLATES",
    "DEFAULT_FIELD_POOL",
    "make_task",
    "build_tasks",
    "render_page",
    "run_parser",
    "verify",
    "harvest",
    "evaluate",
    "Extraction",
    "run_candidates",
    "self_consistent",
    "repair_loop",
]

TEMPLATES = ("table", "divs", "list")

_WORDS = (
    "alpha beta gamma delta echo fox golf hotel india juliet kilo lima mike nova oscar papa "
    "quebec romeo sierra tango umber victor whisky xray yankee zulu"
).split()


def _sample_records(schema: dict[str, str], n_rows: int, rng: np.random.RandomState) -> list[dict[str, Any]]:
    """Seeded records for a ``{field: 'str'|'int'|'float'}`` schema (the truth the page is built from)."""
    rows = []
    for _ in range(n_rows):
        row: dict[str, Any] = {}
        for name, kind in schema.items():
            if kind == "int":
                row[name] = int(rng.randint(0, 5000))
            elif kind == "float":
                row[name] = round(float(rng.uniform(0.5, 999.0)), 2)
            else:
                row[name] = f"{_WORDS[rng.randint(len(_WORDS))]}-{rng.randint(100)}"
        rows.append(row)
    return rows


def _noise_comment(rng: np.random.RandomState) -> str:
    return f"<!-- {_WORDS[rng.randint(len(_WORDS))]} {rng.randint(1000)} -->" if rng.random() < 0.6 else ""


def render_page(records: Sequence[dict[str, Any]], template: str, seed: int = 0, noise: float = 0.5) -> str:
    """Render ``records`` into an HTML page in one of the :data:`TEMPLATES`, with seeded noise.

    Noise injects comments, unrelated nav/aside sections, wrapper divs, and whitespace jitter --
    enough that a parser has to *select*, not just read everything -- while the record content stays
    exactly recoverable (that is the point: the label survives rendering).
    """
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    rng = np.random.RandomState(seed)
    fields = list(records[0].keys()) if records else []
    nl = "\n" + " " * int(rng.randint(0, 3)) if noise > 0 else "\n"

    def maybe(s: str) -> str:
        return s if (noise > 0 and rng.random() < noise) else ""

    parts = ["<html><head><title>%s report</title></head><body>" % _WORDS[rng.randint(len(_WORDS))]]
    parts.append(maybe(f"<nav><a href='/'>home</a><a href='/x{rng.randint(99)}'>archive</a></nav>"))
    parts.append(maybe(_noise_comment(rng)))
    parts.append(maybe(f"<aside class='promo'>only {rng.randint(9)} days left</aside>"))

    if template == "table":
        parts.append("<table id='data'>")
        parts.append("<tr>" + "".join(f"<th>{f}</th>" for f in fields) + "</tr>")
        for r in records:
            parts.append(maybe(_noise_comment(rng)))
            parts.append("<tr>" + "".join(f"<td>{r[f]}</td>" for f in fields) + "</tr>")
        parts.append("</table>")
    elif template == "divs":
        parts.append("<div class='items'>")
        for r in records:
            parts.append(maybe("<div class='spacer'></div>"))
            cells = "".join(f"<span class='{f}'>{r[f]}</span>{nl}" for f in fields)
            parts.append(f"<div class='item'>{nl}{cells}</div>")
        parts.append("</div>")
    else:  # list
        parts.append("<ul class='records'>")
        for r in records:
            body = " | ".join(f"{f}: {r[f]}" for f in fields)
            parts.append(f"<li>{body}</li>")
            parts.append(maybe(_noise_comment(rng)))
        parts.append("</ul>")

    parts.append(maybe(f"<footer>generated {rng.randint(2020, 2027)}</footer>"))
    parts.append("</body></html>")
    return "\n".join(p for p in parts if p)


@dataclass(frozen=True)
class CodeTask:
    """One training/eval unit: the page, the schema to extract, and the truth it was rendered from."""

    html: str
    schema: dict[str, str]
    records: list[dict[str, Any]]
    template: str
    seed: int


def make_task(
    schema: dict[str, str] | None = None,
    n_rows: int = 5,
    template: str | None = None,
    seed: int = 0,
    noise: float = 0.5,
) -> CodeTask:
    """Manufacture one task: sample records for ``schema``, render them, keep the truth attached."""
    rng = np.random.RandomState(seed)
    schema = schema or {"name": "str", "price": "float", "qty": "int"}
    template = template or TEMPLATES[rng.randint(len(TEMPLATES))]
    records = _sample_records(schema, n_rows, rng)
    return CodeTask(render_page(records, template, seed=seed, noise=noise), dict(schema), records, template, seed)


DEFAULT_FIELD_POOL = {
    "name": "str",
    "price": "float",
    "qty": "int",
    "city": "str",
    "score": "float",
    "rank": "int",
    "code": "str",
    "mass": "float",
}


def build_tasks(
    n: int,
    *,
    pool: dict[str, str] | None = None,
    fields_per_task: int = 3,
    n_rows: int = 4,
    templates: Sequence[str] | None = None,
    noise: float = 0.5,
    seed: int = 0,
) -> list[CodeTask]:
    """Sample ``n`` tasks: each a random field-subset of ``pool`` rendered in a random template.

    The generator is the curriculum -- ``pool`` width, ``fields_per_task``, ``templates``, and ``noise``
    are the difficulty knobs. Deterministic in ``seed``; every task carries its ground-truth records.
    """
    pool = pool or DEFAULT_FIELD_POOL
    names = list(pool)
    if fields_per_task > len(names):
        raise ValueError(f"fields_per_task={fields_per_task} exceeds pool size {len(names)}")
    templates = tuple(templates or TEMPLATES)
    rng = np.random.RandomState(seed)
    tasks = []
    for i in range(n):
        chosen = list(rng.choice(names, size=fields_per_task, replace=False))
        schema = {k: pool[k] for k in chosen}
        tpl = templates[rng.randint(len(templates))]
        tasks.append(make_task(schema, n_rows, template=tpl, seed=int(rng.randint(1 << 31)), noise=noise))
    return tasks


# --- the sandbox: run candidate parser code against a page ---------------------------------------------------

_RUNNER = """
import json, sys
html = sys.stdin.read()
ns = {}
exec(compile(open(sys.argv[1]).read(), "parser.py", "exec"), ns)
if "parse" not in ns:
    print(json.dumps({"error": "code must define parse(html) -> list[dict]"})); raise SystemExit(0)
try:
    out = ns["parse"](html)
    print(json.dumps({"records": out}))
except Exception as e:
    print(json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))
"""


def run_parser(code: str, html: str, timeout: float = 5.0) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Execute parser ``code`` (which must define ``parse(html) -> list[dict]``) against ``html``.

    Runs in a subprocess so hangs and crashes are contained (``timeout`` seconds); returns
    ``(records, None)`` on success or ``(None, error_text)`` -- the error text is exactly what a
    repair trajectory feeds back to the code writer. Not a security boundary.
    """
    with tempfile.TemporaryDirectory() as d:
        code_path = Path(d) / "parser.py"
        code_path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER, str(code_path)],
                input=html.encode(),
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout: parser exceeded {timeout}s"
    if proc.returncode != 0:
        return None, (proc.stderr.decode(errors="replace").strip().splitlines() or ["crashed"])[-1]
    try:
        payload = json.loads(proc.stdout.decode(errors="replace").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None, "parser produced no JSON output"
    if "error" in payload:
        return None, str(payload["error"])
    recs = payload.get("records")
    if not isinstance(recs, list) or not all(isinstance(r, dict) for r in recs):
        return None, "parse() must return a list of dicts"
    return recs, None


def _canon(records: Sequence[dict[str, Any]]) -> list[frozenset]:
    return [frozenset((str(k), str(v)) for k, v in r.items()) for r in records]


def verify(parsed: Sequence[dict[str, Any]] | None, truth: Sequence[dict[str, Any]]) -> tuple[bool, float]:
    """Row-set match of parsed vs truth: ``(exact, f1)``. Order-insensitive; values compared as strings."""
    if not parsed:
        return False, 0.0
    p, t = _canon(parsed), _canon(truth)
    t_pool = list(t)
    hits = 0
    for row in p:
        if row in t_pool:
            t_pool.remove(row)
            hits += 1
    precision = hits / len(p)
    recall = hits / len(t) if t else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return (hits == len(t) and len(p) == len(t)), float(f1)


# --- harvesting: teacher writes code, execution decides what enters the dataset ------------------------------


def _prompt(task: CodeTask) -> str:
    schema = ", ".join(f"{k} ({v})" for k, v in task.schema.items())
    return (
        "Write a Python function parse(html) that extracts every record from this page as a list of "
        f"dicts with keys: {schema}. Return only code.\n\n{task.html}"
    )


def extract_code(text: str) -> str:
    """Pull the program out of an LLM reply: the first fenced code block, else the raw text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S) or re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


class LLMTeacher:
    """A teacher backed by any OpenAI-compatible chat endpoint (the mixle-mlops gateway included).

    Drops into :func:`harvest`'s teacher slot: it prompts the model for ``parse(html)`` code (feeding
    back the failed code + sandbox error on repair calls), extracts the first fenced block, and returns
    it -- the EXECUTION verifier still decides what enters the dataset, so a sloppy teacher just lowers
    ``yield_rate``, never correctness. Pass ``client`` to reuse a session or to inject a test transport.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 640,
        timeout: float = 120.0,
        client: Any = None,
    ) -> None:
        import httpx

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def __call__(
        self,
        html: str,
        schema: dict[str, str],
        failed_code: str | None = None,
        error: str | None = None,
        sample: int = 0,
    ) -> str:
        fields = ", ".join(f"{k} ({v})" for k, v in schema.items())
        ask = (
            "Write a Python function parse(html) that extracts every record from this HTML page as a "
            f"list of dicts with keys: {fields}. Use only the standard library. Return only code.\n\n{html}"
        )
        messages = [{"role": "user", "content": ask}]
        if failed_code is not None:
            messages.append({"role": "assistant", "content": f"```python\n{failed_code}```"})
            messages.append(
                {"role": "user", "content": f"That failed when executed: {error}\nFix it. Return only code."}
            )
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": sample,  # distinct draws for best-of-N / self-consistency serving
            },
        )
        resp.raise_for_status()
        return extract_code(resp.json()["choices"][0]["message"]["content"])


@dataclass(frozen=True)
class Trajectory:
    """One verified unit: the prompt/page, the final working code, and any failed attempt it repaired."""

    task: CodeTask
    code: str
    f1: float
    failed_code: str | None = None
    feedback: str | None = None


@dataclass
class VerifiedDataset:
    """Execution-verified trajectories, ready for SFT. ``yield_rate`` is the honest teacher pass rate."""

    trajectories: list[Trajectory] = field(default_factory=list)
    attempted: int = 0

    @property
    def yield_rate(self) -> float:
        return len(self.trajectories) / self.attempted if self.attempted else 0.0

    @property
    def repairs(self) -> list[Trajectory]:
        return [t for t in self.trajectories if t.failed_code is not None]

    def jsonl_rows(self) -> list[dict[str, Any]]:
        """The prompt/completion rows (each verified pair, plus a repair turn feeding the error back)."""
        rows: list[dict[str, Any]] = []
        for t in self.trajectories:
            rows.append({"prompt": _prompt(t.task), "completion": t.code, "f1": t.f1})
            if t.failed_code is not None:
                repair_prompt = (
                    _prompt(t.task)
                    + f"\n\nA previous attempt failed.\nCode:\n{t.failed_code}\nError: {t.feedback}\nFix it."
                )
                rows.append({"prompt": repair_prompt, "completion": t.code, "f1": t.f1})
        return rows

    def jsonl_bytes(self) -> bytes:
        """The SFT corpus as JSONL bytes -- what a blob store / ``/v1/fine_tunes`` upload wants."""
        return "".join(json.dumps(r) + "\n" for r in self.jsonl_rows()).encode()

    def save_jsonl(self, path: str) -> str:
        """Write prompt/completion pairs (plus repair turns with the error fed back) for any SFT loop."""
        with open(path, "wb") as f:
            f.write(self.jsonl_bytes())
        return path


def harvest(
    teacher: Callable[..., str],
    tasks: Sequence[CodeTask],
    *,
    attempts: int = 2,
    min_f1: float = 1.0,
    timeout: float = 5.0,
) -> VerifiedDataset:
    """Run the teacher over ``tasks``; keep only code that EXECUTES to the known records.

    ``teacher(html, schema)`` returns parser code; on failure it is re-asked as
    ``teacher(html, schema, failed_code=..., error=...)`` up to ``attempts`` times, and a successful
    retry is recorded as a repair trajectory (bad code + error + fix). ``min_f1`` admits near-misses
    if you loosen it; the default keeps only exact extractions. Every kept pair re-verified by
    execution -- nothing enters the dataset on the teacher's word.
    """
    out = VerifiedDataset()
    for task in tasks:
        out.attempted += 1
        failed_code = feedback = None
        for attempt in range(max(1, attempts)):
            if attempt == 0:
                code = teacher(task.html, task.schema)
            else:
                code = teacher(task.html, task.schema, failed_code=failed_code, error=feedback)
            parsed, err = run_parser(code, task.html, timeout=timeout)
            _exact, f1 = verify(parsed, task.records)
            if f1 >= min_f1:
                out.trajectories.append(Trajectory(task, code, f1, failed_code, feedback))
                break
            failed_code, feedback = code, (err or f"extraction wrong (f1={f1:.2f} vs truth)")
    return out


def evaluate(
    codegen: Callable[..., str],
    tasks: Sequence[CodeTask],
    *,
    best_of: int = 1,
    timeout: float = 5.0,
) -> dict[str, float]:
    """A model's EXECUTION accuracy: fraction of tasks where generated code runs and reproduces the truth.

    ``best_of > 1`` samples multiple programs and scores the best that verifies -- the honest serving
    mode for a stochastic model (the verifier is free at inference too). Reports exact-rate, mean f1,
    and run-rate (code that at least executed).
    """
    exact = f1s = ran = 0.0
    for task in tasks:
        best_f1, any_ran, got_exact = 0.0, 0.0, 0.0
        for k in range(max(1, best_of)):
            code = codegen(task.html, task.schema) if best_of == 1 else codegen(task.html, task.schema, sample=k)
            parsed, _err = run_parser(code, task.html, timeout=timeout)
            if parsed is not None:
                any_ran = 1.0
            ok, f1 = verify(parsed, task.records)
            best_f1 = max(best_f1, f1)
            got_exact = max(got_exact, 1.0 if ok else 0.0)
            if ok:
                break
        exact += got_exact
        f1s += best_f1
        ran += any_ran
    n = max(1, len(tasks))
    return {"exact": exact / n, "mean_f1": f1s / n, "run_rate": ran / n}


# --- inference-time serving: run the model's own code, and do it WITHOUT ground truth ------------------------
#
# At training time the verifier has the truth (the page was built from it). On a real page at serving time
# it does NOT -- so the label-free signals are (a) does the code EXECUTE and return well-formed records, and
# (b) do independently-sampled programs AGREE. Both are available with zero labels, and (b) doubles as a
# calibrated confidence: high inter-program agreement predicts correctness (measured in the demo).


def _records_key(records: Sequence[dict[str, Any]]) -> frozenset:
    """An order-insensitive, hashable key for a whole record-set (a multiset of rows)."""
    counter: dict[frozenset, int] = {}
    for row in _canon(records):
        counter[row] = counter.get(row, 0) + 1
    return frozenset(counter.items())


@dataclass(frozen=True)
class Extraction:
    """One serving result: the records returned (or ``None`` = abstain), with a label-free confidence.

    ``ran`` = some program executed and returned well-formed records. ``agreement`` = fraction of the
    executed programs that produced *this* record-set (1.0 for a single program; the majority share under
    self-consistency). ``rounds`` = how many write->run->repair turns it took.
    """

    records: list[dict[str, Any]] | None
    code: str
    ran: bool
    agreement: float
    n_candidates: int
    n_ran: int
    rounds: int = 1


def run_candidates(
    codegen: Callable[..., str], html: str, schema: dict[str, str] | None = None, *, n: int = 5, timeout: float = 5.0
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    """Sample ``n`` programs from ``codegen`` and run each; returns ``[(code, records_or_None), ...]``."""
    out = []
    for k in range(max(1, n)):
        code = codegen(html, schema, sample=k)
        parsed, _err = run_parser(code, html, timeout=timeout)
        out.append((code, parsed))
    return out


def self_consistent(
    codegen: Callable[..., str], html: str, schema: dict[str, str] | None = None, *, n: int = 5, timeout: float = 5.0
) -> Extraction:
    """Label-free serving: sample ``n`` programs, run them all, return the record-set the MOST agree on.

    No ground truth is used -- the only signals are "did it run" and "do the programs agree". The winning
    record-set's agreement share is a confidence you can threshold to abstain (return records but flag it).
    """
    cands = run_candidates(codegen, html, schema, n=n, timeout=timeout)
    ran = [(code, recs) for code, recs in cands if recs is not None]
    if not ran:
        return Extraction(None, cands[-1][0] if cands else "", False, 0.0, len(cands), 0)
    tally: dict[frozenset, int] = {}
    rep: dict[frozenset, tuple[str, list[dict[str, Any]]]] = {}
    for code, recs in ran:
        key = _records_key(recs)
        tally[key] = tally.get(key, 0) + 1
        rep.setdefault(key, (code, recs))
    win_key = max(tally, key=lambda k: tally[k])
    code, recs = rep[win_key]
    return Extraction(recs, code, True, tally[win_key] / len(ran), len(cands), len(ran))


def repair_loop(
    codegen: Callable[..., str],
    html: str,
    schema: dict[str, str] | None = None,
    *,
    max_rounds: int = 3,
    timeout: float = 5.0,
) -> Extraction:
    """The write->run->fix loop: run the model's code; on a crash, feed the traceback back and let it retry.

    This is inference-time use of the sandbox with NO labels -- the feedback is the actual Python error, the
    same ``(failed_code, error)`` shape the repair trajectories trained on. Returns as soon as a program
    executes to well-formed records; abstains (``ran=False``) if still crashing after ``max_rounds``.
    """
    failed_code = error = None
    last_code = ""
    for r in range(max(1, max_rounds)):
        if r == 0:
            code = codegen(html, schema, sample=0)
        else:
            code = codegen(html, schema, failed_code=failed_code, error=error)
        last_code = code
        parsed, err = run_parser(code, html, timeout=timeout)
        if parsed is not None:
            return Extraction(parsed, code, True, 1.0, r + 1, 1, rounds=r + 1)
        failed_code, error = code, err
    return Extraction(None, last_code, False, 0.0, max_rounds, 0, rounds=max_rounds)


# --- a reference teacher: real code per template family (proves the machinery; swap in an LLM later) ----------


class ReferenceTeacher:
    """Writes genuine parsing code per template family -- the machinery's ground-truth teacher.

    It emits real Python (regex-based, stdlib-only) that the sandbox actually executes; an LLM teacher
    drops into the same ``(html, schema, failed_code=, error=)`` slot. ``fail_first=True`` corrupts
    its first attempt per page to exercise the repair loop deterministically.

    The programs are deliberately COMPACT and uniform (three skeletons; only the record-building line
    varies with the schema): short targets are what make the harvested pairs learnable by a tiny
    from-scratch model, and uniform structure is what such a model can generalize across schemas.
    """

    def __init__(self, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self._seen: set[int] = set()

    @staticmethod
    def _record(fields: list[str], casts: dict[str, str], value: Callable[[str, int], str]) -> str:
        cast = {"int": "int(%s)", "float": "float(%s)", "str": "%s"}
        return "{" + ", ".join(f'"{f}": ' + cast[casts[f]] % value(f, i) for i, f in enumerate(fields)) + "}"

    def __call__(
        self,
        html: str,
        schema: dict[str, str],
        failed_code: str | None = None,
        error: str | None = None,
        sample: int = 0,  # ignored: this teacher is deterministic (so it can stand in as a codegen)
    ) -> str:
        key = hash(html)
        if self.fail_first and key not in self._seen and failed_code is None:
            self._seen.add(key)
            return "def parse(html):\n    return undefined_name\n"  # deliberately broken first draft
        fields = list(schema)
        casts = dict(schema)
        if "<table" in html:
            body = f"""
import re
def parse(html):
    out = []
    for r in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        v = re.findall(r"<td>(.*?)</td>", r)
        if len(v) == {len(fields)}:
            out.append({self._record(fields, casts, lambda f, i: f"v[{i}]")})
    return out
"""
        elif "class='item'" in html or 'class="item"' in html:
            body = f"""
import re
def parse(html):
    out = []
    for b in re.findall(r"<div class='item'>(.*?)</div>", html, re.S):
        v = re.findall(r">([^<]*)</span>", b)
        out.append({self._record(fields, casts, lambda f, i: f"v[{i}]")})
    return out
"""
        else:  # list template: "field: value | field: value"
            body = f"""
import re
def parse(html):
    out = []
    for it in re.findall(r"<li>(.*?)</li>", html, re.S):
        kv = dict(p.split(": ", 1) for p in it.split(" | "))
        out.append({self._record(fields, casts, lambda f, i: f'kv["{f}"]')})
    return out
"""
        return body.strip() + "\n"
