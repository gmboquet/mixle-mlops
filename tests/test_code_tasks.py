"""The verified data factory for code-writing tasks, end to end with REAL execution.

Every kept training pair is proven by running the code in the sandbox against a page whose ground
truth is known by construction (the page was rendered FROM the records). No mocks anywhere: the
reference teacher emits genuine Python, the sandbox is a genuine subprocess, and the repair path is
exercised by a teacher that really writes broken code first.
"""

from __future__ import annotations

import json

import pytest

from mixle_mlops.datasets.code_tasks import (
    TEMPLATES,
    ReferenceTeacher,
    evaluate,
    harvest,
    make_task,
    render_page,
    run_parser,
    verify,
)

SCHEMA = {"name": "str", "price": "float", "qty": "int"}


# --- rendering: labels by construction --------------------------------------------------------------


def test_render_is_deterministic_and_contains_every_record():
    task = make_task(SCHEMA, n_rows=6, template="table", seed=7)
    again = make_task(SCHEMA, n_rows=6, template="table", seed=7)
    assert task.html == again.html and task.records == again.records
    for r in task.records:  # every truth value literally appears in the page
        for v in r.values():
            assert str(v) in task.html


def test_each_template_family_renders_and_differs():
    pages = {t: render_page(make_task(SCHEMA, 4, template=t, seed=3).records, t, seed=3) for t in TEMPLATES}
    assert "<table" in pages["table"] and "class='item'" in pages["divs"] and "<li>" in pages["list"]
    assert len(set(pages.values())) == len(TEMPLATES)


def test_noise_injects_distractors_without_touching_truth():
    quiet = make_task(SCHEMA, 5, template="table", seed=11, noise=0.0)
    noisy = make_task(SCHEMA, 5, template="table", seed=11, noise=1.0)
    assert quiet.records == noisy.records  # same truth...
    assert len(noisy.html) > len(quiet.html)  # ...more page
    code = ReferenceTeacher()(noisy.html, SCHEMA)
    parsed, err = run_parser(code, noisy.html)
    assert err is None and verify(parsed, noisy.records)[0]  # truth still exactly recoverable


def test_render_rejects_unknown_template():
    with pytest.raises(ValueError):
        render_page([{"a": 1}], "iframe-soup")


# --- the sandbox -------------------------------------------------------------------------------------


def test_run_parser_executes_good_code():
    task = make_task(SCHEMA, 3, template="list", seed=1)
    parsed, err = run_parser(ReferenceTeacher()(task.html, SCHEMA), task.html)
    assert err is None
    assert verify(parsed, task.records) == (True, 1.0)


def test_run_parser_surfaces_errors_as_feedback_text():
    bad_syntax = "def parse(html)\n    return []\n"
    parsed, err = run_parser(bad_syntax, "<html></html>")
    assert parsed is None and "SyntaxError" in err

    raises = "def parse(html):\n    return [x['nope'] for x in [{}]]\n"
    parsed, err = run_parser(raises, "<html></html>")
    assert parsed is None and "KeyError" in err

    no_parse = "x = 1\n"
    parsed, err = run_parser(no_parse, "<html></html>")
    assert parsed is None and "must define parse" in err

    wrong_shape = "def parse(html):\n    return 'not a list'\n"
    parsed, err = run_parser(wrong_shape, "<html></html>")
    assert parsed is None and "list of dicts" in err


def test_run_parser_contains_hangs_with_timeout():
    infinite = "def parse(html):\n    while True:\n        pass\n"
    parsed, err = run_parser(infinite, "<html></html>", timeout=1.0)
    assert parsed is None and "timeout" in err


# --- verification: order-insensitive record-set match ------------------------------------------------


def test_verify_is_order_insensitive_and_scores_near_misses():
    truth = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert verify(list(reversed(truth)), truth) == (True, 1.0)
    assert verify([{"a": "1", "b": "x"}, {"a": "2", "b": "y"}], truth)[0]  # string/int compare as str
    exact, f1 = verify([truth[0]], truth)  # one of two rows
    assert not exact and f1 == pytest.approx(2 * 0.5 / 1.5)
    assert verify([], truth) == (False, 0.0)
    assert verify(None, truth) == (False, 0.0)
    exact, _ = verify(truth + [{"a": 9, "b": "z"}], truth)  # extra row breaks exactness
    assert not exact


# --- harvesting: execution decides what enters the dataset --------------------------------------------


def test_harvest_reference_teacher_yields_verified_pairs_on_all_templates(tmp_path):
    tasks = [make_task(SCHEMA, 4, template=t, seed=s) for t in TEMPLATES for s in (0, 1)]
    ds = harvest(ReferenceTeacher(), tasks)
    assert ds.attempted == len(tasks) and ds.yield_rate == 1.0
    assert all(t.f1 == 1.0 for t in ds.trajectories)

    path = ds.save_jsonl(str(tmp_path / "sft.jsonl"))
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == len(tasks)  # no repairs -> one pair per task
    # each saved completion is code that still executes to the truth for its own prompt's page
    for row, task in zip(rows, tasks):
        assert task.html in row["prompt"] and "parse" in row["completion"]
        parsed, err = run_parser(row["completion"], task.html)
        assert err is None and verify(parsed, task.records)[0]


def test_harvest_captures_repair_trajectories_from_failed_first_drafts(tmp_path):
    tasks = [make_task(SCHEMA, 3, template="table", seed=s) for s in range(3)]
    ds = harvest(ReferenceTeacher(fail_first=True), tasks, attempts=2)
    assert ds.yield_rate == 1.0  # every page recovered on the second draft
    assert len(ds.repairs) == len(tasks)
    for t in ds.repairs:
        assert "undefined_name" in t.failed_code and t.feedback  # the error text was fed back

    rows = [json.loads(line) for line in open(ds.save_jsonl(str(tmp_path / "sft.jsonl")))]
    assert len(rows) == 2 * len(tasks)  # each task saves the pair AND the repair turn
    repair_rows = [r for r in rows if "previous attempt failed" in r["prompt"]]
    assert len(repair_rows) == len(tasks)
    assert all("undefined_name" in r["prompt"] for r in repair_rows)


def test_harvest_rejects_unverified_code_honestly():
    def hopeless_teacher(html, schema, **kw):
        return "def parse(html):\n    return []\n"  # runs fine, extracts nothing

    ds = harvest(hopeless_teacher, [make_task(SCHEMA, 3, seed=0)], attempts=2)
    assert ds.attempted == 1 and len(ds.trajectories) == 0 and ds.yield_rate == 0.0


def test_harvest_min_f1_admits_near_misses_when_loosened():
    def drops_last_row(html, schema, **kw):
        good = ReferenceTeacher()(html, schema)
        return good.replace("return out", "return out[:-1]") if "return out" in good else good

    tasks = [make_task(SCHEMA, 5, template="table", seed=0)]
    strict = harvest(drops_last_row, tasks, attempts=1)
    loose = harvest(drops_last_row, tasks, attempts=1, min_f1=0.7)
    assert strict.yield_rate == 0.0 and loose.yield_rate == 1.0
    assert loose.trajectories[0].f1 < 1.0  # the honest score rides along


# --- evaluation: execution accuracy, the only metric that matters -------------------------------------


def test_evaluate_scores_by_running_the_code():
    tasks = [make_task(SCHEMA, 4, template=t, seed=9) for t in TEMPLATES]
    good = evaluate(ReferenceTeacher(), tasks)
    assert good == {"exact": 1.0, "mean_f1": 1.0, "run_rate": 1.0}

    bad = evaluate(lambda html, schema, **kw: "def parse(html):\n    return []\n", tasks)
    assert bad["exact"] == 0.0 and bad["run_rate"] == 1.0  # runs, but extracts nothing

    broken = evaluate(lambda html, schema, **kw: "not even python (", tasks)
    assert broken == {"exact": 0.0, "mean_f1": 0.0, "run_rate": 0.0}


def test_evaluate_best_of_k_uses_the_free_verifier_at_inference():
    tasks = [make_task(SCHEMA, 3, template="table", seed=2)]

    def flaky_student(html, schema, sample=0, **kw):
        # sample 0 is wrong, sample 1 is right: the verifier promotes the working program
        if sample == 0:
            return "def parse(html):\n    return []\n"
        return ReferenceTeacher()(html, schema)

    assert evaluate(flaky_student, tasks, best_of=1)["exact"] == 0.0
    assert evaluate(flaky_student, tasks, best_of=3)["exact"] == 1.0


# --- the LLM teacher slot: gateway plumbing + code extraction; execution still judges ------------------


def test_extract_code_prefers_fenced_blocks():
    from mixle_mlops.datasets.code_tasks import extract_code

    fenced = "Here you go:\n```python\ndef parse(html):\n    return []\n```\nHope that helps!"
    assert extract_code(fenced) == "def parse(html):\n    return []\n"
    bare = "def parse(html):\n    return []"
    assert extract_code(bare) == "def parse(html):\n    return []\n"
    anon_fence = "```\ndef parse(html):\n    return []\n```"
    assert extract_code(anon_fence) == "def parse(html):\n    return []\n"


def test_llm_teacher_end_to_end_with_mock_gateway():
    import httpx

    from mixle_mlops.datasets.code_tasks import LLMTeacher

    task = make_task(SCHEMA, 3, template="table", seed=5)
    real_code = ReferenceTeacher()(task.html, SCHEMA)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        # a chatty model reply wrapping REAL code in a fence, prose on both sides
        content = f"Sure! Here is the parser:\n```python\n{real_code}```\nLet me know if it works."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/v1")
    teacher = LLMTeacher("smollm2", client=client)
    ds = harvest(teacher, [task])
    assert ds.yield_rate == 1.0  # extracted code EXECUTED to the truth
    assert seen[0]["model"] == "smollm2"
    assert task.html in seen[0]["messages"][0]["content"]


def test_llm_teacher_repair_turn_feeds_error_back():
    import httpx

    from mixle_mlops.datasets.code_tasks import LLMTeacher

    task = make_task(SCHEMA, 3, template="list", seed=6)
    real_code = ReferenceTeacher()(task.html, SCHEMA)
    calls: list[list] = []

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        calls.append(messages)
        code = "def parse(html):\n    return broken\n" if len(messages) == 1 else real_code
        return httpx.Response(200, json={"choices": [{"message": {"content": f"```python\n{code}```"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/v1")
    ds = harvest(LLMTeacher("smollm2", client=client), [task], attempts=2)
    assert ds.yield_rate == 1.0 and len(ds.repairs) == 1
    # the second call carried the failed code and the sandbox's error text back to the model
    assert len(calls[1]) == 3
    assert "broken" in calls[1][1]["content"] and "NameError" in calls[1][2]["content"]
