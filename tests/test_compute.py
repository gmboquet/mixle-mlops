"""Offline tests for the vast.ai training launcher (no provisioning, no spend)."""
import pytest

from mixle_mlops.compute import TrainingJob, plan
from mixle_mlops.compute.jobspec import build_onstart, pip_packages, training_command
from mixle_mlops.compute.launcher import launch
from mixle_mlops.compute.vast import VastClient


def test_destroy_confirmed_polls_until_gone():
    c = VastClient("k")
    seen = {"n": 0}
    c.destroy = lambda i: None  # type: ignore[method-assign]

    def fake_instance(i):  # alive for two polls, then disappears
        seen["n"] += 1
        return {"actual_status": "running"} if seen["n"] < 3 else {}

    c.instance = fake_instance  # type: ignore[method-assign]
    assert c.destroy_confirmed(99, poll=0.0, timeout=5.0) is True


def test_destroy_confirmed_times_out_to_false():
    c = VastClient("k")
    c.destroy = lambda i: None  # type: ignore[method-assign]
    c.instance = lambda i: {"actual_status": "running"}  # never terminates  # type: ignore[method-assign]
    assert c.destroy_confirmed(99, poll=0.0, timeout=0.15) is False


def test_offer_query_shape():
    q = VastClient.build_query(gpu_name="RTX_4090", num_gpus=2, max_price=1.2, limit=10)
    assert q["gpu_name"] == {"in": ["RTX 4090"]}  # underscore -> space
    assert q["num_gpus"] == {"eq": 2}  # exact count, not gte
    assert q["dph_total"] == {"lte": 1.2}
    assert q["rentable"] == {"eq": True}
    assert q["reliability2"] == {"gte": 0.95}  # filter out flaky hosts


def test_validate():
    with pytest.raises(ValueError):
        TrainingJob(name="x", backend="llm").validate()  # llm needs base_model
    with pytest.raises(ValueError):
        TrainingJob(name="x", backend="mixle").validate()  # mixle needs a script
    # valid ones
    TrainingJob(name="x", backend="llm", base_model="Qwen/Qwen2.5-0.5B", mode="onstart").validate()
    TrainingJob(name="x", backend="mixle", script="train.py", workdir=".").validate()


def test_training_command_mixle_and_llm():
    m = training_command(TrainingJob(name="m", backend="mixle", script="t.py", workdir=".", dataset="d.jsonl"))
    assert m.startswith("python") and "t.py" in m and "--dataset" in m and "--output" in m
    llm = training_command(
        TrainingJob(name="l", backend="llm", base_model="Qwen/Qwen2.5-0.5B", dataset="d.jsonl", qlora=True)
    )
    assert "llm_lora_train.py" in llm and "--base_model" in llm and "--qlora" in llm


def test_training_command_resume_from():
    without = training_command(TrainingJob(name="l", backend="llm", base_model="Qwen/Qwen2.5-0.5B"))
    assert "--resume_from" not in without  # unset by default -> no flag, fresh LoRA init
    with_resume = training_command(
        TrainingJob(name="l", backend="llm", base_model="Qwen/Qwen2.5-0.5B", resume_from="/artifacts/prior-adapter")
    )
    assert "--resume_from" in with_resume and "/artifacts/prior-adapter" in with_resume


def test_llm_lora_script_supports_sft_and_resume_from():
    from mixle_mlops.compute.jobspec import LLM_LORA_SCRIPT

    # SFT path: chat-template + completion-only masking on a "messages" dataset
    assert '"messages" in ds.column_names' in LLM_LORA_SCRIPT
    assert "apply_chat_template" in LLM_LORA_SCRIPT
    assert "-100" in LLM_LORA_SCRIPT  # the ignored-loss label id
    assert "DataCollatorForSeq2Seq" in LLM_LORA_SCRIPT  # pads labels with -100, not the pad token
    # CPT path: unchanged flat-text behavior stays the fallback when there's no "messages" column
    assert 'text_col = "text" if "text" in ds.column_names else ds.column_names[0]' in LLM_LORA_SCRIPT
    assert "DataCollatorForLanguageModeling" in LLM_LORA_SCRIPT
    # resume_from: continue an existing adapter instead of a fresh LoraConfig
    assert "--resume_from" in LLM_LORA_SCRIPT
    assert "PeftModel.from_pretrained(model, a.resume_from, is_trainable=True)" in LLM_LORA_SCRIPT

    # every generated script must still be syntactically valid Python
    compile(LLM_LORA_SCRIPT, "<llm_lora_train.py>", "exec")


def test_pip_packages():
    mixle_pkgs = pip_packages(TrainingJob(name="m", backend="mixle", script="t.py", workdir="."))
    assert any("mixle" in p for p in mixle_pkgs)  # installed from git by default
    assert ["pomegranate"] == pip_packages(
        TrainingJob(name="m", backend="mixle", script="t.py", workdir=".", mixle_spec="pomegranate")
    )
    llm = pip_packages(TrainingJob(name="l", backend="llm", base_model="x"))
    assert any("peft" in p for p in llm) and any("transformers" in p for p in llm)


def test_onstart_script():
    # mixle onstart clones the repo and runs the script
    s = build_onstart(TrainingJob(name="m", backend="mixle", mode="onstart", repo="https://git/x.git", script="t.py"))
    assert "git clone" in s and "https://git/x.git" in s and "python" in s and "touch /workspace/DONE" in s
    # llm onstart writes the LoRA script + can upload to s3
    s2 = build_onstart(
        TrainingJob(name="l", backend="llm", mode="onstart", base_model="Qwen/Qwen2.5-0.5B"),
        s3_dest="s3://bucket/runs/l/",
    )
    assert "llm_lora_train.py" in s2 and "aws s3 cp" in s2 and "s3://bucket/runs/l/" in s2


def test_plan_and_dry_run():
    job = TrainingJob(name="demo", backend="llm", base_model="Qwen/Qwen2.5-0.5B", dataset="d.jsonl", gpu="RTX_4090")
    p = plan(job)
    assert p["image"] and p["offer_query"]["gpu_name"] == {"in": ["RTX 4090"]}
    assert "llm_lora_train.py" in p["training_command"]

    logs: list[str] = []
    out = launch(job, dry_run=True, on_log=logs.append)
    assert out["dry_run"] is True
    joined = "\n".join(logs)
    assert "DRY RUN" in joined and "offer search" in joined and "train command" in joined
