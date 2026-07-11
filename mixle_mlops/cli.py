"""Command-line entry points: run the gateway (`mixle-serve`) and admin tasks (`mixle-mlops ...`)."""
from __future__ import annotations

import argparse
import json
import os

from .config import get_settings


def serve() -> None:
    import uvicorn

    get_settings()  # validate config early
    host = os.environ.get("MIXLE_HOST", "0.0.0.0")
    port = int(os.environ.get("MIXLE_PORT", "8000"))
    reload = os.environ.get("MIXLE_RELOAD", "0") == "1"
    uvicorn.run("mixle_mlops.gateway.app:app", host=host, port=port, reload=reload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mixle-mlops", description="mixle-mlops platform admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the gateway")
    sub.add_parser("init-db", help="create database tables")
    cu = sub.add_parser("create-user", help="create a user and print an API key")
    cu.add_argument("email")
    cu.add_argument("password")
    cu.add_argument("--admin", action="store_true")
    ic = sub.add_parser("init-cloud", help="scaffold a provider-correct .env (object store + DB + redis)")
    ic.add_argument("provider", choices=["aws", "azure", "gcp", "alicloud", "local"])
    ic.add_argument("--dest", default=".env")
    ic.add_argument("--force", action="store_true", help="overwrite an existing .env")

    tr = sub.add_parser("train", help="train a mixle model or fine-tune an LLM on a rented vast.ai GPU")
    tr.add_argument("name", help="model/run name")
    tr.add_argument("--backend", choices=["mixle", "llm"], default="mixle")
    tr.add_argument("--mode", choices=["ssh", "onstart"], default="ssh")
    tr.add_argument("--workdir", help="local dir to upload (mixle backend, ssh mode)")
    tr.add_argument("--repo", help="git URL to clone (mixle backend, onstart mode)")
    tr.add_argument("--script", help="training entry script (mixle backend)")
    tr.add_argument("--dataset", help="jsonl path (ssh) or HF dataset id / URL (onstart)")
    tr.add_argument("--base-model", dest="base_model", help="HF model id (llm backend)")
    tr.add_argument("--epochs", type=float, default=1.0)
    tr.add_argument("--qlora", action="store_true", help="4-bit QLoRA (llm backend)")
    tr.add_argument("--gpu", default=None, help="GPU name filter, e.g. RTX_4090")
    tr.add_argument("--num-gpus", dest="num_gpus", type=int, default=1)
    tr.add_argument("--max-price", dest="max_price", type=float, default=None, help="$/hr cap")
    tr.add_argument("--disk", type=int, default=None, help="disk GB")
    tr.add_argument("--max-runtime", dest="max_runtime", type=int, default=60,
                    help="hard cap (minutes): destroy the box after this long no matter what")
    tr.add_argument("--image", default=None, help="docker image override")
    tr.add_argument("--mixle-git", dest="mixle_spec", default=None,
                    help="pip spec for the mixle core on the box (default: pinned PyPI release; env MIXLE_GIT)")
    tr.add_argument("--requirements", "--req", dest="requirements", action="append", default=[],
                    metavar="PKG", help="extra pip package to install on the box (repeatable)")
    tr.add_argument("--no-register", dest="register", action="store_false")
    tr.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="actually rent a GPU and train")
    tr.add_argument("--local", action="store_true", help="run on THIS machine (no vast.ai) — validate before renting")
    tr.add_argument("--s3-dest", dest="s3_dest", default=None, help="object-store URL for onstart artifacts")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        serve()
    elif args.cmd == "init-db":
        from .storage.db import init_db

        init_db()
        print("database initialized")
    elif args.cmd == "create-user":
        from sqlmodel import Session

        from .accounts import service
        from .storage.db import get_engine, init_db

        init_db()
        with Session(get_engine()) as session:
            user = service.create_user(session, args.email, args.password, is_admin=args.admin)
            _key, raw = service.create_api_key(session, user)
            print(f"user {user.email} created")
            print(f"api key: {raw}")
    elif args.cmd == "init-cloud":
        from .cloud_init import init_cloud, next_steps

        path = init_cloud(args.provider, dest=args.dest, overwrite=args.force)
        print(f"wrote {path}")
        print(next_steps(args.provider))
    elif args.cmd == "train":
        from .compute import TrainingJob, launch, run_local
        from .config import get_settings

        s = get_settings()
        job = TrainingJob(
            name=args.name,
            backend=args.backend,
            mode=args.mode,
            workdir=args.workdir,
            repo=args.repo,
            script=args.script,
            dataset=args.dataset,
            base_model=args.base_model,
            epochs=args.epochs,
            qlora=args.qlora,
            gpu=args.gpu or s.vast_default_gpu,
            num_gpus=args.num_gpus,
            max_price=args.max_price if args.max_price is not None else s.vast_max_price,
            disk=args.disk if args.disk is not None else s.vast_default_disk,
            image=args.image,
            requirements=args.requirements,
            register=args.register,
            max_runtime_min=args.max_runtime,
            **({"mixle_spec": ms} if (ms := args.mixle_spec or os.environ.get("MIXLE_GIT")) else {}),
        )
        if args.local:
            result = run_local(job, registry_root=str(s.registry_root))
            print(json.dumps(result, indent=2, default=str))
        else:
            result = launch(
                job,
                api_key=s.vast_api_key,
                dry_run=args.dry_run,
                registry_root=str(s.registry_root),
                s3_dest=args.s3_dest,
            )
            if not args.dry_run:
                print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
