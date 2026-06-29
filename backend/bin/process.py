#!/usr/bin/env python3
"""CLI entry point for SupoClip — process a URL or local file end-to-end.

Usage:
    supoclip-clip <url>
    supoclip-clip /path/to/local.mp4 [--mode fast|quality] [--clips N] [--user-id UUID]

Output:
    - Progress lines on stderr (one JSON object per stage)
    - Final summary as JSON on stdout (task_id + list of clip paths under /app/outputs/<task_id>/)

Designed to be invoked from openhands-lxc via `docker exec supoclip-worker python bin/process.py`.
The .venv is at /app/.venv, so make sure to `source /app/.venv/bin/activate` first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Allow running this script from inside /app without installing the package
WORKDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKDIR))

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from src.config import get_config  # noqa: E402
from src.database import AsyncSessionLocal  # noqa: E402
from src.services.task_service import TaskService  # noqa: E402


def _emit(stage: str, message: str, **extra: Any) -> None:
    """Emit a structured progress line on stderr."""
    payload = {"stage": stage, "message": message, **extra}
    sys.stderr.write(json.dumps(payload, default=str) + "\n")
    sys.stderr.flush()


def _is_url(arg: str) -> bool:
    return arg.startswith("http://") or arg.startswith("https://")


async def _stage_url(url: str, uploads_dir: Path) -> str:
    """Download a remote URL into uploads_dir via yt-dlp. Returns upload:// ref."""
    _emit("download", "fetching", url=url, dest=str(uploads_dir))
    # Delegate to the same yt-dlp logic the worker uses (deno + JS challenge solver
    # is already installed in this image). We import lazily so an offline run
    # without yt-dlp doesn't fail the import chain.
    from src.services.video_service import VideoService

    # yt-dlp writes progress to stdout by default; silence it so our JSON summary
    # stays clean on stdout. Errors still surface via stderr/exception.
    saved_stdout_fd = os.dup(1)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        try:
            video_path = await VideoService.download_video(url)
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.close(devnull)
    finally:
        os.close(saved_stdout_fd)

    if not video_path or not video_path.exists():
        raise RuntimeError(f"yt-dlp failed to download {url}")

    # Move the file under uploads/ with a UUID name and return the upload:// ref.
    unique = f"{uuid.uuid4()}{video_path.suffix or '.mp4'}"
    target = uploads_dir / unique
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(video_path), target)
    _emit("download", "saved", path=str(target), size=target.stat().st_size)
    return f"upload://{unique}"


async def _stage_file(local: Path, uploads_dir: Path) -> str:
    """Copy a local file into uploads_dir. Returns upload:// ref."""
    if not local.exists():
        raise FileNotFoundError(f"No such file: {local}")
    if not local.is_file():
        raise ValueError(f"Not a regular file: {local}")

    uploads_dir.mkdir(parents=True, exist_ok=True)
    ext = local.suffix or ".mp4"
    unique = f"{uuid.uuid4()}{ext}"
    target = uploads_dir / unique
    shutil.copy2(local, target)
    _emit("upload", "copied", source=str(local), target=str(target), size=target.stat().st_size)
    return f"upload://{unique}"


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    uploads_dir = Path(cfg.temp_dir) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    user_id = args.user_id or os.environ.get("SUPOCLIP_CLI_USER_ID")
    if not user_id:
        _emit(
            "error",
            "missing user_id — pass --user-id or set SUPOCLIP_CLI_USER_ID",
        )
        return 2

    # ── Stage 1: get upload:// ref ────────────────────────────────────
    if _is_url(args.source):
        upload_ref = await _stage_url(args.source, uploads_dir)
    else:
        upload_ref = await _stage_file(Path(args.source).expanduser(), uploads_dir)

    # ── Stage 2: create task row ──────────────────────────────────────
    async with AsyncSessionLocal() as db:
        task_svc = TaskService(db=db)

        title = args.title
        if not title:
            if _is_url(args.source):
                from src.services.video_service import VideoService

                title = await VideoService.get_video_title(args.source)
            else:
                title = Path(args.source).stem

        _emit("task", "creating", title=title)
        task_id = await task_svc.create_task_with_source(
            user_id=user_id,
            url=upload_ref,
            title=title,
            processing_mode=args.mode,
        )
        _emit("task", "created", task_id=task_id)

        # ── Stage 3: process (transcribe → analyze → cut clips) ────
        async def _progress(pct: int, msg: str, status: str) -> None:
            _emit("progress", msg, percent=pct, status=status)

        _emit("process", "starting", task_id=task_id, mode=args.mode)
        result = await task_svc.process_task(
            task_id=task_id,
            url=upload_ref,
            source_type="video_url",
            processing_mode=args.mode,
            output_format=args.output_format,
            add_subtitles=not args.no_subtitles,
            progress_callback=_progress,
        )

        # Stage 4: pull clip paths from the DB so we report accurate locations
        from sqlalchemy import select

        from src.models import GeneratedClip

        clip_rows = (
            await db.execute(
                select(GeneratedClip).where(GeneratedClip.task_id == task_id)
            )
        ).scalars().all()
        clips = [
            {
                "id": str(row.id),
                "file_path": row.file_path,
                "virality_score": getattr(row, "virality_score", None),
                "start_time": getattr(row, "start_time", None),
                "end_time": getattr(row, "end_time", None),
            }
            for row in clip_rows
        ]

    summary = {
        "task_id": task_id,
        "source": args.source,
        "upload_ref": upload_ref,
        "processing_mode": args.mode,
        "clips": clips,
        "result": result,
    }
    sys.stdout.write(json.dumps(summary, indent=2, default=str) + "\n")
    sys.stdout.flush()
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the SupoClip pipeline from CLI")
    p.add_argument("source", help="HTTP(S) URL or path to a local video file")
    p.add_argument("--mode", choices=["fast", "quality"], default="fast")
    p.add_argument(
        "--output-format",
        choices=["vertical", "horizontal", "square"],
        default="vertical",
    )
    p.add_argument("--no-subtitles", action="store_true")
    p.add_argument("--title", help="Override the auto-detected title")
    p.add_argument(
        "--user-id",
        help="Owner of the task. Defaults to $SUPOCLIP_CLI_USER_ID. Required.",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress DEBUG-level logs from the worker modules",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    # The internal modules are chatty at INFO; we surface our own progress on
    # stderr above, so keep them at WARNING unless the user opts in.
    if args.quiet:
        for noisy in ("src", "httpx", "httpcore", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        _emit("error", "interrupted")
        code = 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        _emit("error", "fatal", error=str(exc))
        logging.getLogger(__name__).exception("process.py failed")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
