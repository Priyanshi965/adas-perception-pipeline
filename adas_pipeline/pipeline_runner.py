"""
pipeline_runner.py — Orchestrator that runs each stage in order,
saves checkpoints after each, and supports resuming from failures.
"""

import json
import logging
import os
import pickle
import traceback
from datetime import datetime
from typing import Callable, Dict, List, Optional

import config

# Keys whose values are large lists/dicts — stored in a sidecar .pkl instead of the JSON
_LARGE_KEYS = {"frame_records", "track_history"}


def _checkpoint_path(stage_name: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{stage_name}.json")


def _sidecar_path(stage_name: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{stage_name}.pkl")


def _load_checkpoint(stage_name: str) -> Optional[Dict]:
    path = _checkpoint_path(stage_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.getLogger("PipelineRunner").warning(
            f"Checkpoint '{stage_name}' is corrupt ({e}) — treating as missing"
        )
        # Remove the bad file so it doesn't block future runs
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    # Restore large keys from sidecar pickle if present
    pkl = _sidecar_path(stage_name)
    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                sidecar = pickle.load(f)
            data.update(sidecar)
        except Exception as e:
            logging.getLogger("PipelineRunner").warning(
                f"Could not load sidecar pickle for '{stage_name}': {e}"
            )

    return data


def _save_checkpoint(stage_name: str, output_path: str, extra: Optional[Dict] = None):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    data = {
        "stage": stage_name,
        "status": "done",
        "output_path": output_path,
        "completed_at": datetime.now().isoformat(),
    }

    sidecar = {}
    if extra:
        for k, v in extra.items():
            if k in _LARGE_KEYS:
                sidecar[k] = v
            else:
                data[k] = v

    # Write lightweight JSON checkpoint
    with open(_checkpoint_path(stage_name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Write large data to sidecar pickle
    if sidecar:
        with open(_sidecar_path(stage_name), "wb") as f:
            pickle.dump(sidecar, f)


def _setup_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to the pipeline log file only.
    Console output is handled by the root logger (configured in main.py),
    so we disable propagation to avoid duplicate console lines.
    """
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # root logger handles console; we only add file handler

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # File handler
    fh = logging.FileHandler(
        os.path.join(config.LOGS_DIR, "pipeline.log"), encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler with ASCII-safe encoding fallback for Windows cp1252
    import sys
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    sh.stream = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        errors="replace",
        closefd=False,
        buffering=1,
    )
    logger.addHandler(sh)

    return logger


class PipelineRunner:
    """
    Runs a list of (stage_name, stage_fn) pairs in order.

    Each stage_fn receives (context: dict) and must return a dict that
    gets merged into the shared context for downstream stages.

    Usage:
        runner = PipelineRunner(stages, resume=True)
        runner.run(initial_context)
    """

    def __init__(self, stages: List[tuple], resume: bool = False):
        """
        Args:
            stages: list of (name: str, fn: Callable[[dict], dict])
            resume:  if True, skip stages that already have a checkpoint
        """
        self.stages = stages
        self.resume = resume
        self.logger = _setup_logger("PipelineRunner")

    def run(self, initial_context: Optional[Dict] = None) -> Dict:
        context = initial_context or {}

        # Pre-load any existing checkpoints into context so resumed stages
        # have access to prior stages' outputs.
        for name, _ in self.stages:
            cp = _load_checkpoint(name)
            if cp:
                context[f"_checkpoint_{name}"] = cp

        for name, fn in self.stages:
            cp = _load_checkpoint(name)

            if self.resume and cp and cp.get("status") == "done":
                self.logger.info(f"[SKIP] Stage '{name}' already completed - resuming.")
                # Merge all checkpoint data (including sidecar large keys) into context
                for k, v in cp.items():
                    if k not in {"stage", "status", "completed_at"}:
                        context[k] = v
                continue

            self.logger.info(f"[START] Stage: {name}")
            try:
                result = fn(context)
            except Exception as e:
                self.logger.error(
                    f"[FAIL] Stage '{name}' raised an exception:\n"
                    + traceback.format_exc()
                )
                raise RuntimeError(f"Pipeline failed at stage '{name}': {e}") from e

            # result must be a dict
            if result is None:
                result = {}

            output_path = result.get("output_path", "")
            _save_checkpoint(name, output_path, extra=result)
            context.update(result)
            self.logger.info(f"[DONE] Stage: {name} -> {output_path}")

        self.logger.info("Pipeline complete.")
        return context

    def clear_checkpoints(self):
        """Delete all checkpoint files — forces a full re-run."""
        for name, _ in self.stages:
            for path in (_checkpoint_path(name), _sidecar_path(name)):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        self.logger.info(f"Cleared checkp