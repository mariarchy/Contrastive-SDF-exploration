"""Persistent Modal classifier workers and their local process controls."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.awareness.evaluate import now
from contrastive_sdf.evals.awareness.refinement.engine import BudgetStop


class ModalTarget:
    """A dedicated interpreter avoids the Modal module's global target binding."""

    def __init__(self, output: Path, name: str, manifest: dict):
        self.name, self.output = name, output
        self.started = time.monotonic()
        self.finished = None
        self.log = (output / f"modal-{name}.log").open("a")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "contrastive_sdf.evals.awareness.refinement.modal_worker",
                "--output",
                str(output),
                "--target",
                name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.events = queue.Queue()
        self.manifest = manifest
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                self.log.write(line)
                self.log.flush()
        self.events.put({"error": "Modal worker exited; inspect its log"})

    def receive(self, guard):
        while True:
            guard()
            try:
                result = self.events.get(timeout=1)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
            except queue.Empty:
                pass

    def send(self, data):
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            try:
                self.send({"stop": True})
                self.process.wait(timeout=20)
            except BrokenPipeError, subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.finished = time.monotonic()
        self.log.close()


class ModalClassifiers:
    def __init__(self, output: Path, manifest: dict):
        self.output, self.manifest = output, manifest
        self.workers = {}
        self.previous = (
            json.loads((output / "costs.json").read_text())
            if (output / "costs.json").exists()
            else {}
        )
        self.prior_estimate = self.previous.get("modal", {}).get(
            "budget_accounted_usd", 0
        )
        self.watchdog_stop = threading.Event()

    def estimate(self):
        return self.prior_estimate + sum(
            (
                (w.finished or time.monotonic())
                - w.started
                + self.manifest["config"]["modal_shutdown_reserve_seconds"]
            )
            * self.manifest["config"]["modal_usd_per_worker_second"]
            for w in self.workers.values()
        )

    def guard(self):
        if self.estimate() >= self.manifest["config"]["modal_budget_usd"]:
            raise BudgetStop("Conservative Modal resource-time allowance reached")

    def _watchdog(self):
        while not self.watchdog_stop.wait(1):
            try:
                self.guard()
            except BudgetStop:
                # Stop the private apps as well as local processes, including if
                # the parent is in a slow reviser call. App IDs are durable.
                for name in self.workers:
                    self.stop_app(name)
                return

    def start(self):
        self.guard()
        self.workers = {
            name: ModalTarget(self.output, name, self.manifest)
            for name in self.manifest["targets"]
        }
        threading.Thread(target=self._watchdog, daemon=True).start()
        for worker in self.workers.values():
            event = worker.receive(self.guard)
            if event.get("event") != "assets_ready":
                raise RuntimeError("Unexpected worker asset event")
        for worker in self.workers.values():
            worker.send({"load": True})
        for worker in self.workers.values():
            event = worker.receive(self.guard)
            if event.get("event") != "ready":
                raise RuntimeError("Unexpected worker startup event")

    def __call__(self, name: str, requests: list[dict]):
        self.guard()
        worker = self.workers[name]
        worker.send({"requests": requests})
        return worker.receive(self.guard)["results"]

    def stop_app(self, name):
        path = self.output / f"modal-{name}-app.json"
        if path.exists():
            app_id = json.loads(path.read_text())["app_id"]
            result = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", app_id, "--yes"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            atomic_json(
                self.output / f"modal-{name}-stop.json",
                {
                    "at_utc": now(),
                    "app_id": app_id,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )

    def close(self):
        self.watchdog_stop.set()
        for name, worker in self.workers.items():
            worker.close()
            self.stop_app(name)

    def accounting(self):
        return {
            "budget_accounted_usd": self.estimate(),
            "actual_billed_usd": None,
            "note": "Conservative local process wall time × configured resource rate, plus shutdown reserves. Includes queueing/loading/idle time; not an invoice. Shared storage and preparation require billing reconciliation.",
            "previous_sessions": self.previous.get("modal", {}),
        }
