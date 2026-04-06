import json
import time

from PyQt5.QtCore import QThread, pyqtSignal


class ReplayEngine(QThread):
    """
    Replays a saved JSON experiment file.

    Fixes vs original replay_engine.py:
    - Was calling Qt UI methods from a raw threading.Thread → not thread-safe,
      caused random crashes. Now subclasses QThread and emits a Qt signal
      instead of calling the callback directly.
    - Handles both the minimal format  {"round_accuracy": [...]}
      and the full format saved by ExportManager.
    - speed parameter exposed; default 1.0 × real-time.

    Expected full JSON format (produced by ExportManager):
    {
        "metadata": { "rounds": N, "clients": K, ... },
        "rounds": [
            {
                "round": 1,
                "global_accuracy": 72.3,
                "drifts": [0.4, 0.3, 0.5],
                "client_sizes": [2000, 1600, 1200],
                "client_acc": [70.1, 68.4, 66.0],
                "sources": ["EyePACS", "APTOS", "Messidor"]
            },
            ...
        ]
    }

    Minimal legacy format:
    {
        "round_accuracy": [72.3, 74.1, ...]
    }
    """

    # Matches TrainingWorker.round_signal signature exactly
    round_signal = pyqtSignal(int, float, list, list, list, list)

    def __init__(self, json_file: str, speed: float = 1.0):
        super().__init__()
        self.speed    = max(0.1, speed)
        self._running = True

        with open(json_file) as f:
            raw = json.load(f)

        # Normalise both formats into a unified list of round dicts
        self._rounds = self._parse(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(data: dict) -> list:
        """Return list of normalised round dicts regardless of input format."""

        # ---- Full format ----
        if "rounds" in data:
            out = []
            for entry in data["rounds"]:
                out.append({
                    "round":           entry.get("round", len(out) + 1),
                    "global_accuracy": float(entry.get("global_accuracy", 0)),
                    "drifts":          entry.get("drifts", []),
                    "client_sizes":    entry.get("client_sizes", []),
                    "client_acc":      entry.get("client_acc", []),
                    "sources":         entry.get("sources", []),
                })
            return out

        # ---- Legacy minimal format ----
        if "round_accuracy" in data:
            return [
                {
                    "round":           i + 1,
                    "global_accuracy": float(acc),
                    "drifts":          [],
                    "client_sizes":    [],
                    "client_acc":      [],
                    "sources":         [],
                }
                for i, acc in enumerate(data["round_accuracy"])
            ]

        raise ValueError(
            "Unrecognised JSON format. Expected 'rounds' or 'round_accuracy' key."
        )

    # ------------------------------------------------------------------
    def run(self):
        delay = 1.0 / self.speed
        for entry in self._rounds:
            if not self._running:
                break
            self.round_signal.emit(
                entry["round"],
                entry["global_accuracy"],
                entry["drifts"],
                entry["client_sizes"],
                entry["client_acc"],
                entry["sources"],
            )
            time.sleep(delay)

    def stop(self):
        self._running = False
