import json
import os
from datetime import datetime


class ExportManager:
    """
    Collects per-round data during training and exports to JSON.

    Usage:
        em = ExportManager(rounds=20, clients=3)
        em.record(r, global_acc, drifts, client_sizes, client_acc, sources)
        ...
        em.save("/path/to/file.json")

    The saved file is compatible with ReplayEngine's full format.
    """

    def __init__(self, rounds: int, clients: int,
                 lr: float = 0.01, notes: str = ""):
        self._meta = {
            "experiment": "FRIED-RICE Federated Learning",
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "rounds":     rounds,
            "clients":    clients,
            "lr":         lr,
            "notes":      notes,
        }
        self._rounds: list = []

    # ------------------------------------------------------------------
    def record(self, round_num: int, global_acc: float,
               drifts: list, client_sizes: list,
               client_acc: list, sources: list):
        """Append one round's data."""
        self._rounds.append({
            "round":           round_num,
            "global_accuracy": round(global_acc, 4),
            "drifts":          [round(d, 4) for d in drifts],
            "client_sizes":    client_sizes,
            "client_acc":      [round(a, 4) for a in client_acc],
            "sources":         sources,
        })

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> str:
        """
        Write JSON to filepath.
        Returns the absolute path of the written file.
        Raises IOError on permission/disk errors.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        payload = {
            "metadata": self._meta,
            "rounds":   self._rounds,
            # Convenience flat list for quick plotting / legacy tools
            "round_accuracy": [r["global_accuracy"] for r in self._rounds],
        }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    def reset(self):
        """Clear collected rounds (keeps metadata template)."""
        self._rounds = []

    # ------------------------------------------------------------------
    @property
    def round_count(self) -> int:
        return len(self._rounds)
