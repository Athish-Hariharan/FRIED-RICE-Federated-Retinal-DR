from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QStatusBar, QTabWidget
)
from PyQt5.QtCore import QDateTime

from ui.training_worker    import TrainingWorker
from ui.accuracy_plot      import AccuracyPlot
from ui.network_canvas     import NetworkCanvas
from ui.control_panel      import ControlPanel
from ui.replay_engine      import ReplayEngine
from ui.export_manager     import ExportManager
from ui.dr_diagnostic_page import DRDiagnosticPage
# from ui.improvements_page  import ImprovementsPage
from ui.research_analysis_page import ResearchAnalysisPage


class MainWindow(QMainWindow):
    """
    Top-level window — two tabs:
      1. Training Dashboard  — controls, network graph, accuracy plot
      2. DR Diagnostic       — model metrics + fundus image grader

    Signal wiring changes:
      - TrainingWorker.round_signal now carries 9 arguments:
        (round, global_acc, drifts, client_sizes, client_acc, sources,
         iid_scores, kappas, label_dists)
      - TrainingWorker.baseline_signal(float) wired to DR page and accuracy plot
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FRIED-RICE  ·  Federated Research Dashboard")
        self.showMaximized()

        self._worker          = None
        self._replay          = None
        self._export          = ExportManager(rounds=20, clients=3)
        self._total_rounds    = 20
        self._last_global_acc = 87.0
        self._centralized_baseline = 85.79

        # Sub-pages
        self._training_tab  = self._build_training_tab()
        self._dr_page       = DRDiagnosticPage()
        self._research_page = ResearchAnalysisPage()
        # self._improvements  = ImprovementsPage()

        # Top-level tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border:none; border-top:1px solid #1e2d3d; }
            QTabBar::tab {
                background:#050a0f; color:#4a6278;
                padding:9px 28px; border:none;
                font-size:12px; letter-spacing:1px;
            }
            QTabBar::tab:selected { color:#00d4ff; border-bottom:2px solid #00d4ff; background:#0b0f14; }
            QTabBar::tab:hover    { color:#8899a6; }
        """)
        self._tabs.addTab(self._training_tab,  "  Training Dashboard  ")
        self._tabs.addTab(self._dr_page,       "  DR Diagnostic  ")
        self._tabs.addTab(self._research_page, "  Research Analysis  ")
        # self._tabs.addTab(self._improvements,  "  Improvements  ")
        self.setCentralWidget(self._tabs)

        self._status = QStatusBar()
        self._status.showMessage("Ready  ·  Configure experiment and press Start Training")
        self.setStatusBar(self._status)

    # ── Training tab ──────────────────────────────────────────────────────────

    def _build_training_tab(self) -> QWidget:
        self._controls = ControlPanel()
        self._plot     = AccuracyPlot()
        self._network  = NetworkCanvas(clients=3)

        self._controls.sig_start.connect(self._start_training)
        self._controls.sig_stop.connect(self._stop_training)
        self._controls.sig_export.connect(self._export_json)
        self._controls.sig_replay.connect(self._start_replay)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.setRowStretch(0, 2)
        layout.setRowStretch(1, 3)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 3)
        layout.addWidget(self._controls, 0, 0, 2, 1)
        layout.addWidget(self._network,  0, 1)
        layout.addWidget(self._plot,     1, 1)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    # ── Training lifecycle ────────────────────────────────────────────────────

    def _start_training(self):
        self._hard_stop()
        n      = self._controls.clients
        rounds = self._controls.rounds
        lr     = self._controls.lr
        self._total_rounds = rounds

        self._plot.reset()
        self._research_page.reset_live()
        self._export = ExportManager(rounds=rounds, clients=n, lr=lr)

        layout = self._training_tab.layout()
        layout.removeWidget(self._network)
        self._network.deleteLater()
        self._network = NetworkCanvas(clients=n)
        layout.addWidget(self._network, 0, 1)

        self._worker = TrainingWorker(
            rounds=rounds, clients=n, lr=lr,
            alpha=self._controls.alpha,
            algorithm=self._controls.algorithm,
        )
        self._worker.baseline_signal.connect(self._on_baseline)
        self._worker.round_signal.connect(self._on_round)
        self._worker.algo_metrics_signal.connect(self._on_algo_metrics)
        self._worker.finished_signal.connect(self._on_training_finished)
        self._worker.start()

        # Trigger background comparison simulations
        self._research_page.trigger_precompute(
            rounds=rounds, alpha=self._controls.alpha,
            clients=n, algorithm=self._controls.algorithm,
        )

        self._controls.log(
            f"Training started — {rounds} rounds, {n} clients, "
            f"LR={lr:.3f}, α={self._controls.alpha:.1f}, "
            f"algo={self._controls.algorithm}", "ok"
        )
        self._status.showMessage("Training in progress …")

    def _on_algo_metrics(self, metrics: dict):
        """Receive per-round algorithm comparison metrics."""
        pass   # available for future extension / DR page display

    def _stop_training(self):
        self._hard_stop()
        self._controls.set_training_active(False)
        self._controls.log("Training stopped by user.", "warn")
        self._status.showMessage("Training stopped.")
        if self._export.round_count > 0:
            self._controls.enable_export(True)

    def _hard_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        if self._replay and self._replay.isRunning():
            self._replay.stop()
            self._replay.wait(2000)

    # ── Signals ───────────────────────────────────────────────────────────────

    def _on_baseline(self, baseline: float):
        """Emitted once before training starts — computed from data config."""
        self._centralized_baseline = baseline
        self._plot.set_baseline(baseline)
        self._dr_page.set_centralized_baseline(baseline)
        self._controls.log(f"Centralized baseline estimated: {baseline:.2f}%", "ok")

    def _on_round(self, r, acc, drifts, client_sizes, client_acc, sources,
                  iid_scores, kappas, label_dists):
        n = self._network.clients

        # Pad/truncate all lists to match client count
        def _fit(lst, default):
            return (list(lst) + [default]*n)[:n]

        drifts       = _fit(drifts,       0.5)
        client_sizes = _fit(client_sizes, 1000)
        client_acc   = _fit(client_acc,   65.0)
        sources      = _fit(sources,      "Unknown")
        iid_scores   = _fit(iid_scores,   0.0)
        kappas       = _fit(kappas,       0.0)
        label_dists  = _fit(label_dists,  [200]*5)

        self._plot.update_plot(r, acc, client_acc=client_acc)
        self._network.update_network(
            drifts,
            client_sizes=client_sizes,
            client_acc=client_acc,
            sources=sources,
            iid_scores=iid_scores,
            kappas=kappas,
            label_dists=label_dists,
            current_round=r,
            total_rounds=self._total_rounds,
        )
        self._export.record(r, acc, drifts, client_sizes, client_acc, sources)

        avg_drift = sum(drifts) / len(drifts) if drifts else 0.0
        avg_iid   = sum(iid_scores) / len(iid_scores) if iid_scores else 0.0
        is_last   = (r >= self._total_rounds)
        status    = "Complete" if is_last else (
            "Replaying" if self._replay and self._replay.isRunning() else "Training"
        )
        self._controls.update_stats(
            round_num=r, total_rounds=self._total_rounds,
            global_acc=acc, avg_drift=avg_drift, status=status,
        )
        self._status.showMessage(
            f"Round {r}/{self._total_rounds}  ·  "
            f"Acc: {acc:.2f}%  ·  "
            f"Drift: {avg_drift:.4f}  ·  "
            f"Avg non-IID: {avg_iid:.3f}  ·  "
            f"{QDateTime.currentDateTime().toString('hh:mm:ss')}"
        )
        self._last_global_acc = acc
        self._dr_page.set_global_accuracy(acc)
        self._research_page.update_round(
            r, acc, client_acc, drifts, iid_scores, kappas,
            algorithm=self._controls.algorithm,
        )

    def _on_training_finished(self):
        self._controls.set_training_active(False)
        self._controls.enable_export(True)
        self._controls.log("Training complete.", "ok")
        self._status.showMessage(
            f"Training finished — {self._export.round_count} rounds recorded. "
            "Use 'Export JSON' to save."
        )
        self._dr_page.set_global_accuracy(self._last_global_acc)

    # ── Replay ────────────────────────────────────────────────────────────────

    def _start_replay(self, filepath: str):
        self._hard_stop()
        speed              = self._controls.replay_speed
        self._total_rounds = 0
        self._plot.reset()

        try:
            self._replay = ReplayEngine(filepath, speed=speed)
        except Exception as exc:
            self._controls.log(f"Replay error: {exc}", "err")
            self._controls.set_training_active(False)
            return

        self._total_rounds = len(self._replay._rounds)

        if self._replay._rounds:
            first = self._replay._rounds[0]
            n = len(first.get("sources") or first.get("drifts") or [])
            n = n if n > 0 else 3
            if n != self._network.clients:
                layout = self._training_tab.layout()
                layout.removeWidget(self._network)
                self._network.deleteLater()
                self._network = NetworkCanvas(clients=n)
                layout.addWidget(self._network, 0, 1)

        self._replay.round_signal.connect(self._on_replay_round)
        self._replay.finished_signal.connect(self._on_replay_finished)
        self._replay.start()
        self._controls.set_training_active(True)
        self._controls.log(f"Replaying {self._total_rounds} rounds at {speed:.1f}×", "ok")
        self._status.showMessage(f"Replaying: {filepath}")

    def _on_replay_round(self, r, acc, drifts, client_sizes, client_acc, sources):
        """Replay engine emits 6-arg signal — pad with defaults for new fields."""
        n = self._network.clients
        iid_scores  = [0.0] * n
        kappas      = [0.0] * n
        label_dists = [[200]*5] * n
        self._on_round(r, acc, drifts, client_sizes, client_acc, sources,
                       iid_scores, kappas, label_dists)

    def _on_replay_finished(self):
        self._controls.set_training_active(False)
        self._controls.log("Replay complete.", "ok")
        self._status.showMessage("Replay finished.")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_json(self):
        from PyQt5.QtWidgets import QFileDialog
        import os
        default_name = (
            f"fried_rice_"
            f"{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json"
        )
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Experiment JSON",
            os.path.join(os.path.expanduser("~"), default_name),
            "JSON Files (*.json)"
        )
        if not filepath:
            return
        try:
            saved = self._export.save(filepath)
            self._controls.log(f"Exported: {os.path.basename(saved)}", "ok")
            self._status.showMessage(f"Saved to: {saved}")
        except IOError as exc:
            self._controls.log(f"Export failed: {exc}", "err")

    def closeEvent(self, event):
        self._hard_stop()
        event.accept()
