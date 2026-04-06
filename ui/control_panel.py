from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QSpinBox, QComboBox,
    QGroupBox, QPlainTextEdit, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont


class ControlPanel(QWidget):
    """
    Left-hand experiment control panel.

    Signals emitted (connect in MainWindow):
        sig_start  : ()  – user pressed Start
        sig_stop   : ()  – user pressed Stop
        sig_export : ()  – user pressed Export JSON
        sig_replay : (str filepath) – user loaded a JSON file

    Properties (read by MainWindow before starting):
        rounds, clients, lr, replay_speed
    """

    sig_start  = pyqtSignal()
    sig_stop   = pyqtSignal()
    sig_export = pyqtSignal()
    sig_replay = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(270)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Title ────────────────────────────────────────────────────
        title = QLabel("FRIED-RICE")
        title.setFont(QFont("Courier New", 13, QFont.Bold))
        title.setStyleSheet("color: #00d4ff; letter-spacing: 3px;")
        root.addWidget(title)

        sub = QLabel("Federated Research Dashboard")
        sub.setStyleSheet("color: #4a6278; font-size: 10px;")
        root.addWidget(sub)

        root.addSpacing(4)

        # ── Live stats ───────────────────────────────────────────────
        stats_box = QGroupBox("Live Stats")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(4)

        self._lbl_round  = self._stat_row(stats_layout, "Round",    "0 / 0")
        self._lbl_acc    = self._stat_row(stats_layout, "Global Acc", "—")
        self._lbl_drift  = self._stat_row(stats_layout, "Avg Drift", "—")
        self._lbl_status = self._stat_row(stats_layout, "Status",   "Idle")

        root.addWidget(stats_box)

        # ── Config ───────────────────────────────────────────────────
        cfg_box = QGroupBox("Config")
        cfg_layout = QVBoxLayout(cfg_box)
        cfg_layout.setSpacing(6)

        self._sld_rounds,  self._lbl_rounds  = self._slider_row(
            cfg_layout, "Rounds", 5, 50, 20, step=5)
        self._sld_clients, self._lbl_clients = self._slider_row(
            cfg_layout, "Clients", 2, 6, 3)
        self._sld_lr,      self._lbl_lr      = self._slider_row(
            cfg_layout, "LR (×0.001)", 1, 50, 10)

        # ── Dirichlet α (non-IID control) ────────────────────────────
        # α=1→moderate non-IID, α=10→near IID, α=0.1→extreme non-IID
        # Slider maps 1..20 → α = value * 0.1  (range 0.1 .. 2.0)
        self._sld_alpha, self._lbl_alpha = self._slider_row(
            cfg_layout, "Dirichlet α", 1, 20, 5, step=1,
            fmt=lambda v: f"{v*0.1:.1f}")
        alpha_hint = QLabel("α↓ = more non-IID  |  α↑ = more IID")
        alpha_hint.setStyleSheet("color:#2a4058; font-size:9px; margin-left:4px;")
        cfg_layout.addWidget(alpha_hint)

        # ── Algorithm selector ───────────────────────────────────────
        algo_row = QHBoxLayout()
        algo_lbl = QLabel("Algorithm")
        algo_lbl.setStyleSheet("color:#8899a6; font-size:11px;")
        algo_lbl.setFixedWidth(110)
        self._algo_combo = QComboBox()
        self._algo_combo.addItems(["FedAvg", "FedProx", "DriftAware"])
        self._algo_combo.setCurrentIndex(0)
        self._algo_combo.setStyleSheet(
            "QComboBox { font-size:11px; color:#00d4ff; "
            "background:#111720; border:1px solid #1e2d3d; "
            "border-radius:3px; padding:2px 6px; }"
            "QComboBox QAbstractItemView { background:#111720; color:#e2e8f0; "
            "selection-background-color:#1e2d3d; }"
            "QComboBox::drop-down { border:none; }"
        )
        algo_row.addWidget(algo_lbl)
        algo_row.addWidget(self._algo_combo, 1)
        cfg_layout.addLayout(algo_row)

        self._sld_speed,   self._lbl_speed   = self._slider_row(
            cfg_layout, "Replay Speed", 1, 10, 3)

        root.addWidget(cfg_box)

        # ── Buttons ──────────────────────────────────────────────────
        self._btn_start = QPushButton("▶  Start Training")
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("stop_btn")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.sig_stop)

        self._btn_export = QPushButton("⬇  Export JSON")
        self._btn_export.setObjectName("export_btn")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self.sig_export)

        self._btn_replay = QPushButton("↺  Replay JSON")
        self._btn_replay.clicked.connect(self._on_load_json)

        for btn in (self._btn_start, self._btn_stop,
                    self._btn_export, self._btn_replay):
            root.addWidget(btn)

        # ── Log ──────────────────────────────────────────────────────
        log_box = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_box)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setFont(QFont("Courier New", 9))
        log_layout.addWidget(self._log)
        root.addWidget(log_box)

        root.addStretch()

        self.log("System ready.")

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _stat_row(parent_layout, label: str, default: str):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #4a6278; font-size: 11px;")
        val = QLabel(default)
        val.setStyleSheet("color: #00ffcc; font-size: 12px; font-weight: bold;")
        val.setAlignment(Qt.AlignRight)
        row.addWidget(lbl)
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    @staticmethod
    def _slider_row(parent_layout, label: str,
                    lo: int, hi: int, default: int, step: int = 1,
                    fmt=None):
        """Returns (QSlider, QLabel-for-value). fmt(v) formats the display label."""
        row = QHBoxLayout()

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #8899a6; font-size: 11px;")
        lbl.setFixedWidth(110)

        sld = QSlider(Qt.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(default)
        sld.setSingleStep(step)
        sld.setPageStep(step)

        init_text = fmt(default) if fmt else str(default)
        val_lbl = QLabel(init_text)
        val_lbl.setStyleSheet("color: #00d4ff; font-size: 11px;")
        val_lbl.setFixedWidth(34)
        val_lbl.setAlignment(Qt.AlignRight)

        if fmt:
            sld.valueChanged.connect(lambda v, l=val_lbl, f=fmt: l.setText(f(v)))
        else:
            sld.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))

        row.addWidget(lbl)
        row.addWidget(sld)
        row.addWidget(val_lbl)
        parent_layout.addLayout(row)
        return sld, val_lbl

    # ── Public read-only properties ──────────────────────────────────

    @property
    def rounds(self) -> int:
        return self._sld_rounds.value()

    @property
    def clients(self) -> int:
        return self._sld_clients.value()

    @property
    def lr(self) -> float:
        return self._sld_lr.value() * 0.001

    @property
    def alpha(self) -> float:
        """Dirichlet concentration: slider 1-20 → α 0.1-2.0."""
        return self._sld_alpha.value() * 0.1

    @property
    def algorithm(self) -> str:
        return self._algo_combo.currentText()

    @property
    def replay_speed(self) -> float:
        return self._sld_speed.value() * 1.0

    # ── Public update methods (called by MainWindow) ─────────────────

    def update_stats(self, round_num: int, total_rounds: int,
                     global_acc: float, avg_drift: float, status: str):
        self._lbl_round.setText(f"{round_num} / {total_rounds}")
        self._lbl_acc.setText(f"{global_acc:.2f}%")
        self._lbl_drift.setText(f"{avg_drift:.4f}")
        self._lbl_status.setText(status)

    def set_training_active(self, active: bool):
        """Toggle button states when training starts/stops."""
        self._btn_start.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        self._btn_replay.setEnabled(not active)
        self._btn_export.setEnabled(not active)
        # Lock sliders during training
        for sld in (self._sld_rounds, self._sld_clients,
                    self._sld_lr, self._sld_alpha):
            sld.setEnabled(not active)
        self._algo_combo.setEnabled(not active)

    def enable_export(self, enabled: bool = True):
        self._btn_export.setEnabled(enabled)

    def log(self, message: str, level: str = "info"):
        ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        prefix = {"info": "·", "ok": "✓", "warn": "⚠", "err": "✗"}.get(level, "·")
        self._log.appendPlainText(f"[{ts}] {prefix} {message}")
        # Scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Private slots ────────────────────────────────────────────────

    def _on_start(self):
        self.set_training_active(True)
        self.sig_start.emit()

    def _on_load_json(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Experiment JSON", "", "JSON Files (*.json)"
        )
        if filepath:
            self.log(f"Loading: {filepath}", "ok")
            self.sig_replay.emit(filepath)
