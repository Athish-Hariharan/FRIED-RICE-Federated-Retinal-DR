from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import numpy as np


class AccuracyPlot(Canvas):
    """
    Live global-accuracy line chart.

    Fixes vs original accuracy_plot.py:
    - Tick/spine colors were reset on every clear(); now set once after clear
    - Axes were fully redrawn each round (slow); now uses set_data() for the
      main line and replaces only the smoothed/baseline artists
    - Added smoothed overlay and centralized baseline (from plots.py)
    - Grid added
    - Y-axis sensibly bounded (50–100 %)

    New features:
    - reset() clears history for a new training run
    - Per-client accuracy lines shown as faint traces when provided
    """

    CENTRALIZED_BASELINE = 85.79   # reference accuracy from centralized training

    def __init__(self):
        fig = Figure(facecolor="#0b0f14")
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

        self.rounds      = []
        self.accs        = []
        self._client_histories = {}   # {client_idx: [acc, ...]}

        self._init_axes()

        # Persistent line objects — updated via set_data() to avoid full redraw
        self._line_raw,   = self.ax.plot([], [], color="#00ffcc",
                                         linewidth=2, label="Global (raw)", zorder=3)
        self._line_smooth,= self.ax.plot([], [], color="#ffaa00",
                                         linewidth=2.5, linestyle="--",
                                         label="Smoothed", zorder=4)
        self._baseline    = self.ax.axhline(
            self.CENTRALIZED_BASELINE,
            color="#ff4444", linewidth=1.2, linestyle=":",
            label=f"Centralized ({self.CENTRALIZED_BASELINE:.1f}%)", zorder=2
        )
        self._client_lines = {}

        self.ax.legend(
            facecolor="#0b0f14", edgecolor="#1e2d3d",
            labelcolor="white", fontsize=9, loc="lower right"
        )

    # ------------------------------------------------------------------
    def _init_axes(self):
        self.ax.set_facecolor("#0b0f14")
        self.ax.set_title("Global Accuracy", color="white", fontsize=12, pad=8)
        self.ax.set_xlabel("Round", color="#8899a6", fontsize=10)
        self.ax.set_ylabel("Accuracy (%)", color="#8899a6", fontsize=10)
        self.ax.set_ylim(50, 100)
        self.ax.tick_params(colors="#8899a6", labelsize=9)
        self.ax.grid(color="#1e2d3d", linewidth=0.6, linestyle="--")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#1e2d3d")

    # ------------------------------------------------------------------
    def update_plot(self, r, acc, client_acc=None):
        """
        r          : round number (int)
        acc        : global accuracy (float)
        client_acc : list of per-client accuracies (optional)
        """
        self.rounds.append(r)
        self.accs.append(acc)

        # ---- Main raw line ----
        self._line_raw.set_data(self.rounds, self.accs)

        # ---- Smoothed line ----
        # Pad with edge values before convolving so mode='valid' returns the
        # same length as the input — avoids the end-of-series dip that
        # mode='same' produces when the window overlaps with zeros.
        if len(self.accs) >= 3:
            window = 3
            padded = np.pad(self.accs, window // 2, mode="edge")
            smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
            smooth = smooth[:len(self.accs)]
            self._line_smooth.set_data(self.rounds, smooth)
        else:
            self._line_smooth.set_data(self.rounds, self.accs)

        # ---- Optional per-client traces ----
        if client_acc:
            colors = ["#3a86ff", "#ff006e", "#fb5607",
                      "#8338ec", "#06d6a0", "#118ab2"]
            for i, ca in enumerate(client_acc):
                if i not in self._client_histories:
                    self._client_histories[i] = []
                    color = colors[i % len(colors)]
                    line, = self.ax.plot(
                        [], [], color=color, linewidth=1,
                        alpha=0.45, linestyle="-",
                        label=f"C{i+1}", zorder=2
                    )
                    self._client_lines[i] = line
                    # Refresh legend when new client line added
                    self.ax.legend(
                        facecolor="#0b0f14", edgecolor="#1e2d3d",
                        labelcolor="white", fontsize=9, loc="lower right"
                    )
                self._client_histories[i].append(ca)
                self._client_lines[i].set_data(
                    self.rounds[:len(self._client_histories[i])],
                    self._client_histories[i]
                )

        # Rescale x-axis
        self.ax.set_xlim(1, max(self.rounds[-1] + 1, 5))

        self.draw_idle()   # more efficient than draw()

    # ------------------------------------------------------------------
    def set_baseline(self, baseline: float):
        """Update the centralized baseline reference line."""
        self.CENTRALIZED_BASELINE = baseline
        self._baseline.set_ydata([baseline, baseline])
        # Update legend label
        self.ax.legend(
            facecolor="#0b0f14", edgecolor="#1e2d3d",
            labelcolor="white", fontsize=9, loc="lower right"
        )
        self.draw_idle()

    def reset(self):
        """Clear all history — call before starting a new training run."""
        self.rounds = []
        self.accs   = []
        self._client_histories = {}

        self._line_raw.set_data([], [])
        self._line_smooth.set_data([], [])

        for line in self._client_lines.values():
            line.remove()
        self._client_lines = {}

        self.ax.legend(
            facecolor="#0b0f14", edgecolor="#1e2d3d",
            labelcolor="white", fontsize=9, loc="lower right"
        )
        self.draw_idle()
