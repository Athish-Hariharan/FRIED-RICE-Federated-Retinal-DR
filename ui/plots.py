"""
plots.py — DEPRECATED

The smoothing + baseline logic from this file has been merged into
ui.accuracy_plot.AccuracyPlot. This file is kept only to avoid
import errors in any external code.
"""
import warnings
warnings.warn(
    "ui.plots is deprecated. Use ui.accuracy_plot.AccuracyPlot instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export so any existing `from ui.plots import AccuracyPlot` still works
from ui.accuracy_plot import AccuracyPlot  # noqa: F401, E402

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.figure import Figure
# import numpy as np
# 
# class AccuracyPlot(FigureCanvasQTAgg):
# 
    # def __init__(self):
        # self.fig = Figure(facecolor="#111111")
        # super().__init__(self.fig)
# 
        # self.ax = self.fig.add_subplot(111)
        # self.ax.set_facecolor("#111111")
# 
        # self.rounds = []
        # self.acc = []
# 
        # self.baseline = 85.79   # centralized reference
# 
        # self.ax.set_title("Global Accuracy", color="white")
        # self.ax.set_xlabel("Round", color="white")
        # self.ax.set_ylabel("Accuracy (%)", color="white")
# 
        # self.ax.tick_params(colors="white")
        # self.ax.grid(color="#333333")
# 
    # def update_plot(self, r, acc):
# 
        # self.rounds.append(r)
        # self.acc.append(acc)
# 
        # self.ax.clear()
        # self.ax.set_facecolor("#111111")
# 
        # # smoothing
        # smooth = np.convolve(self.acc, np.ones(3)/3, mode='same')
# 
        # self.ax.plot(self.rounds, self.acc,
                     # color="#00ffaa", linewidth=2, label="Raw")
# 
        # self.ax.plot(self.rounds, smooth,
                     # color="#ffaa00", linewidth=3, label="Smoothed")
# 
        # self.ax.axhline(self.baseline,
                        # color="#ff4444",
                        # linestyle="--",
                        # label="Centralized")
# 
        # self.ax.legend(facecolor="#111111", edgecolor="white")
# 
        # self.ax.set_title("Global Accuracy", color="white")
        # self.ax.set_xlabel("Round", color="white")
        # self.ax.set_ylabel("Accuracy (%)", color="white")
        # self.ax.tick_params(colors="white")
        # self.ax.grid(color="#333333")
# 
        # self.draw()
