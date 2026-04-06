"""
dashboard.py — DEPRECATED

This file was an earlier prototype and is no longer used by the application.
MainWindow (main_window.py) is the current entry point.

Kept here to avoid import errors if any external code references it.
"""
import warnings
warnings.warn(
    "ui.dashboard is deprecated and unused. "
    "Use ui.main_window.MainWindow instead.",
    DeprecationWarning,
    stacklevel=2,
)

# import sys
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# 
# import matplotlib
# matplotlib.use("Qt5Agg")
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# 
# 
# class AccuracyPlot(FigureCanvas):
# 
    # def __init__(self):
        # self.fig, self.ax = plt.subplots(facecolor="#121212")
        # super().__init__(self.fig)
# 
        # self.ax.set_facecolor("#121212")
        # self.ax.tick_params(colors="white")
        # self.ax.spines['bottom'].set_color('white')
        # self.ax.spines['left'].set_color('white')
# 
        # self.rounds = []
        # self.acc = []
# 
        # self.line, = self.ax.plot([], [], color="#00FFAA", linewidth=3)
# 
        # self.ax.set_title("Global Accuracy", color="white")
        # self.ax.set_xlabel("Round", color="white")
        # self.ax.set_ylabel("Accuracy", color="white")
# 
    # def update_plot(self, r, a):
        # self.rounds.append(r)
        # self.acc.append(a)
# 
        # self.line.set_data(self.rounds, self.acc)
        # self.ax.relim()
        # self.ax.autoscale_view()
        # self.draw()
# 
# 
# class FederatedGraph(QWidget):
# 
    # def __init__(self):
        # super().__init__()
        # self.round = 0
# 
    # def paintEvent(self, event):
# 
        # painter = QPainter(self)
        # painter.setRenderHint(QPainter.Antialiasing)
# 
        # w = self.width()
        # h = self.height()
# 
        # # background
        # painter.fillRect(self.rect(), QColor("#121212"))
# 
        # pen = QPen(QColor("#00FFAA"), 3)
        # painter.setPen(pen)
# 
        # # server
        # server = QPoint(w//2, h//2)
# 
        # painter.setBrush(QColor("#00FFAA"))
        # painter.drawEllipse(server, 40, 40)
        # painter.drawText(server + QPoint(-30, 70), "SERVER")
# 
        # # clients
        # clients = [
            # QPoint(200, 150),
            # QPoint(w-200, 150),
            # QPoint(w//2, h-150)
        # ]
# 
        # painter.setBrush(QColor("#FFAA00"))
# 
        # for i, c in enumerate(clients):
            # painter.drawEllipse(c, 30, 30)
            # painter.drawLine(c, server)
            # painter.drawText(c + QPoint(-30, 50), f"Client {i+1}")
# 
        # painter.setPen(QColor("white"))
        # painter.setFont(QFont("Arial", 20))
        # painter.drawText(40, 40, f"Round: {self.round}")
# 
# 
# class Dashboard(QMainWindow):
# 
    # def __init__(self):
        # super().__init__()
# 
        # self.setWindowTitle("FRIED RICE Federated Dashboard")
# 
        # self.setStyleSheet("""
            # background-color: #121212;
            # color: white;
        # """)
# 
        # central = QWidget()
        # layout = QHBoxLayout()
# 
        # self.graph = FederatedGraph()
        # self.plot = AccuracyPlot()
# 
        # layout.addWidget(self.graph, 2)
        # layout.addWidget(self.plot, 3)
# 
        # central.setLayout(layout)
        # self.setCentralWidget(central)
# 
        # self.showFullScreen()
# 
    # def update_round(self, r, acc):
        # self.graph.round = r
        # self.graph.update()
# 
        # self.plot.update_plot(r, acc)

