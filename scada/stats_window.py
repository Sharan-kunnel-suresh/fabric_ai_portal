"""STATISTICS WINDOW - production counters, AI accuracy, trend charts, CSV export."""
import csv

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)


class StatsWindow(QWidget):
    def __init__(self, worker):
        super().__init__()
        self.setWindowTitle("Statistics")
        self.resize(900, 700)
        self.worker = worker
        self.fig = Figure(figsize=(9, 4), tight_layout=True)
        self.canvas = Canvas(self.fig)
        self.ax1, self.ax2, self.ax3 = (self.fig.add_subplot(1, 3, i) for i in (1, 2, 3))
        self.kpi = QLabel(); self.kpi.setStyleSheet("font-size:15px; font-weight:bold;")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Time", "Item", "AI result", "P(defect)", "Ground truth", "ms"])
        self.table.horizontalHeader().setStretchLastSection(True)
        b_exp, b_clr = QPushButton("Export CSV..."), QPushButton("Clear history")
        b_exp.clicked.connect(self.export); b_clr.clicked.connect(self.clear)
        row = QHBoxLayout(); row.addWidget(b_exp); row.addWidget(b_clr); row.addStretch()
        lay = QVBoxLayout(self)
        lay.addWidget(self.kpi); lay.addWidget(self.canvas, 2); lay.addWidget(self.table, 1); lay.addLayout(row)
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(1000)
        self.refresh()

    def refresh(self):
        ev = list(self.worker.events)
        n = len(ev)
        n_def = sum(e["is_defect"] for e in ev)
        correct = sum(e["label"] == e["truth"] for e in ev)
        self.kpi.setText(f"Inspected: {n}    Good: {n - n_def}    Defect: {n_def}    "
                         f"Defect rate: {100 * n_def / max(n, 1):.1f}%    "
                         f"AI accuracy vs ground truth: {100 * correct / max(n, 1):.1f}%")
        a1, a2, a3 = self.ax1, self.ax2, self.ax3
        for a in (a1, a2, a3):
            a.clear()
        a1.bar(["good", "defect"], [n - n_def, n_def], color=["#2ecc71", "#e74c3c"]); a1.set_title("Counts")
        if n:
            d = np.array([e["is_defect"] for e in ev], float)
            k = min(10, n)
            a2.plot(np.convolve(d, np.ones(k) / k, mode="valid") * 100)
            a3.hist([e["p_defect"] for e in ev], bins=10, range=(0, 1), color="#5e81ac")
        a2.set(title="Defect rate % (moving avg)", xlabel="item", ylim=(0, 100))
        a3.set(title="P(defect) distribution", xlabel="probability")
        self.canvas.draw_idle()
        last = ev[-12:][::-1]
        self.table.setRowCount(len(last))
        for r, e in enumerate(last):
            vals = [e["time"], e["item_id"], e["label"], f"{e['p_defect']:.3f}", e["truth"], f"{e['latency_ms']:.1f}"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "inspections.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["time", "item_id", "label", "p_defect", "truth", "latency_ms", "path"])
                for e in self.worker.events:
                    w.writerow([e["time"], e["item_id"], e["label"], f"{e['p_defect']:.4f}", e["truth"],
                                f"{e['latency_ms']:.2f}", e["path"]])

    def clear(self):
        self.worker.events.clear()
        self.refresh()
