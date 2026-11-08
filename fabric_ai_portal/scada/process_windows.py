"""PROCESS WINDOWS (at least 2): 1) AI inspection  2) Conveyor line."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)

from scada.widgets import ConveyorWidget, Lamp


class InspectionWindow(QWidget):
    """Process window 1 - what the camera sees and what the AI decides."""

    def __init__(self, worker):
        super().__init__()
        self.setWindowTitle("Process 1 - AI fabric inspection")
        self.worker = worker
        self.resize(760, 520)

        self.image = QLabel("waiting for the first item...")
        self.image.setFixedSize(300, 300); self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("background:#222; color:#aaa; border:2px solid #555;")
        self.verdict = QLabel("-"); self.verdict.setFont(QFont("Arial", 26, QFont.Bold)); self.verdict.setAlignment(Qt.AlignCenter)
        self.info = QLabel(""); self.info.setWordWrap(True)
        self.bar_good, self.bar_def = QProgressBar(), QProgressBar()
        for b, c in ((self.bar_good, "#2ecc71"), (self.bar_def, "#e74c3c")):
            b.setRange(0, 100); b.setStyleSheet(f"QProgressBar::chunk{{background:{c};}}")
        self.history = QListWidget()
        btn = QPushButton("Classify an image from file...")
        btn.clicked.connect(self.manual)

        box = QGroupBox("AI decision")
        bl = QVBoxLayout(box)
        for w in (self.verdict, QLabel("P(good)"), self.bar_good, QLabel("P(defect)"), self.bar_def, self.info, btn):
            bl.addWidget(w)
        top = QHBoxLayout(); top.addWidget(self.image); top.addWidget(box, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addWidget(QLabel("Last inspections")); lay.addWidget(self.history)
        worker.classified.connect(self.show_result)

    def show_result(self, r, manual=False):
        pm = QPixmap(r["path"]).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pm)
        defect = r["is_defect"]
        self.verdict.setText("DEFECT" if defect else "GOOD")
        self.verdict.setStyleSheet(f"color:{'#e74c3c' if defect else '#2ecc71'};")
        self.bar_good.setValue(int(r["p_good"] * 100)); self.bar_def.setValue(int(r["p_defect"] * 100))
        item = "manual" if manual else f"item #{r['item_id']}"
        self.info.setText(f"{item}   threshold {r['threshold']:.2f}   latency {r['latency_ms']:.1f} ms\n"
                          f"ground truth: {r['truth']}  ->  {'correct' if r['truth'] == r['label'] else 'WRONG'}")
        self.history.insertItem(0, f"{r.get('time', '--')}  {item}: {r['label'].upper()} ({r['p_defect']:.2f})")
        while self.history.count() > 12:
            self.history.takeItem(self.history.count() - 1)

    def manual(self):
        if self.worker.classifier is None:
            QMessageBox.warning(self, "AI", "The AI model is not loaded (train it first).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.show_result(self.worker.classify_path(path), manual=True)


class ConveyorWindow(QWidget):
    """Process window 2 - animated line with lamps."""

    def __init__(self, worker):
        super().__init__()
        self.setWindowTitle("Process 2 - Conveyor & reject station")
        self.resize(820, 420)
        self.conv = ConveyorWidget()
        self.lg, self.lr, self.lp, self.lc = (Lamp("#2ecc71", "RUN"), Lamp("#e74c3c", "FAULT"),
                                              Lamp("#e67e22", "PUSHER"), Lamp("#f1c40f", "CAMERA"))
        row = QHBoxLayout()
        for l in (self.lg, self.lr, self.lc, self.lp):
            row.addWidget(l)
        lay = QVBoxLayout(self); lay.addWidget(self.conv, 1); lay.addLayout(row)
        worker.tags.connect(self.update_tags)
        worker.classified.connect(lambda r: self.conv.set_verdict(r["item_id"], r["is_defect"]))

    def update_tags(self, t):
        self.conv.set_tags(t)
        self.lg.set_on(t["Lamp_Green"]); self.lr.set_on(t["Lamp_Red"])
        self.lp.set_on(t["Reject_Pusher"]); self.lc.set_on(t["Camera_Trigger"])
