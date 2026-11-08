"""PLC SIMULATION WINDOW - live tag table (like TIA 'watch table') + force/write controls."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from plc.tags import STATES, TAGS


class PLCWindow(QWidget):
    def __init__(self, worker):
        super().__init__()
        self.setWindowTitle("PLC simulation (OPC UA watch table)")
        self.resize(760, 640)
        self.worker = worker
        self.table = QTableWidget(len(TAGS), 4)
        self.table.setHorizontalHeaderLabels(["Tag", "Value", "Direction", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.rows = {}
        for r, (name, (_, _, writable, desc)) in enumerate(TAGS.items()):
            self.rows[name] = r
            for c, v in enumerate([name, "", "SCADA -> PLC" if writable else "PLC -> SCADA", desc]):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.state = QLabel(); self.state.setStyleSheet("font-size:16px; font-weight:bold;")

        w = QGroupBox("Write / force a tag")
        self.combo = QComboBox(); self.combo.addItems([n for n, v in TAGS.items() if v[2]])
        self.edit = QLineEdit("1"); self.edit.setPlaceholderText("value (1/0 for bool)")
        go = QPushButton("Write"); go.clicked.connect(self.write)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 100); self.slider.setValue(20)
        self.slider.valueChanged.connect(lambda v: (worker.write("Belt_Speed", float(v)), self.sl_lbl.setText(f"Belt speed: {v}")))
        self.sl_lbl = QLabel("Belt speed: 20")
        row = QHBoxLayout()
        for x in (self.combo, self.edit, go):
            row.addWidget(x)
        wl = QVBoxLayout(w); wl.addLayout(row); wl.addWidget(self.sl_lbl); wl.addWidget(self.slider)
        lay = QVBoxLayout(self); lay.addWidget(self.state); lay.addWidget(self.table); lay.addWidget(w)
        worker.tags.connect(self.update_tags)

    def update_tags(self, t):
        for name, v in t.items():
            it = self.table.item(self.rows[name], 1)
            it.setText(f"{v:.1f}" if isinstance(v, float) else str(v))
            it.setBackground(QColor("#1f6f3f") if v is True else QColor("#2b2b2b") if v is False else QColor("#00000000"))
            it.setForeground(QColor("white") if isinstance(v, bool) else QColor("#dddddd"))
        self.state.setText(f"PLC state: {STATES.get(t['State'], '?')}")

    def write(self):
        name, vt = self.combo.currentText(), TAGS[self.combo.currentText()][1]
        txt = self.edit.text().strip().lower()
        try:
            if vt.name == "Boolean":
                v = txt in ("1", "true", "yes", "on")
            elif vt.name == "Float":
                v = float(txt)
            else:
                v = int(float(txt))
            self.worker.write(name, v)
        except ValueError:
            self.edit.setText("invalid")
