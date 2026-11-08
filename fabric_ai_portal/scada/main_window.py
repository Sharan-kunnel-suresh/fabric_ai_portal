"""MAIN WINDOW - process control, state, KPIs, navigation, event log."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import config
from plc.tags import STATES
from scada.plc_window import PLCWindow
from scada.process_windows import ConveyorWindow, InspectionWindow
from scada.stats_window import StatsWindow
from scada.training_window import TrainingWindow
from scada.widgets import Lamp
from scada.worker import PLCWorker

STATE_COLOR = {0: "#95a5a6", 1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#e74c3c"}
SOURCES = {"Simulated PLC (local)": config.OPCUA_ENDPOINT, "Real S7-1500 (edit address)": "opc.tcp://192.168.0.1:4840"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabric AI Portal - SCADA")
        self.resize(820, 620)
        self.worker = PLCWorker()
        self.windows = {}
        self.prev_state = None

        # --- status ---
        self.state = QLabel("OFFLINE"); self.state.setAlignment(Qt.AlignCenter); self.state.setFont(QFont("Arial", 24, QFont.Bold))
        self.conn = QLabel("connecting...")
        self.lamps = {k: Lamp(c, k) for k, c in (("RUN", "#2ecc71"), ("FAULT", "#e74c3c"), ("CAMERA", "#f1c40f"), ("PUSHER", "#e67e22"))}

        # --- control group ---
        ctl = QGroupBox("Process control")
        b_start, b_stop, b_rst = QPushButton("START"), QPushButton("STOP"), QPushButton("RESET")
        self.b_estop = QPushButton("E-STOP"); self.b_estop.setCheckable(True)
        b_start.setStyleSheet("background:#27ae60; color:white; font-weight:bold; padding:12px;")
        b_stop.setStyleSheet("background:#7f8c8d; color:white; font-weight:bold; padding:12px;")
        self.b_estop.setStyleSheet("QPushButton{background:#c0392b; color:white; font-weight:bold; padding:12px;}"
                                   "QPushButton:checked{background:#641e16;}")
        b_rst.setStyleSheet("padding:12px;")
        b_start.clicked.connect(lambda: self.worker.write("Start_Cmd", True))
        b_stop.clicked.connect(lambda: self.worker.write("Stop_Cmd", True))
        b_rst.clicked.connect(lambda: self.worker.write("Reset_Cmd", True))
        self.b_estop.toggled.connect(lambda on: self.worker.write("EStop", on))
        cl = QHBoxLayout(ctl)
        for b in (b_start, b_stop, self.b_estop, b_rst):
            cl.addWidget(b)

        # --- PLC source group (ComboBox) ---
        src = QGroupBox("PLC connection (OPC UA)")
        self.src_combo = QComboBox(); self.src_combo.addItems(SOURCES)
        self.endpoint = QLineEdit(config.OPCUA_ENDPOINT)
        b_conn = QPushButton("Connect")
        self.src_combo.currentTextChanged.connect(lambda k: self.endpoint.setText(SOURCES[k]))
        b_conn.clicked.connect(lambda: self.worker.set_endpoint(self.endpoint.text().strip(), self.src_combo.currentIndex() == 1))
        sl = QHBoxLayout(src)
        for w in (self.src_combo, self.endpoint, b_conn):
            sl.addWidget(w)
        sl.setStretch(1, 1)

        # --- KPI group ---
        kp = QGroupBox("Production")
        self.k = {n: QLabel("0") for n in ("Inspected", "Good", "Rejected", "Defect rate")}
        kl = QGridLayout(kp)
        for i, (n, lbl) in enumerate(self.k.items()):
            lbl.setFont(QFont("Arial", 22, QFont.Bold)); lbl.setAlignment(Qt.AlignCenter)
            kl.addWidget(QLabel(n), 0, i, Qt.AlignCenter); kl.addWidget(lbl, 1, i)

        # --- navigation group ---
        nav = QGroupBox("Screens")
        nl = QGridLayout(nav)
        for i, (name, key) in enumerate([("Process 1: AI inspection", "insp"), ("Process 2: Conveyor", "conv"),
                                         ("Statistics", "stats"), ("PLC simulation", "plc"), ("AI training results", "train")]):
            b = QPushButton(name); b.clicked.connect(lambda _, k=key: self.open(k)); nl.addWidget(b, i // 3, i % 3)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(300)
        central = QWidget(); lay = QVBoxLayout(central)
        row = QHBoxLayout(); row.addWidget(self.state, 1)
        for l in self.lamps.values():
            row.addWidget(l)
        for w in (self.conn,):
            lay.addWidget(w)
        lay.addLayout(row)
        for w in (ctl, src, kp, nav, self.log):
            lay.addWidget(w)
        self.setCentralWidget(central)

        self.worker.tags.connect(self.on_tags)
        self.worker.connection.connect(self.on_conn)
        self.worker.classified.connect(self.on_ai)
        self.worker.start()

    # ---------- slots ----------
    def open(self, key):
        factories = {"insp": InspectionWindow, "conv": ConveyorWindow, "stats": StatsWindow, "plc": PLCWindow, "train": TrainingWindow}
        if key not in self.windows:
            self.windows[key] = factories[key]() if key == "train" else factories[key](self.worker)
        self.windows[key].show(); self.windows[key].raise_()

    def on_conn(self, ok, msg):
        self.conn.setText(("● " if ok else "○ ") + msg)
        self.conn.setStyleSheet(f"color:{'#2ecc71' if ok else '#e74c3c'};")
        self.log.appendPlainText(msg)
        if not ok:
            self.state.setText("OFFLINE"); self.state.setStyleSheet("color:#95a5a6;")

    def on_tags(self, t):
        s = t["State"]
        self.state.setText(STATES[s]); self.state.setStyleSheet(f"color:{STATE_COLOR[s]};")
        self.lamps["RUN"].set_on(t["Lamp_Green"]); self.lamps["FAULT"].set_on(t["Lamp_Red"])
        self.lamps["CAMERA"].set_on(t["Camera_Trigger"]); self.lamps["PUSHER"].set_on(t["Reject_Pusher"])
        tot = t["Cnt_Total"]
        self.k["Inspected"].setText(str(tot)); self.k["Good"].setText(str(t["Cnt_Good"]))
        self.k["Rejected"].setText(str(t["Cnt_Rejected"]))
        self.k["Defect rate"].setText(f"{100 * t['Cnt_Rejected'] / tot:.0f}%" if tot else "-")
        if s != self.prev_state:
            self.log.appendPlainText(f"PLC state -> {STATES[s]}")
            if s in (4, 5) and self.prev_state is not None:
                box = QMessageBox(QMessageBox.Warning, "PLC alarm",
                                  "EMERGENCY STOP active." if s == 4 else "FAULT: AI did not answer in time.\nPress RESET.", parent=self)
                box.setAttribute(Qt.WA_DeleteOnClose); box.setModal(False); box.show()
            self.prev_state = s

    def on_ai(self, r):
        self.log.appendPlainText(f"item #{r['item_id']}: AI says {r['label'].upper()} (p_defect={r['p_defect']:.2f}, {r['latency_ms']:.1f} ms)")

    def closeEvent(self, e):
        for w in self.windows.values():
            w.close()
        self.worker.stop()
        e.accept()
