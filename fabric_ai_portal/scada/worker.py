"""Background thread = the glue between OPC UA (PLC) and PyTorch (AI).
    PLC --Camera_Trigger--> worker --image--> model --prediction--> worker --AI_Result/AI_Done--> PLC
"""
import queue
import random
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

import config
from plc.client import PLCClient


class PLCWorker(QThread):
    tags = pyqtSignal(dict)            # all PLC tags, every cycle
    connection = pyqtSignal(bool, str)  # connected?, message
    classified = pyqtSignal(dict)      # one AI decision (also stored in self.events)

    def __init__(self, endpoint=config.OPCUA_ENDPOINT, parent=None):
        super().__init__(parent)
        self.endpoint, self.real = endpoint, False
        self.cmds = queue.Queue()
        self.running = True
        self.events = []               # history for the statistics window
        self.classifier = None
        self.last_tags = {}
        self._last_item = -1
        self._reconnect = False
        self._samples = self._load_samples()

    # ---------- API used by the GUI thread ----------
    def write(self, name, value):
        self.cmds.put((name, value))

    def set_endpoint(self, endpoint, real=False):
        self.endpoint, self.real = endpoint, real
        self._reconnect = True

    def stop(self):
        self.running = False
        self.wait(3000)

    # ---------- helpers ----------
    def _load_samples(self):
        if (config.OUT_DIR / "splits.json").exists():
            from ai.data_prep import load_splits
            return load_splits()["test"]
        return [str(p) for p in config.DATA_DIR.rglob("*.*")]

    @staticmethod
    def truth_of(path):
        return "defect" if "defect" in Path(path).parent.name.lower() else "good"

    def classify_path(self, path):
        res = self.classifier.predict(path)
        res["path"], res["truth"] = str(path), self.truth_of(path)
        return res

    def _handle_trigger(self, t):
        """Camera_Trigger is True and we did not yet process this item -> run the AI, answer the PLC."""
        item = t["Item_Id"]
        if not (t["Camera_Trigger"] and item != self._last_item):
            return
        self._last_item = item
        if self.classifier is None or not self._samples:
            return                      # PLC will time out into FAULT (shows the fail-safe behaviour)
        res = self.classify_path(random.choice(self._samples))   # stand-in for a camera grab
        res.update(item_id=item, time=time.strftime("%H:%M:%S"))
        conf = res["p_defect"] if res["is_defect"] else res["p_good"]
        self.write("AI_Result", 1 if res["is_defect"] else 0)
        self.write("AI_Confidence", float(conf))
        self.write("AI_Done", True)
        self.events.append(res)
        self.classified.emit(res)

    # ---------- thread body ----------
    def run(self):
        try:
            from ai.inference import Classifier
            self.classifier = Classifier()
        except Exception as e:                                # model missing -> GUI still works
            self.connection.emit(False, f"AI model not loaded: {e}")
        plc = None
        while self.running:
            try:
                if plc is None or self._reconnect:
                    if plc:
                        plc.disconnect()
                    self._reconnect = False
                    plc = PLCClient(self.endpoint, real=self.real)
                    plc.connect()
                    self.connection.emit(True, f"connected to {self.endpoint}")
                while not self.cmds.empty():
                    plc.write(*self.cmds.get())
                t = plc.read_all()
                self.last_tags = t
                self.tags.emit(t)
                self._handle_trigger(t)
                time.sleep(0.1)
            except Exception as e:
                if plc:
                    plc.disconnect()
                plc = None
                self.connection.emit(False, f"PLC offline ({type(e).__name__}) - retrying")
                for _ in range(20):
                    if not self.running or self._reconnect:
                        break
                    time.sleep(0.1)
        if plc:
            plc.disconnect()
