"""ADDITIONAL WINDOW - shows the saved training / evaluation results (plots + metrics)."""
import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QComboBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

import config

PLOTS = {"Loss curves": "loss_curves.png", "Confusion matrix": "confusion_matrix.png", "ROC curve": "roc_curve.png",
         "Threshold sweep": "threshold_sweep.png", "LR finder": "lr_finder.png",
         "First-layer filters": "filters.png", "Feature maps": "feature_maps.png"}


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI model - training & evaluation results")
        self.resize(900, 700)
        self.combo = QComboBox(); self.combo.addItems(PLOTS)
        self.combo.currentTextChanged.connect(self.show_plot)
        self.img = QLabel(); self.img.setAlignment(Qt.AlignCenter)
        self.txt = QPlainTextEdit(); self.txt.setReadOnly(True); self.txt.setMaximumHeight(150)
        m = config.OUT_DIR / "metrics.json"
        if m.exists():
            d = json.loads(m.read_text()); t = d["test_at_chosen_threshold"]
            self.txt.setPlainText(
                f"best epoch {d['best_epoch']}   val loss {d['val_loss']:.4f}   test loss {d['test_loss']:.4f}   AUC {d['auc_test']:.4f}\n"
                f"threshold {t['threshold']:.2f}   TP {t['TP']}  FP {t['FP']}  TN {t['TN']}  FN {t['FN']}\n"
                f"accuracy {t['accuracy']:.3f}  precision {t['precision']:.3f}  recall(TPR) {t['recall']:.3f}  FPR {t['FPR']:.3f}")
        else:
            self.txt.setPlainText("No metrics yet: run  python -m ai.train  and  python -m ai.evaluate")
        lay = QVBoxLayout(self); lay.addWidget(self.combo); lay.addWidget(self.img, 1); lay.addWidget(self.txt)
        self.show_plot(self.combo.currentText())

    def show_plot(self, name):
        f = config.OUT_DIR / PLOTS[name]
        if f.exists():
            self.img.setPixmap(QPixmap(str(f)).scaled(860, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img.setText(f"{f.name} not found")
