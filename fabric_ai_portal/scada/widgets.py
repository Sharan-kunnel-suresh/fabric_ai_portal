"""Custom-painted widgets (QPainter): signal lamp and animated conveyor."""
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from plc.tags import CAMERA_POS, GATE_POS, STATES


class Lamp(QWidget):
    def __init__(self, color="#2ecc71", text="", parent=None):
        super().__init__(parent)
        self.color, self.text, self.on = QColor(color), text, False
        self.setMinimumSize(70, 60)

    def set_on(self, on):
        if on != self.on:
            self.on = on
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = min(self.width(), self.height() - 18) - 8
        cx = self.width() / 2
        p.setPen(QPen(QColor("#555"), 2))
        p.setBrush(QBrush(self.color if self.on else self.color.darker(300)))
        p.drawEllipse(QRectF(cx - r / 2, 4, r, r))
        p.setPen(self.palette().text().color()); p.setFont(QFont("Arial", 8))
        p.drawText(QRectF(0, r + 8, self.width(), 16), Qt.AlignCenter, self.text)


class ConveyorWidget(QWidget):
    """Animated side view of the inspection line: belt, camera, reject pusher, bins."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t, self.verdict = {}, {}       # verdict: item_id -> True if defect
        self.setMinimumSize(640, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_tags(self, t):
        self.t = t
        self.update()

    def set_verdict(self, item_id, is_defect):
        self.verdict[item_id] = is_defect

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1e2430"))
        t = self.t
        W, H = self.width(), self.height()
        x0, x1 = 40, W - 40
        belt_y = H * 0.55
        px = lambda pos: x0 + (x1 - x0) * pos / 100.0

        # belt + rollers (roller pattern moves with the item position when the motor runs)
        p.setBrush(QColor("#3b4252")); p.setPen(QPen(QColor("#88c0d0"), 2))
        p.drawRoundedRect(QRectF(x0, belt_y, x1 - x0, 22), 10, 10)
        p.setPen(QPen(QColor("#4c566a"), 2))
        off = (t.get("Item_Position", 0) * 4) % 24 if t.get("Motor_Run") else 0
        x = x0 + 12 + off
        while x < x1 - 8:
            p.drawLine(int(x), int(belt_y + 3), int(x), int(belt_y + 19)); x += 24

        # camera station
        cx = px(CAMERA_POS)
        trig = t.get("Camera_Trigger", False)
        p.setBrush(QColor("#ebcb8b") if trig else QColor("#5e81ac")); p.setPen(Qt.NoPen)
        p.drawRect(QRectF(cx - 22, belt_y - 120, 44, 28))
        p.drawRect(QRectF(cx - 3, belt_y - 92, 6, 30))
        if trig:
            p.setBrush(QColor(235, 203, 139, 70))
            p.drawPolygon(*[QPointF(a, b) for a, b in
                            [(cx - 8, belt_y - 62), (cx + 8, belt_y - 62), (cx + 45, belt_y), (cx - 45, belt_y)]])
        p.setPen(QColor("#d8dee9")); p.setFont(QFont("Arial", 9))
        p.drawText(QRectF(cx - 40, belt_y - 148, 80, 18), Qt.AlignCenter, "AI camera")

        # reject pusher + bins
        gx = px(GATE_POS)
        ext = 38 if t.get("Reject_Pusher") else 6
        p.setBrush(QColor("#bf616a") if t.get("Reject_Pusher") else QColor("#7b5560")); p.setPen(Qt.NoPen)
        p.drawRect(QRectF(gx - 12, belt_y - 100 + ext, 24, 6 + 0)); p.drawRect(QRectF(gx - 4, belt_y - 120, 8, 20 + ext))
        p.setPen(QPen(QColor("#bf616a"), 2)); p.setBrush(QColor(191, 97, 106, 40))
        p.drawRect(QRectF(gx - 30, belt_y + 40, 60, 60))
        p.setPen(QPen(QColor("#a3be8c"), 2)); p.setBrush(QColor(163, 190, 140, 40))
        p.drawRect(QRectF(x1 - 30, belt_y + 40, 60, 60))
        p.setPen(QColor("#d8dee9"))
        p.drawText(QRectF(gx - 40, belt_y + 100, 80, 18), Qt.AlignCenter, "REJECT")
        p.drawText(QRectF(x1 - 40, belt_y + 100, 80, 18), Qt.AlignCenter, "OK")

        # item (fabric roll)
        if t.get("Item_Present"):
            pos = t.get("Item_Position", 0)
            item = t.get("Item_Id", 0)
            seen = pos >= CAMERA_POS and item in self.verdict
            col = QColor("#bf616a") if seen and self.verdict[item] else QColor("#a3be8c") if seen else QColor("#d8dee9")
            ix = px(pos)
            dy = 0
            if t.get("Reject_Pusher"):
                dy = 55
            p.setBrush(col); p.setPen(QPen(QColor("#2e3440"), 2))
            p.drawRoundedRect(QRectF(ix - 20, belt_y - 34 + dy, 40, 34), 6, 6)
            p.setPen(QColor("#2e3440")); p.drawText(QRectF(ix - 20, belt_y - 34 + dy, 40, 34), Qt.AlignCenter, f"#{item}")

        # status line
        p.setPen(QColor("#eceff4")); p.setFont(QFont("Arial", 11, QFont.Bold))
        p.drawText(QRectF(12, 8, W - 24, 22), Qt.AlignLeft, f"State: {STATES.get(t.get('State', 0), '?')}"
                   f"    Motor: {'ON' if t.get('Motor_Run') else 'off'}    Speed: {t.get('Belt_Speed', 0):.0f} u/s")
