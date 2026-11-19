"""Start the SCADA.  Add --with-sim to also start the simulated PLC in the background.

    python run_scada.py --with-sim
"""
import subprocess
import sys

from PyQt5.QtWidgets import QApplication

from scada.main_window import MainWindow

if __name__ == "__main__":
    sim = None
    if "--with-sim" in sys.argv:
        sim = subprocess.Popen([sys.executable, "-m", "plc.opcua_server"])
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    code = app.exec_()
    if sim:
        sim.terminate()
    sys.exit(code)
