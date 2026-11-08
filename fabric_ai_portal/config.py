"""Central configuration for the Fabric AI Portal project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"          # put the unpacked Kaggle download here
DATA_DIR = ROOT / "data" / "dataset"     # cleaned dataset: DATA_DIR/good, DATA_DIR/defect
OUT_DIR = ROOT / "outputs"               # checkpoints, plots, metrics
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["good", "defect"]             # index 0 = good, index 1 = defect (positive class)
IMG_SIZE = 128
SEED = 42

# training defaults
BATCH_SIZE = 32
EPOCHS = 15
VAL_FRAC = 0.15
TEST_FRAC = 0.15
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

# OPC UA (the simulated PLC and the SCADA client both use these)
OPCUA_ENDPOINT = "opc.tcp://127.0.0.1:4840/fabric/plc"
OPCUA_NAMESPACE = "http://fabric.portal/plc"

# Real S7-1500: global data block "FabricPLC" published through the PLC's OPC UA server.
# ns=3 is the usual namespace index of the PLC's own symbols - check it with UaExpert and edit if needed.
REAL_NODE_FMT = 'ns=3;s="FabricPLC"."{tag}"'
