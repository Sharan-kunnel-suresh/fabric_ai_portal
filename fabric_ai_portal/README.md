# Fabric AI Portal

Python-based industrial fabric inspection project: **PyTorch fabric-defect classifier + PyQt5 SCADA + OPC UA PLC (simulated).**

## Project structure

```text
fabric_ai_portal/
│
├── config.py                    # Paths, image size and OPC UA endpoint
├── prepare_dataset.py           # General dataset preparation
├── prepare_tilda.py             # TILDA dataset preparation and group split
├── make_synthetic_dataset.py    # Synthetic images for testing
│
├── ai/
│   ├── data_prep.py              # General data preparation
│   ├── model.py                  # FabricNet neural network
│   ├── train.py                  # General model training
│   ├── evaluate.py               # General model evaluation
│   ├── inference.py              # Model inference
│   ├── tilda_data_prep.py        # TILDA data preparation
│   ├── train_tilda.py            # TILDA V1 training
│   ├── evaluate_tilda.py         # TILDA V1 evaluation
│   ├── evaluate_tilda_v2.py      # TILDA V2 evaluation
│   ├── tilda_category_analysis.py
│   ├── tilda_error_analysis.py
│   └── tilda_group_analysis.py
│
├── plc/
│   ├── tags.py                   # PLC tag definitions
│   ├── opcua_server.py           # Simulated OPC UA PLC
│   └── client.py                 # OPC UA client
│
├── scada/
│   ├── main_window.py            # Main SCADA dashboard
│   ├── process_windows.py        # AI inspection and conveyor windows
│   ├── stats_window.py           # Production statistics
│   ├── plc_window.py             # PLC tag monitoring
│   ├── training_window.py        # AI training results
│   ├── worker.py                 # PLC communication and AI inference
│   └── widgets.py                # Reusable GUI widgets
│
├── data/
│   ├── raw/tilda/                # Original TILDA categories
│   ├── tilda_binary/             # Prepared train/val/test data
│   └── tilda_manifest.csv        # Dataset split information
│
├── outputs/
│   ├── tilda/                    # V1 checkpoint and evaluation results
│   ├── tilda_v2/                 # V2 experiment results
│   └── tilda_data_stats.json     # Normalization statistics
│
├── docs/                         # Project documentation
├── requirements.txt
├── run_scada.py                  # Start SCADA
└── README.md
```

## Run it

The tested setup uses two terminals: one for the simulated PLC and one for SCADA.

```powershell
# 1) Open the project folder
cd path\to\fabric_ai_portal

# 2) Create virtual environment (first time only)
python -m venv .venv

# 3) Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4) Install dependencies (first time only)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Terminal 1 — Start simulated PLC**

```powershell
python -m plc.opcua_server
```

**Terminal 2 — Start SCADA application**

```powershell
python run_scada.py
```

**OR in One terminal**

```powershell
python run_scada.py --with-sim
```

Connect SCADA to:

```text
opc.tcp://127.0.0.1:4840/fabric/plc
```

Keep the PLC server terminal running while using SCADA.

In SCADA, press **Connect**, then **START**. Open the PLC simulation window to monitor tags and trigger inspections. Open **Statistics** to view production counters and inspection results.

## Requirements

- Windows 10/11
- Python 3.11
- Dependencies listed in `requirements.txt`

Required files for the existing TILDA V1 demonstration:

```text
outputs/tilda/best.pt
outputs/tilda_data_stats.json
data/tilda_binary/test/good/
data/tilda_binary/test/defect/
```

These files must be present before launching SCADA.

## Dataset

**Dataset:** TILDA 400 × 64 × 64 fabric patches.

Original categories:

- good
- hole
- objects
- oil spot
- thread error

For the current model, the four defect categories are combined into one `defect` class. The model performs binary classification: `good` or `defect`.

| Original category |     Images |
| ----------------- | ---------: |
| Good              |     23,170 |
| Hole              |        337 |
| Objects           |        837 |
| Oil spot          |        636 |
| Thread error      |        620 |
| **Total**         | **25,600** |

### Dataset preparation

Original TILDA folders are placed under:

```text
data/raw/tilda/
├── good/
├── hole/
├── objects/
├── oil spot/
└── thread error/
```

Prepare the dataset:

```powershell
python prepare_tilda.py
```

The script creates `data/tilda_binary/` and `data/tilda_manifest.csv`.

Patches belonging to the same source image are assigned to the same split to reduce data leakage.

| Split      |      Total |       Good |    Defect |
| ---------- | ---------: | ---------: | --------: |
| Train      |     17,920 |     16,214 |     1,706 |
| Validation |      3,840 |      3,479 |       361 |
| Test       |      3,840 |      3,477 |       363 |
| **Total**  | **25,600** | **23,170** | **2,430** |

Do not rerun preparation unless you intend to rebuild the dataset split.

## AI training and evaluation

The TILDA data pipeline includes:

- Image resizing to 128 × 128.
- Training-set normalization.
- Training data augmentation.
- WeightedRandomSampler for class imbalance.
- Separate training, validation and testing datasets.

Run the following commands:

```powershell
# Train TILDA V1
python -m ai.train_tilda

# Evaluate TILDA V1
python -m ai.evaluate_tilda

# Evaluate TILDA V2
python -m ai.evaluate_tilda_v2

# Run inference on an image
python -m ai.inference "data/tilda_binary/test/good/example.png"
```

The integrated SCADA application uses the existing V1 checkpoint:

```text
outputs/tilda/best.pt
```

Normalization values are loaded from:

```text
outputs/tilda_data_stats.json
```

Training is not required every time SCADA is started.

## Model evaluation — TILDA V1

The V1 classification threshold was selected using the validation set with the Youden J criterion.

| Metric             | Test result |
| ------------------ | ----------: |
| Accuracy           |      88.49% |
| Precision          |      44.44% |
| Recall             |      87.05% |
| F1-score           |      58.85% |
| ROC-AUC            |      0.9472 |
| Selected threshold |      0.3198 |

### V1 vs V2 comparison

| Test metric |     V1 |     V2 |
| ----------- | -----: | -----: |
| ROC-AUC     | 0.9472 | 0.8895 |
| Recall      | 87.05% | 68.04% |
| Precision   | 44.44% | 43.87% |
| F1-score    | 58.85% | 53.35% |
| Accuracy    | 88.49% | 88.75% |

**Selected model: TILDA V1.**

V1 achieved higher test ROC-AUC and defect recall than V2 at the selected threshold. V2 is retained as an experimental comparison.

### Defect category analysis

The deployed model performs binary classification. The following results represent separate analysis of binary predictions grouped by original defect category.

| Defect category | Detected / Total | Recall |
| --------------- | ---------------: | -----: |
| Hole            |          45 / 46 | 97.83% |
| Objects         |        114 / 140 | 81.43% |
| Oil spot        |          79 / 91 | 86.81% |
| Thread error    |          78 / 86 | 90.70% |

## SCADA and PLC integration

The SCADA application is developed using PyQt5.

The background worker in `scada/worker.py` loads the TILDA V1 model, monitors PLC tags and performs image classification when an inspection trigger is received.

### Main PLC tags

| Tag              | Direction   | Description              |
| ---------------- | ----------- | ------------------------ |
| `Start_Cmd`      | SCADA → PLC | Start production         |
| `Stop_Cmd`       | SCADA → PLC | Stop production          |
| `EStop_Cmd`      | SCADA → PLC | Emergency stop           |
| `Reset_Cmd`      | SCADA → PLC | Reset fault/E-stop       |
| `Belt_Speed`     | SCADA → PLC | Conveyor speed           |
| `Camera_Trigger` | PLC → SCADA | Request AI inspection    |
| `AI_Done`        | SCADA → PLC | Classification completed |
| `AI_Result`      | SCADA → PLC | 0 = good, 1 = defect     |
| `AI_Confidence`  | SCADA → PLC | Prediction confidence    |
| `State`          | PLC → SCADA | Production state         |
| `Motor_Run`      | PLC → SCADA | Conveyor motor status    |
| `Reject_Pusher`  | PLC → SCADA | Rejection actuator       |
| `Lamp_Green`     | PLC → SCADA | Normal operation         |
| `Lamp_Red`       | PLC → SCADA | Fault/E-stop             |
| `Item_Present`   | PLC → SCADA | Item on conveyor         |

### SCADA windows

| File                       | Function                                      |
| -------------------------- | --------------------------------------------- |
| `scada/main_window.py`     | Main dashboard and production controls        |
| `scada/process_windows.py` | AI inspection and conveyor views              |
| `scada/stats_window.py`    | Counters, charts and inspection history       |
| `scada/plc_window.py`      | PLC tag monitoring and writing                |
| `scada/training_window.py` | AI training and evaluation information        |
| `scada/widgets.py`         | Reusable GUI components                       |
| `scada/worker.py`          | Background PLC communication and AI inference |

## Grading items and implementation

| Grading item          | Main files                             | Implementation                                             |
| --------------------- | -------------------------------------- | ---------------------------------------------------------- |
| Project description   | `docs/`                                | Project documentation                                      |
| Dataset               | `prepare_tilda.py`                     | TILDA dataset preparation                                  |
| Data preparation      | `ai/tilda_data_prep.py`                | Resize, normalization, augmentation and imbalance handling |
| Model configuration   | `ai/model.py`                          | FabricNet architecture                                     |
| Model training        | `ai/train_tilda.py`                    | Model training and checkpoint saving                       |
| Model evaluation      | `ai/evaluate_tilda.py`                 | Metrics, confusion matrix and ROC-AUC                      |
| Model integration     | `ai/inference.py`, `scada/worker.py`   | AI inference triggered by PLC                              |
| Main window           | `scada/main_window.py`                 | Dashboard and production controls                          |
| Statistics            | `scada/stats_window.py`                | Counters, charts and history                               |
| Process windows       | `scada/process_windows.py`             | AI inspection and conveyor                                 |
| PLC simulation window | `scada/plc_window.py`                  | OPC UA tag monitoring                                      |
| Training window       | `scada/training_window.py`             | Training/evaluation information                            |
| OPC UA simulation     | `plc/opcua_server.py`, `plc/client.py` | Simulated PLC communication                                |

## Testing and limitations

**Tested:**

- TILDA dataset preparation and group-based splitting.
- V1 and V2 training and evaluation.
- Single-image AI inference.
- Loading TILDA V1 inside SCADA.
- Local OPC UA PLC connection.
- PLC tag monitoring and writing.
- AI inspection triggers and classification results.
- SCADA dashboard, process windows and statistics.

**Limitations:**

- **Camera:** The worker selects a random image from the TILDA test set for each inspection. No live camera is connected.
- **PLC:** The simulated OPC UA PLC has been tested. A physical Siemens S7-1500 has not been connected or tested.
- **Classification:** The model predicts `good` or `defect`; it does not identify individual defect types during inference.
- **False alarms:** V1 has relatively low precision at the selected threshold.
- **Production counters:** Counter consistency should be checked before using the statistics as real production measurements.
- **OPC UA security:** The local demonstration endpoint does not have certificates or encryption configured.

## Troubleshooting

**SCADA reports `ConnectionRefusedError`:**

- Make sure `python -m plc.opcua_server` is running.
- Check the endpoint: `opc.tcp://127.0.0.1:4840/fabric/plc`.
- Click Connect again.

**A Python package is missing:**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Model checkpoint not found:**

- Check that `outputs/tilda/best.pt` exists.
- Check that `outputs/tilda_data_stats.json` exists.

**No inspection images found:**

- Check that `data/tilda_binary/test/good/` and `data/tilda_binary/test/defect/` contain images.

## Conclusion

Fabric AI Portal combines a PyTorch fabric-defect classifier, a PyQt5 SCADA application and a simulated OPC UA PLC. The prototype demonstrates dataset-based AI inspection, model inference and communication of inspection results to a simulated industrial monitoring and control system.
