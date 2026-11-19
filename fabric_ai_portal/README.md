# AI-Based Industrial Fabric Defect Detection and Quality Monitoring System

**Project:** Fabric AI Portal\
**Stack:** Python, PyTorch, PyQt5, OPC UA (`asyncua`)\
**Dataset:** TILDA 400 × 64 × 64 patches\
**Integrated model:** TILDA V1, `outputs/tilda/best.pt`\
**PLC:** Local simulated OPC UA PLC

## 1. Project overview

This project is a software prototype for AI-assisted fabric quality
inspection. A PyTorch image classifier predicts whether a fabric image
is **good** or **defective**. The trained model is integrated into a
PyQt5 SCADA application. The SCADA worker communicates with a simulated
PLC using OPC UA, sends classification results and displays production
information.

The current demonstration uses images selected from the prepared TILDA
test set instead of a live industrial camera. It uses a simulated OPC UA
PLC; a physical Siemens S7-1500 has not been connected or tested.

### Objectives

- Prepare a real fabric image dataset for binary classification.
- Train and evaluate a deep-learning classifier.
- Integrate the trained model into a SCADA application.
- Exchange inspection signals with a simulated PLC through OPC UA.
- Display production state, inspection results and statistics.

## 2. System workflow

The project has two stages:

**Model development:** TILDA dataset → data preparation → model training
→ evaluation → saved checkpoint.

**Simulated production:** PLC camera trigger → SCADA worker selects a
TILDA test image → model inference → result written to PLC tags → SCADA
displays production and inspection information.

```text
TILDA images
    |
    v
Data preparation and group-based split
    |
    v
PyTorch FabricNet training
    |
    v
Evaluation and saved checkpoint
    |
    v
TILDA V1 inference
    |
    v
SCADA worker <---- OPC UA ----> Simulated PLC
    |
    v
SCADA dashboard, process windows and statistics
```

## 3. Project structure

```text
fabric_ai_portal/
|-- ai/
|   |-- data_prep.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- tilda_data_prep.py
|   |-- train_tilda.py
|   |-- evaluate_tilda.py
|   |-- evaluate_tilda_v2.py
|   |-- tilda_category_analysis.py
|   |-- tilda_error_analysis.py
|   |-- tilda_group_analysis.py
|-- data/
|   |-- raw/tilda/                 # Original five TILDA categories
|   |-- tilda_binary/              # Prepared train/val/test folders
|   |-- tilda_manifest.csv
|-- outputs/
|   |-- tilda/                     # V1 checkpoint, metrics and plots
|   |-- tilda_v2/                  # V2 experiment results
|   |-- tilda_data_stats.json
|-- plc/
|   |-- __init__.py
|   |-- tags.py
|   |-- opcua_server.py
|   |-- client.py
|-- scada/
|   |-- __init__.py
|   |-- main_window.py
|   |-- plc_window.py
|   |-- process_windows.py
|   |-- stats_window.py
|   |-- training_window.py
|   |-- widgets.py
|   |-- worker.py
|-- docs/
|-- config.py
|-- prepare_dataset.py
|-- make_synthetic_dataset.py
|-- prepare_tilda.py
|-- run_scada.py
|-- requirements.txt
|-- README.md
```

`.venv/` and `__pycache__/` are generated environment/cache folders, not
source modules. Check the course submission rules before including large
datasets and model checkpoints in an archive.

## 4. Dataset

The project uses the TILDA 400 × 64 × 64 patches dataset. Its original
categories are:

- `good`
- `hole`
- `objects`
- `oil spot`
- `thread error`

For the current model, the task is binary classification: `good` remains
good, and the four defect categories are combined into `defect`.

Original category Images

---

Good 23,170
Hole 337
Objects 837
Oil spot 636
Thread error 620
**Total** **25,600**

Binary class Images

---

Good 23,170
Defect 2,430
**Total** **25,600**

### Group-based split

The preparation script groups patches by their source-image identifier
(filename prefix before `_patch`) and assigns whole groups to one split.
This helps prevent patches from the same source image appearing in
multiple splits.

Split Total Good Defect

---

Train 17,920 16,214 1,706
Validation 3,840 3,479 361
Test 3,840 3,477 363
**Total** **25,600** **23,170** **2,430**

The split is approximately 70%/15%/15%. `prepare_tilda.py` writes
`data/tilda_manifest.csv`, recording the original category, binary
label, group identifier and split.

## 5. Data preparation

The TILDA data pipeline is implemented in `ai/tilda_data_prep.py`. It:

- Loads the prepared train, validation and test folders.
- Resizes images to 128 × 128 pixels.
- Converts images to tensors.
- Calculates channel mean and standard deviation using training
  images.
- Normalises images using those statistics.
- Applies training augmentation (horizontal/vertical flips, rotation
  and colour jitter).
- Uses `WeightedRandomSampler` to help address class imbalance.
- Applies the same training-derived normalisation statistics during
  validation, testing and inference.

The statistics are saved in `outputs/tilda_data_stats.json`. Validation
and test images are not augmented.

## 6. Model and training

The model is implemented in `ai/model.py` as `FabricNet`. It uses
convolutional feature extraction and residual blocks followed by a
classification head. It produces two class scores:

- Class 0: `good`
- Class 1: `defect`

PyTorch is used for model training and inference.

Two TILDA experiments were retained:

- **V1:** baseline training configuration.
- **V2:** experiment with additional training augmentation, including
  small translations.

V1 was selected for SCADA integration because it achieved better test
ROC-AUC and higher defect recall at its validation-selected threshold.
V2 is retained as a comparison experiment.

The V1 checkpoint is `outputs/tilda/best.pt`. Its normalisation values
are stored separately in `outputs/tilda_data_stats.json`; the inference
module loads both.

## 7. Model evaluation

The V1 threshold was selected on the validation set using the Youden J
criterion and then applied to the test set.

### TILDA V1 test results

Metric Result

---

Accuracy 88.49%
Precision 44.44%
Recall 87.05%
F1-score 58.85%
ROC-AUC 0.9472
Selected threshold 0.3198

### Confusion matrix

                    Predicted good   Predicted defect

---

Actual good 3,082 (TN) 395 (FP)
Actual defect 47 (FN) 316 (TP)

Accuracy is the proportion of correct predictions. Precision measures
how many predicted defects are actual defects. Recall measures how many
actual defects are detected. F1-score is the harmonic mean of precision
and recall. ROC-AUC measures ranking performance across classification
thresholds.

The results show good defect recall, but relatively low precision and a
substantial number of false alarms. This is an important limitation for
production use.

### V1 versus V2

Test metric V1 V2

---

ROC-AUC 0.9472 0.8895
Recall at selected threshold 87.05% 68.04%
Precision at selected threshold 44.44% 43.87%
F1-score 58.85% 53.35%
Accuracy at selected threshold 88.49% 88.75%

V1 is the model used by the current SCADA integration.

### Analysis by original defect category

The deployed model is binary, but `ai/tilda_category_analysis.py`
analyses its binary predictions grouped by the original defect category.

Category Detected / total Recall

---

Hole 45 / 46 97.83%
Objects 114 / 140 81.43%
Oil spot 79 / 91 86.81%
Thread error 78 / 86 90.70%

This is category-wise analysis, not multi-class defect prediction.

## 8. AI inference and SCADA integration

`ai/inference.py` loads the V1 checkpoint and the saved normalisation
statistics, prepares the image and runs the model in evaluation mode.
Its prediction includes the label, defect/good probabilities, decision
threshold and inference latency.

`scada/worker.py` loads the classifier and communicates with the PLC
through `plc/client.py`. When a new camera trigger is detected, the
worker selects an image from the prepared TILDA test set, runs inference
and writes the result to PLC tags.

**Camera input is simulated:** the current worker selects a dataset
image; it does not capture a live camera frame.

## 9. PLC simulation and OPC UA

The simulated PLC uses the Python `asyncua` library. Its local endpoint
is:

```text
opc.tcp://127.0.0.1:4840/fabric/plc
```

The server is implemented in `plc/opcua_server.py`, the client in
`plc/client.py`, and tag definitions in `plc/tags.py`.

Tag Direction Purpose

---

`Start_Cmd` SCADA → PLC Start production
`Stop_Cmd` SCADA → PLC Stop production
`EStop_Cmd` SCADA → PLC Emergency-stop command
`Reset_Cmd` SCADA → PLC Reset fault/E-stop
`Belt_Speed` SCADA → PLC Conveyor speed setting
`Camera_Trigger` PLC → SCADA/AI Request inspection
`AI_Done` SCADA/AI → PLC Classification completed
`AI_Result` SCADA/AI → PLC 0 = good, 1 = defect
`AI_Confidence` SCADA/AI → PLC Confidence of predicted class
`State` PLC → SCADA Production state
`Motor_Run` PLC → SCADA Motor status
`Reject_Pusher` PLC → SCADA Rejection actuator status
`Lamp_Green` PLC → SCADA Normal-operation indicator
`Lamp_Red` PLC → SCADA Fault/E-stop indicator
`Item_Present` PLC → SCADA Item-presence signal

The project has been tested with the simulated OPC UA server. A physical
Siemens S7-1500 has not been connected or tested.

## 10. SCADA application

The user interface is developed using PyQt5.

---

File Responsibility

---

`scada/main_window.py` Main dashboard, production
controls, connection status and
navigation

`scada/process_windows.py` AI inspection and conveyor process
windows

`scada/stats_window.py` Production counters, charts,
history and CSV export

`scada/plc_window.py` OPC UA tag monitoring and manual
tag writing

`scada/training_window.py` AI training/evaluation information

`scada/widgets.py` Reusable interface widgets

`scada/worker.py` Background PLC communication and AI
inspection worker

---

The worker runs in a background thread so PLC communication and model
inference do not block the main GUI thread.

## 11. Installation

The development environment used Python 3.11 on Windows.

From the project root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python
executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 12. Dataset preparation and AI commands

The original TILDA categories should be placed in:

```text
data/raw/tilda/
    good/
    hole/
    objects/
    oil spot/
    thread error/
```

Prepare the binary group-based splits:

```powershell
python prepare_tilda.py
```

This creates `data/tilda_binary/` and `data/tilda_manifest.csv`. Do not
rerun it unnecessarily if you want to preserve the existing split.

Train TILDA V1:

```powershell
python -m ai.train_tilda
```

Evaluate V1:

```powershell
python -m ai.evaluate_tilda
```

Evaluate V2:

```powershell
python -m ai.evaluate_tilda_v2
```

Run inference on an existing image (replace the example path if that
filename is not present):

```powershell
python -m ai.inference "data/tilda_binary/test/good/good_003_patch0-2.png"
```

The existing V1 checkpoint is already available, so retraining is not
required to run SCADA.

## 13. Starting the SCADA demonstration

Use two terminals from the project root, with the virtual environment
activated in each.

**Terminal 1 --- simulated PLC:**

```powershell
python -m plc.opcua_server
```

Keep this terminal running.

**Terminal 2 --- SCADA:**

```powershell
python run_scada.py
```

In the application:

1.  Connect to `opc.tcp://127.0.0.1:4840/fabric/plc`.
2.  Confirm that the PLC connection is established.
3.  Start the simulated production line.
4.  Open the PLC simulation window to monitor tags.
5.  Trigger an inspection and observe `AI_Done`, `AI_Result` and
    `AI_Confidence`.
6.  Open Statistics to inspect the recorded production information.

The local demonstration server does not have certificates or an
encrypted endpoint configured. This setup is intended for local testing,
not an untrusted production network.

## 14. Testing performed

The following have been exercised in the development environment:

- TILDA preparation and group-based splitting.
- V1 and V2 training experiments.
- Model evaluation and comparison.
- Single-image V1 inference.
- Loading TILDA V1 from the SCADA worker.
- Starting the simulated OPC UA PLC and connecting SCADA.
- Monitoring and writing PLC tags.
- Triggering simulated AI inspections and observing classification
  results.
- Opening the SCADA process and statistics windows.

These are software-simulation tests, not tests on a physical production
line.

## 15. Limitations and future work

### Limitations

1.  **No live camera:** inspection uses a randomly selected TILDA test
    image.
2.  **Simulated PLC:** the local OPC UA server was tested; physical
    S7-1500 communication was not.
3.  **Binary output:** the model predicts good or defect, not the
    specific defect subtype.
4.  **False alarms:** V1 precision is relatively low at the selected
    threshold.
5.  **Production counters:** counter consistency should be checked
    before using statistics for real production reporting.
6.  **OPC UA security:** the local simulation does not have certificates
    or encryption configured.

### Future improvements

- Integrate a live camera using OpenCV.
- Test with a physical Siemens S7-1500.
- Improve precision while maintaining an acceptable defect recall.
- Collect more representative defective samples.
- Extend the model to classify individual defect types.
- Validate production counters and actuator timing over longer runs.
- Configure OPC UA authentication, certificates and encryption.

## 16. Conclusion

This project demonstrates an end-to-end software prototype for AI-based
fabric quality inspection. A PyTorch model trained on TILDA images
classifies fabric patches as good or defective. The model is integrated
with a PyQt5 SCADA application, which exchanges inspection information
with a simulated PLC through OPC UA.

The prototype demonstrates how image-based machine learning can be
connected to industrial monitoring concepts. It provides a foundation
for future work with live camera acquisition, physical PLC communication
and further improvements to classification performance.
