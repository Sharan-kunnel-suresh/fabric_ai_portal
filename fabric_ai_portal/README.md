# Fabric AI Portal

Python production project: **PyTorch fabric-defect classifier + PyQt5 SCADA + OPC UA PLC (simulated or real S7-1500)**.

```
fabric_ai_portal/
  config.py                  paths, image size, OPC UA endpoint
  prepare_dataset.py         raw Kaggle download  ->  data/dataset/{good,defect}
  make_synthetic_dataset.py  1500 fake fabric images for testing (no download needed)
  ai/  data_prep.py  model.py  train.py  evaluate.py  inference.py
  plc/ tags.py  opcua_server.py (simulated PLC)  client.py
  scada/ main_window.py  process_windows.py  stats_window.py  plc_window.py  training_window.py  worker.py  widgets.py
  run_scada.py               start the SCADA (--with-sim also starts the simulated PLC)
  docs/PROJECT_DESCRIPTION.md   part (a)-(f) of the task, 20 p
  docs/S7_1500_SCL.md           real PLC instructions + SCL program
```

## Run it
The zip already contains a synthetic demo dataset and a model trained on it, so `python run_scada.py --with-sim` works immediately.
**To use real data, first delete `data/dataset/*` and `outputs/*`**, then follow the steps below.

```bash
pip install -r requirements.txt

# 1) DATA - real data (Kaggle: WFDD or TILDA): unzip into data/raw/ then
python prepare_dataset.py --src data/raw/<folder>
#    or, to test without downloading anything:
python make_synthetic_dataset.py --n 1500

# 2) AI
python -m ai.data_prep          # prints split sizes / class balance / mean+std
python -m ai.train --epochs 15  # LR finder -> OneCycle training -> outputs/best.pt
python -m ai.evaluate           # metrics, plots, threshold -> outputs/
python -m ai.inference some_image.png

# 3) SCADA + simulated PLC (one command)
python run_scada.py --with-sim
#    or two terminals:  python -m plc.opcua_server   /   python run_scada.py
```
In the SCADA press **START**; the line runs, the AI classifies every item, defects are pushed out.
`python -m ai.train --resume --epochs 25` continues from `outputs/last.pt`.

## Where each grading item lives
| Item (points) | Code | What it does |
|---|---|---|
| Project description (20) | `docs/PROJECT_DESCRIPTION.md` | inputs, outputs, motion, task, construction |
| Dataset (15) | `prepare_dataset.py` | WFDD/TILDA -> good/defect, 1000+ images check |
| Data preparation (35) | `ai/data_prep.py` | resize, standardization (train mean/std), normalize, stratified index splitter, augmentation (flip/rotate/colour jitter), imbalance handling (WeightedRandomSampler or class weights), train/val/test sets, tensor transforms |
| Model configuration (40) | `ai/model.py`, `ai/train.py` | featurizer (conv stem + 3 residual blocks), classifier (dropout + FC), batch norm, cross-entropy, AdamW + weight decay, LR finder, OneCycle scheduler, batch size |
| Model training (30) | `ai/train.py` | training loop, checkpoints (`model/optimizer/scheduler state_dict`, epoch, losses), best + last, resume, early stopping |
| Model evaluation (35) | `ai/evaluate.py` | val/test loss, loss curves, confusion matrix (TN/FP/FN/TP), TPR/FPR, precision/recall/accuracy/F1, threshold sweep + choice, ROC + AUC, filters, feature maps |
| Model integration (35) | `ai/inference.py`, `scada/worker.py` | `Classifier.predict()` is called when the PLC triggers the camera; result goes back to the PLC |
| Main window (15) | `scada/main_window.py` | control buttons, state, lamps, KPIs, connection, log, navigation, alarm message boxes |
| Statistics window (20) | `scada/stats_window.py` | counters, defect-rate trend, probability histogram, AI accuracy, table, CSV export |
| Process windows (35) | `scada/process_windows.py` | (1) AI inspection view, (2) animated conveyor (QPainter) |
| PLC simulation window (20) | `scada/plc_window.py` | live OPC UA tag table + write/force + speed slider |
| Additional window (20) | `scada/training_window.py` | training/evaluation plots and metrics |
| OPC UA simulated | `plc/opcua_server.py`, `plc/client.py` | asyncua server with a 50 ms PLC scan cycle |
| OPC UA real S7-1500 (50) | `docs/S7_1500_SCL.md` | TIA setup + SCL; SCADA option *Real S7-1500* |

## What was tested, and what was not
* Tested here (CPU only): dataset prep, training, evaluation, inference, the OPC UA PLC, and the full SCADA loop
  (start, AI classification, reject, E-stop) - all windows were run headless and checked via screenshots.
* All numbers in `outputs/` come from the **synthetic** dataset, where defects are easy (accuracy ~100%).
  On real WFDD/TILDA images expect lower scores - that is normal; retrain and re-run `ai.evaluate`,
  then describe your own results in the presentation.
* The "camera" is simulated: a random test-set image is shown for each item. Replace `_handle_trigger` in
  `scada/worker.py` with an OpenCV grab (`cv2.VideoCapture`) for a real camera.
* The S7-1500 part is documented but could not be tried on hardware.

## Tuning tips for real data
* WFDD images are large: keep `IMG_SIZE=128` first, try 224 if defects are tiny (slower).
* Few defects? keep `--imbalance sampler`, or try `--imbalance weights`.
* Overfitting (val loss rises): raise `DROPOUT`/`WEIGHT_DECAY` in `config.py`, add augmentation.
