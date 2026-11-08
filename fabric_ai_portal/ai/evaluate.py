"""MODEL EVALUATION (max 35 p)
Validation/test loss, loss curves, classification threshold, confusion matrix (TN/FP/FN/TP),
TPR/FPR, recall, precision, accuracy, ROC curve + AUC, filters and feature maps.

    python -m ai.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import config
from ai.data_prep import build_dataloaders, eval_tf, load_splits
from ai.model import build_model

OUT = config.OUT_DIR


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    probs, ys, loss = [], [], 0.0
    for x, y in loader:
        out = model(x.to(device))
        loss += crit(out, y.to(device)).item()
        probs.append(torch.softmax(out, 1)[:, 1].cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), loss / len(loader.dataset)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(thr), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "TPR(recall)": tp / max(tp + fn, 1), "FPR": fp / max(fp + tn, 1),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "F1": float(f1_score(y, pred, zero_division=0))}


def plot_losses(h):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["train_loss"], label="train"); ax[0].plot(h["val_loss"], label="validation")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    ax[1].plot(h["train_acc"], label="train"); ax[1].plot(h["val_acc"], label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "loss_curves.png", dpi=120); plt.close(fig)


def plot_confusion(m, name):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=config.CLASSES, yticklabels=config.CLASSES,
           xlabel="predicted", ylabel="actual", title=f"Confusion matrix ({name}, thr={m['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=120); plt.close(fig)


def plot_roc(y, p, thr):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    m = metrics_at(y, p, thr)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.scatter([m["FPR"]], [m["TPR(recall)"]], c="r", zorder=3, label=f"threshold {thr:.2f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve (test)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=120); plt.close()
    return float(auc)


def plot_threshold_sweep(y, p):
    ts = np.linspace(0.01, 0.99, 99)
    pr = [precision_score(y, p >= t, zero_division=0) for t in ts]
    rc = [recall_score(y, p >= t, zero_division=0) for t in ts]
    plt.figure(figsize=(6, 4)); plt.plot(ts, pr, label="precision"); plt.plot(ts, rc, label="recall")
    plt.xlabel("classification threshold"); plt.legend(); plt.title("Threshold sweep (validation)"); plt.tight_layout()
    plt.savefig(OUT / "threshold_sweep.png", dpi=120); plt.close()


def plot_filters(model):
    w = model.featurizer.stem[0].weight.detach().cpu()
    w = (w - w.min()) / (w.max() - w.min())
    fig, axs = plt.subplots(2, 8, figsize=(10, 3))
    for a, f in zip(axs.flat, w):
        a.imshow(f.permute(1, 2, 0)); a.axis("off")
    fig.suptitle("First-layer filters"); fig.tight_layout(); fig.savefig(OUT / "filters.png", dpi=120); plt.close(fig)


@torch.no_grad()
def plot_feature_maps(model, device, info):
    splits = load_splits()
    path = next((q for q in splits["test"] if "defect" in q.replace("\\", "/").split("/")[-2]), splits["test"][0])
    img = Image.open(path).convert("RGB")
    x = eval_tf(info["mean"], info["std"])(img).unsqueeze(0).to(device)
    acts = {}
    hooks = [getattr(model.featurizer, n).register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o.cpu()))
             for n in ("stem", "layer1", "layer3")]
    model.eval(); model(x)
    for h in hooks:
        h.remove()
    fig, axs = plt.subplots(3, 9, figsize=(13, 5))
    for r, name in enumerate(acts):
        if r == 0:
            axs[r, 0].imshow(img.resize((64, 64)))
        for c in range(8):
            axs[r, c + 1].imshow(acts[name][0, c], cmap="viridis")
        for a in axs[r]:
            a.set_xticks([]); a.set_yticks([])
        axs[r, 0].set_xlabel(name, fontsize=11, fontweight="bold")
        if r > 0:
            for sp in axs[r, 0].spines.values():
                sp.set_visible(False)
    fig.suptitle(f"Feature maps ({path.replace(chr(92), '/').split('/')[-1]})"); fig.tight_layout()
    fig.savefig(OUT / "feature_maps.png", dpi=120); plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, info = build_dataloaders()
    ck = torch.load(OUT / "best.pt", map_location=device)
    model = build_model(device); model.load_state_dict(ck["model_state_dict"])

    pv, yv, lv = collect(model, loaders["val"], device)
    pt, yt, lt = collect(model, loaders["test"], device)
    # choose the classification threshold on the VALIDATION set (Youden J = TPR - FPR), apply to TEST
    fpr, tpr, thr = roc_curve(yv, pv)
    best_thr = float(np.clip(thr[np.argmax(tpr - fpr)], 0.05, 0.95))
    (OUT / "threshold.json").write_text(json.dumps({"threshold": best_thr}))

    mt = metrics_at(yt, pt, best_thr)
    result = {"val_loss": lv, "test_loss": lt, "auc_test": plot_roc(yt, pt, best_thr),
              "auc_val": float(roc_auc_score(yv, pv)), "test_at_chosen_threshold": mt,
              "test_at_0.5": metrics_at(yt, pt, 0.5), "best_epoch": ck["epoch"] + 1}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    plot_losses(json.loads((OUT / "history.json").read_text()))
    plot_confusion(mt, "test"); plot_threshold_sweep(yv, pv); plot_filters(model); plot_feature_maps(model, device, info)
    print(json.dumps(result, indent=2))
    print("plots saved in", OUT)


if __name__ == "__main__":
    main()
