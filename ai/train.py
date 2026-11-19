"""MODEL TRAINING (max 30 p) + learning-rate finder, scheduler, checkpointing.

    python -m ai.train --epochs 15
    python -m ai.train --epochs 25 --resume        # continue from outputs/last.pt
"""
import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config
from ai.data_prep import build_dataloaders
from ai.model import build_model


def find_lr(model, loader, criterion, device, start=1e-6, end=1e-1, iters=100):
    """LR range test: grow the LR exponentially, record the loss, restore the weights afterwards."""
    state = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=start)
    gamma = (end / start) ** (1 / iters)
    lrs, losses, best, lr = [], [], float("inf"), start
    model.train()
    it = iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward(); opt.step()
        lrs.append(lr); losses.append(loss.item())
        best = min(best, loss.item())
        if loss.item() > 4 * best:
            break
        lr *= gamma
    model.load_state_dict(state)
    k = 5
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    steepest = int(np.argmin(np.gradient(smooth)))           # point where the loss falls fastest
    best_lr = lrs[steepest + k // 2]
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, losses); plt.xscale("log"); plt.axvline(best_lr, color="r", ls="--", label=f"chosen {best_lr:.1e}")
    plt.xlabel("learning rate"); plt.ylabel("loss"); plt.title("LR finder"); plt.legend(); plt.tight_layout()
    plt.savefig(config.OUT_DIR / "lr_finder.png", dpi=120); plt.close()
    return float(np.clip(best_lr, 3e-4, 3e-3))


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    tot, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if scheduler is not None:
                    scheduler.step()                       # OneCycleLR steps every batch
            tot += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()
    return tot / n, correct / n


def save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_val):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "history": history, "best_val_loss": best_val,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=None, help="skip the LR finder and use this max LR")
    ap.add_argument("--imbalance", choices=["sampler", "weights", "none"], default="sampler")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, info = build_dataloaders(a.batch, a.imbalance)
    print(f"device={device}  train/val/test = {info['n_train']}/{info['n_val']}/{info['n_test']}  "
          f"class counts={info['train_class_counts']}")
    model = build_model(device)
    weight = torch.tensor(info["class_weights"], dtype=torch.float32, device=device) if a.imbalance == "weights" else None
    criterion = nn.CrossEntropyLoss(weight=weight)          # loss function

    max_lr = a.lr or find_lr(model, loaders["train"], criterion, device)
    print(f"max learning rate: {max_lr:.2e}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=config.WEIGHT_DECAY)  # optimizer + L2 reg.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=a.epochs, steps_per_epoch=len(loaders["train"]))            # LR scheduler

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    start_epoch, best_val, bad = 0, float("inf"), 0
    last = config.OUT_DIR / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model_state_dict"]); optimizer.load_state_dict(ck["optimizer_state_dict"])
        history, best_val, start_epoch = ck["history"], ck["best_val_loss"], ck["epoch"] + 1
        try:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        except Exception:
            print("scheduler state not compatible with the new --epochs; scheduler restarted")
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders["train"], criterion, device, optimizer, scheduler)
        vl, va = run_epoch(model, loaders["val"], criterion, device)
        for k, v in zip(history, (tl, vl, ta, va, optimizer.param_groups[0]["lr"])):
            history[k].append(v)
        print(f"epoch {epoch + 1:02d}/{a.epochs}  train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{va:.3f}  "
              f"lr {history['lr'][-1]:.2e}  {time.time() - t0:.1f}s")
        save_checkpoint(last, model, optimizer, scheduler, epoch, history, best_val)
        if vl < best_val:
            best_val, bad = vl, 0
            save_checkpoint(config.OUT_DIR / "best.pt", model, optimizer, scheduler, epoch, history, best_val)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
    (config.OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val loss {best_val:.4f}  -> {config.OUT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
