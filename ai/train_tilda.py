
"""
Train the fabric defect CNN using the real TILDA dataset.

Usage:
    python -m ai.train_tilda --epochs 10 --lr 0.001
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import config
from ai.model import build_model
from ai.tilda_data_prep import build_tilda_dataloaders


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(path, model, optimizer, scheduler, epoch, history, best_val, data_info):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_val_loss": best_val,
        "classes": ["good", "defect"],
        "dataset": "TILDA",
        "img_size": config.IMG_SIZE,
        "normalization_mean": data_info["mean"],
        "normalization_std": data_info["std"],
        "training_class_counts": data_info["train_class_counts"],
        "imbalance_strategy": data_info["imbalance_strategy"],
        "experiment": "TILDA V2",
        "learning_rate": optimizer.param_groups[0]["lr"],
    }, path)


def save_history(history, output_dir):
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("TILDA Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Training")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("TILDA Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--imbalance",
        choices=["sampler", "weights", "none"],
        default="sampler",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = config.OUT_DIR / "tilda_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "experiment": "TILDA V2",
        "epochs": args.epochs,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "patience": args.patience,
        "imbalance": args.imbalance,
        "image_size": config.IMG_SIZE,
    "seed": config.SEED,
    "weight_decay": config.WEIGHT_DECAY,
    "optimizer": "AdamW",
    "scheduler": "OneCycleLR",
    "model": "FabricNet",
    "augmentation": {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation_degrees": 15,
        "brightness": 0.15,
        "contrast": 0.15,
        "translation": 0.05
      }
    }
    (output_dir / "experiment_config.json").write_text(
    json.dumps(experiment_config, indent=2)
    )
    
    print("=" * 55)
    print("TILDA FABRIC DEFECT MODEL TRAINING")
    print("=" * 55)
    print("Device:", device)

    loaders, info = build_tilda_dataloaders(
        batch_size=args.batch,
        imbalance=args.imbalance,
    )

    print("Training images:", info["n_train"])
    print("Validation images:", info["n_val"])
    print("Test images:", info["n_test"])
    print("Training class counts:", info["train_class_counts"])

    model = build_model(device)

    class_weights = torch.tensor(
        info["class_weights"],
        dtype=torch.float32,
        device=device,
    )

    loss_weights = (
        class_weights if args.imbalance == "weights" else None
    )

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(loaders["train"]),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            scheduler,
        )

        val_loss, val_acc = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        improved = val_loss < best_val

        if improved:
            best_val = val_loss
            bad_epochs = 0

            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                best_val,
                info
            )
        else:
            bad_epochs += 1

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            best_val,
            info
        )

        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        save_history(history, output_dir)

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    print("\nTRAINING COMPLETE")
    print("Best validation loss:", round(best_val, 5))
    print("Best checkpoint:", output_dir / "best.pt")
    print("Training history:", output_dir / "history.json")
    print("Training curves:", output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
  