from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from dian_spring_test.data import make_fashion_mnist_loaders
from dian_spring_test.models import VisionTransformerClassifier
from dian_spring_test.utils import ensure_dir, get_default_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    return parser.parse_args()


def evaluate(model: nn.Module, data_loader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / total, correct / total


def plot_history(history: dict, output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vit_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_accuracy"], label="val_accuracy")
    plt.plot(epochs, history["test_accuracy"], label="test_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "vit_accuracy_curve.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_default_device()

    project_root = Path(__file__).resolve().parents[1]
    data_root = ensure_dir(project_root / args.data_root)
    output_dir = ensure_dir(project_root / args.output_dir)

    train_loader, val_loader, test_loader = make_fashion_mnist_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = VisionTransformerClassifier(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
    }

    best_val_acc = -1.0
    best_model_path = output_dir / "vit_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = total_loss / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["test_accuracy"].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    plot_history(history, output_dir)
    save_json(
        {
            "config": vars(args),
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": history["test_accuracy"][-1],
            "history": history,
        },
        output_dir / "vit_metrics.json",
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"best_checkpoint_test_loss={test_loss:.4f} best_checkpoint_test_acc={test_acc:.4f}")
    print(f"saved_model={best_model_path}")
    print(f"saved_plots={[str(output_dir / 'vit_loss_curve.png'), str(output_dir / 'vit_accuracy_curve.png')]}")


if __name__ == "__main__":
    main()
