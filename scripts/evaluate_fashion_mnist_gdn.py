from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from dian_spring_test.data import make_fashion_mnist_loaders
from dian_spring_test.models import GatedDeltaNetClassifier
from dian_spring_test.utils import get_default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results/gdn_best.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--disable-pos-emb", action="store_true")
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


def main() -> None:
    args = parse_args()
    device = get_default_device()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint = project_root / args.checkpoint
    data_root = project_root / args.data_root

    _, _, test_loader = make_fashion_mnist_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = GatedDeltaNetClassifier(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
        use_positional_embedding=not args.disable_pos_emb,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    loss, acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"checkpoint={checkpoint}")
    print(f"test_loss={loss:.4f}")
    print(f"test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
