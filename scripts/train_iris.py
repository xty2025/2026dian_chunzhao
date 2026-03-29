from __future__ import annotations

from pathlib import Path

import torch

from dian_spring_test.data import make_iris_loaders
from dian_spring_test.models import IrisMLP, custom_cross_entropy
from dian_spring_test.utils import ensure_dir, get_default_device, save_json, set_seed


def evaluate(model: IrisMLP, data_loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits, _ = model(features)
            preds = logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def main() -> None:
    set_seed(42)
    device = get_default_device()
    train_loader, test_loader = make_iris_loaders(batch_size=16, test_ratio=0.2, seed=42)

    model = IrisMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    history = []
    epochs = 200
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(features)
            loss = custom_cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_accuracy": test_acc})
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} test_acc={test_acc:.4f}")

    output_dir = ensure_dir(Path(__file__).resolve().parents[1] / "results")
    save_json(
        {
            "final_test_accuracy": history[-1]["test_accuracy"],
            "history": history,
        },
        output_dir / "iris_metrics.json",
    )
    print(f"Final test accuracy: {history[-1]['test_accuracy']:.4f}")
    print(f"Saved metrics to {output_dir / 'iris_metrics.json'}")


if __name__ == "__main__":
    main()
