from __future__ import annotations

import argparse

import torch

from dian_spring_test.models import GatedDeltaNetClassifier
from dian_spring_test.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_default_device()

    model = GatedDeltaNetClassifier(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=0.0,
    ).to(device)
    model.eval()

    images = torch.randn(args.batch_size, 1, 28, 28, device=device)
    with torch.no_grad():
        parallel_logits = model(images)
        recurrent_logits = model.forward_recurrent(images)

    max_abs_diff = (parallel_logits - recurrent_logits).abs().max().item()
    print(f"parallel_shape={tuple(parallel_logits.shape)}")
    print(f"recurrent_shape={tuple(recurrent_logits.shape)}")
    print(f"max_abs_diff={max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
