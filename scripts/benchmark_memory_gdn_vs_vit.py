from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dian_spring_test.models import GatedDeltaNetClassifier, VisionTransformerClassifier
from dian_spring_test.utils import ensure_dir, get_default_device, save_json, set_seed


MODEL_LABELS = ["gdn_parallel", "gdn_recurrent", "vit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[28, 56, 84, 112, 140, 168, 196, 224])
    parser.add_argument("--token-dim", type=int, default=28)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def peak_vram_mb(forward_fn, tokens: torch.Tensor) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM benchmark")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = forward_fn(tokens)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def plot_curves(lengths: list[int], curves: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name in MODEL_LABELS:
        plt.plot(lengths, curves[name], marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak VRAM (MB)")
    plt.title("Inference Peak VRAM vs Sequence Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "memory_benchmark_gdn_vs_vit.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_default_device()
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA.")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(project_root / args.output_dir)
    max_seq_len = max(args.lengths)

    gdn = GatedDeltaNetClassifier(
        token_dim=args.token_dim,
        max_seq_len=max_seq_len,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    vit = VisionTransformerClassifier(
        token_dim=args.token_dim,
        max_seq_len=max_seq_len,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    gdn.eval()
    vit.eval()

    results = {
        "config": vars(args),
        "parameter_count": {
            "gdn": count_parameters(gdn),
            "vit": count_parameters(vit),
        },
        "peak_vram_mb": {name: [] for name in MODEL_LABELS},
    }

    print(f"parameter_count gdn={results['parameter_count']['gdn']} vit={results['parameter_count']['vit']}")
    for length in args.lengths:
        tokens = torch.randn(1, length, args.token_dim, device=device)
        gdn_parallel_mb = peak_vram_mb(gdn.forward_tokens, tokens)
        gdn_recurrent_mb = peak_vram_mb(gdn.forward_tokens_recurrent, tokens)
        vit_mb = peak_vram_mb(vit.forward_tokens, tokens)

        results["peak_vram_mb"]["gdn_parallel"].append(gdn_parallel_mb)
        results["peak_vram_mb"]["gdn_recurrent"].append(gdn_recurrent_mb)
        results["peak_vram_mb"]["vit"].append(vit_mb)
        print(
            f"length={length:03d} gdn_parallel={gdn_parallel_mb:.2f}MB "
            f"gdn_recurrent={gdn_recurrent_mb:.2f}MB vit={vit_mb:.2f}MB"
        )

    save_json(results, output_dir / "memory_benchmark_gdn_vs_vit.json")
    plot_curves(args.lengths, results["peak_vram_mb"], output_dir)
    print(f"saved_json={output_dir / 'memory_benchmark_gdn_vs_vit.json'}")
    print(f"saved_plot={output_dir / 'memory_benchmark_gdn_vs_vit.png'}")


if __name__ == "__main__":
    main()
