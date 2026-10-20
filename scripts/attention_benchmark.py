"""Causal multi-head attention: correctness checks and how cost scales with context.

1. Equivalence: our MultiHeadAttention (explicit softmax(QKᵀ/√d)V with a mask)
   matches PyTorch's fused scaled_dot_product_attention to float precision.
2. Causality: changing future tokens never changes earlier outputs (no look-ahead
   leak -- the same discipline a backtest needs).
3. Cost: time and peak memory of forward+backward vs sequence length T, explicit
   vs fused. The explicit version stores a (b, h, T, T) matrix, so memory grows
   with T²; the fused kernel avoids that.

Output: results/attention_benchmark.json + figure.
"""

import torch

from _common import base_parser, plt, rel, savefig, SERIES
from finllm.attention import MultiHeadAttention
from finllm.paths import results_dirs
from finllm.utils import Timer, device_name, get_device, run_metadata, save_json, set_seed


def make_pair(d, heads, ctx, device):
    torch.manual_seed(0)
    a = MultiHeadAttention(d, d, ctx, 0.0, heads, qkv_bias=True, use_sdpa=False).to(device)
    b = MultiHeadAttention(d, d, ctx, 0.0, heads, qkv_bias=True, use_sdpa=True).to(device)
    b.load_state_dict(a.state_dict())
    return a.eval(), b.eval()


def bench(mha, x, reps):
    dev = x.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    mha.train()
    for _ in range(2):  # warm-up
        mha(x).sum().backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with Timer() as t:
        for _ in range(reps):
            mha(x).sum().backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if dev.type == "cuda" else None
    return t.seconds / reps * 1000, peak


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)
    d, heads, batch = (64, 4, 2) if args.smoke else (768, 12, 8)
    lengths = [32, 64, 128] if args.smoke else [128, 256, 512, 1024]
    reps = 2 if args.smoke else 10

    # 1. equivalence
    explicit, fused = make_pair(d, heads, max(lengths), device)
    x = torch.randn(batch, max(lengths), d, device=device)
    with torch.no_grad():
        max_diff = (explicit(x) - fused(x)).abs().max().item()
    print(f"max |explicit - fused| = {max_diff:.2e}")

    # 2. causality
    T = max(lengths)
    x2 = x.clone()
    x2[:, T // 2:, :] = torch.randn_like(x2[:, T // 2:, :])
    with torch.no_grad():
        leak = (explicit(x)[:, :T // 2] - explicit(x2)[:, :T // 2]).abs().max().item()
        changed = (explicit(x)[:, T // 2:] - explicit(x2)[:, T // 2:]).abs().max().item()
    print(f"change in past outputs after editing the future: {leak:.2e} (future outputs moved {changed:.2f})")

    # 3. scaling
    rows = []
    for T in lengths:
        explicit, fused = make_pair(d, heads, T, device)
        xb = torch.randn(batch, T, d, device=device)
        row = {"T": T}
        for name, m in [("explicit", explicit), ("fused_sdpa", fused)]:
            try:
                ms, mb = bench(m, xb, reps)
                row[f"{name}_ms"], row[f"{name}_peak_mb"] = ms, mb
            except torch.OutOfMemoryError:
                row[f"{name}_ms"], row[f"{name}_peak_mb"] = None, "OOM"
                torch.cuda.empty_cache()
        rows.append(row)
        print(row)

    fig, axes = plt.subplots(1, 2 if device.type == "cuda" else 1, figsize=(9 if device.type == "cuda" else 5, 3.2))
    axes = axes if hasattr(axes, "__len__") else [axes]
    Ts = [r["T"] for r in rows]
    for i, (name, label) in enumerate([("explicit", "explicit (textbook)"), ("fused_sdpa", "fused SDPA")]):
        axes[0].plot(Ts, [r[f"{name}_ms"] for r in rows], marker="o", ms=5, color=SERIES[i], label=label)
        if device.type == "cuda":
            axes[1].plot(Ts, [r[f"{name}_peak_mb"] if isinstance(r[f"{name}_peak_mb"], float) else None for r in rows],
                         marker="o", ms=5, color=SERIES[i], label=label)
    axes[0].set(xlabel="sequence length T", ylabel="ms per fwd+bwd", title="Attention time vs context length")
    axes[0].legend()
    if device.type == "cuda":
        axes[1].set(xlabel="sequence length T", ylabel="peak MB", title="Attention memory vs context length")
    savefig(fig, figs, "attention_scaling.png")

    result = {"experiment": "attention_benchmark", "smoke": args.smoke, "device": device_name(device),
              "config": {"d_model": d, "heads": heads, "batch": batch},
              "max_abs_diff_explicit_vs_fused": max_diff,
              "causal_leak_max_abs_change": leak, "future_outputs_changed_by": changed,
              "scaling": rows, "figures": [rel(figs / "attention_scaling.png")],
              "meta": run_metadata(device, seed=args.seed)}
    print("saved", rel(save_json(result, out_dir / "attention_benchmark.json")))


if __name__ == "__main__":
    main()
