"""Model size accounting and "what fits on my GPU".

1. Parameter counts for the four GPT-2 sizes, built on PyTorch's 'meta' device
   (no memory used), checked against a closed-form formula.
2. Measured peak GPU memory and throughput of one full training step
   (forward + backward + AdamW) for GPT-2 small/medium at several batch sizes,
   fp32 vs bf16-autocast, explicit vs fused attention. This decides the batch
   sizes used for every training run on a 6 GB laptop GPU.
3. Text from an untrained model (gibberish, which is why pretraining matters).

Output: results/memory_profile.json + figure.
"""

import torch

from _common import base_parser, plt, rel, savefig, SERIES
from finllm.config import GPT_CONFIGS, TINY_CONFIG, get_config
from finllm.generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from finllm.model import GPTModel, analytic_param_count, count_parameters, training_flops_per_token
from finllm.paths import results_dirs
from finllm.tokenizer import get_gpt2_tokenizer
from finllm.utils import Timer, bf16_supported, device_name, get_device, run_metadata, save_json, set_seed


def size_table():
    rows = []
    for name, cfg in GPT_CONFIGS.items():
        cfg = dict(cfg, qkv_bias=True)
        with torch.device("meta"):
            m = GPTModel(cfg)
        untied = count_parameters(m)
        tied = count_parameters(m, tied=True)
        assert tied == analytic_param_count(cfg, tied=True)
        rows.append({"model": name, "params_untied": untied, "params_tied": tied,
                     "fp32_weights_gb": tied * 4 / 1024 ** 3,
                     "adamw_train_state_gb": tied * 16 / 1024 ** 3,  # weights+grads+2 Adam moments, fp32
                     "train_gflops_per_token": training_flops_per_token(cfg) / 1e9})
    return rows


def train_step_probe(cfg, batch, seq, device, amp, reps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = GPTModel(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randint(0, cfg["vocab_size"], (batch, seq), device=device)
        for i in range(reps + 1):
            if i == 1:
                torch.cuda.synchronize()
                timer = Timer().__enter__()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), x.flatten())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        timer.__exit__()
        secs = timer.seconds / reps
        return {"peak_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "tokens_per_s": batch * seq / secs, "status": "ok"}
    except torch.OutOfMemoryError:
        return {"peak_gb": None, "tokens_per_s": None, "status": "OOM"}
    finally:
        del model, opt
        torch.cuda.empty_cache()


def main():
    args = base_parser(__doc__).parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir, figs = results_dirs(args.smoke)

    sizes = size_table()
    for r in sizes:
        print(f"{r['model']:20s} {r['params_tied']:>14,} params  fp32 {r['fp32_weights_gb']:.2f} GB  "
              f"train state {r['adamw_train_state_gb']:.1f} GB")

    probes = []
    if device.type == "cuda":
        seq = 256
        for name in ["gpt2-small (124M)", "gpt2-medium (355M)"]:
            for sdpa in (False, True):
                for amp in ((False, True) if bf16_supported(device) else (False,)):
                    for batch in (1, 2, 4, 8, 16):
                        cfg = get_config(name, context_length=1024, qkv_bias=True, use_sdpa=sdpa, drop_rate=0.0)
                        r = train_step_probe(cfg, batch, seq, device, amp)
                        r.update({"model": name, "batch": batch, "seq": seq, "sdpa": sdpa, "bf16": amp})
                        probes.append(r)
                        print(r)
                        if r["status"] == "OOM":
                            break
    else:
        print("No CUDA device: skipping GPU memory probes (run on your GPU for the real table).")

    # Untrained generation
    tok = get_gpt2_tokenizer()
    cfg = TINY_CONFIG if args.smoke else get_config("small")
    torch.manual_seed(args.seed)
    model = GPTModel(cfg).eval()
    prompt = "Apple shares rose after"
    out = generate_text_simple(model, text_to_token_ids(prompt, tok), 10, cfg["context_length"])
    untrained = token_ids_to_text(out, tok)
    print("untrained model:", repr(untrained))

    figs_out = []
    if probes:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
        for ax, name in zip(axes, ["gpt2-small (124M)", "gpt2-medium (355M)"]):
            combos = [(False, False, "fp32, explicit attn"), (True, False, "fp32, fused attn"),
                      (True, True, "bf16, fused attn")]
            for i, (sdpa, amp, label) in enumerate(combos):
                pts = [(p["batch"], p["peak_gb"]) for p in probes
                       if p["model"] == name and p["sdpa"] == sdpa and p["bf16"] == amp and p["status"] == "ok"]
                if pts:
                    ax.plot(*zip(*pts), marker="o", ms=5, color=SERIES[i], label=label)
            total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
            ax.axhline(total, color="#e34948", lw=1, ls="--")
            ax.text(1, total * 0.97, f"GPU memory {total:.1f} GB", va="top", fontsize=8, color="#52514e")
            ax.set(xscale="log", xlabel="batch size (seq 256)", title=name)
            ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        axes[0].set_ylabel("peak memory, one AdamW step (GB)")
        axes[0].legend(loc="upper left", fontsize=8)
        figs_out.append(rel(savefig(fig, figs, "gpu_memory_fit.png")))

    result = {"experiment": "memory_profile", "smoke": args.smoke, "device": device_name(device),
              "sizes": sizes, "train_step_probes": probes,
              "untrained_generation": {"prompt": prompt, "output": untrained},
              "figures": figs_out, "meta": run_metadata(device, seed=args.seed)}
    print("saved", rel(save_json(result, out_dir / "memory_profile.json")))


if __name__ == "__main__":
    main()
