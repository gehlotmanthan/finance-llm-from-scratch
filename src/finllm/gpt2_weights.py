"""Chapter 5.5 — loading OpenAI's pretrained GPT-2 weights into our own GPTModel.

The book downloads the original TensorFlow checkpoint. We use the same weights
in safetensors format from Hugging Face (openai-community/gpt2*), which needs no
TensorFlow -- the approach in the book repo's ch05/02_alternative_weight_loading.

Mapping (HF uses a "Conv1D" layer whose weight is stored (in, out), so every
matrix is transposed for our nn.Linear, which stores (out, in)):

    wte / wpe                       -> tok_emb / pos_emb
    h.{i}.attn.c_attn  (D, 3D)      -> split into W_query / W_key / W_value
    h.{i}.attn.c_proj               -> att.out_proj
    h.{i}.mlp.c_fc / c_proj         -> ff.layers[0] / ff.layers[2]
    h.{i}.ln_1 / ln_2               -> norm1 / norm2   (weight->scale, bias->shift)
    ln_f                            -> final_norm
    wte (again)                     -> out_head        (weight tying)
"""

from pathlib import Path

import requests
import torch
from tqdm import tqdm

from .config import HF_REPOS, pretrained_config, resolve_name
from .model import GPTModel
from .paths import MODELS_DIR

HF_URL = "https://huggingface.co/openai-community/{repo}/resolve/main/model.safetensors"


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download with a progress bar; skip if already complete. Writes to .part first."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    tmp.replace(dest)
    return dest


def download_gpt2(model_name: str = "gpt2-small (124M)", models_dir: Path = MODELS_DIR) -> Path:
    repo = HF_REPOS[resolve_name(model_name)]
    return download_file(HF_URL.format(repo=repo), Path(models_dir) / repo / "model.safetensors")


def load_hf_params(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    params = load_file(str(path))
    # Some exports prefix keys with "transformer."
    return {k.removeprefix("transformer."): v for k, v in params.items()}


def _assign(left: torch.Tensor, right: torch.Tensor, name: str) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch for {name}: model {tuple(left.shape)} vs checkpoint {tuple(right.shape)}")
    return torch.nn.Parameter(right.clone().detach().to(left.dtype))


def load_weights_into_gpt(gpt: GPTModel, params: dict[str, torch.Tensor]) -> GPTModel:
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe.weight"], "wpe")
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte.weight"], "wte")
    for b, blk in enumerate(gpt.trf_blocks):
        p = f"h.{b}."
        q_w, k_w, v_w = torch.chunk(params[p + "attn.c_attn.weight"], 3, dim=-1)
        blk.att.W_query.weight = _assign(blk.att.W_query.weight, q_w.T, p + "q_w")
        blk.att.W_key.weight = _assign(blk.att.W_key.weight, k_w.T, p + "k_w")
        blk.att.W_value.weight = _assign(blk.att.W_value.weight, v_w.T, p + "v_w")
        if blk.att.W_query.bias is None:
            raise ValueError("GPT-2 weights need qkv_bias=True (use config.pretrained_config)")
        q_b, k_b, v_b = torch.chunk(params[p + "attn.c_attn.bias"], 3, dim=-1)
        blk.att.W_query.bias = _assign(blk.att.W_query.bias, q_b, p + "q_b")
        blk.att.W_key.bias = _assign(blk.att.W_key.bias, k_b, p + "k_b")
        blk.att.W_value.bias = _assign(blk.att.W_value.bias, v_b, p + "v_b")
        blk.att.out_proj.weight = _assign(blk.att.out_proj.weight, params[p + "attn.c_proj.weight"].T, p + "c_proj")
        blk.att.out_proj.bias = _assign(blk.att.out_proj.bias, params[p + "attn.c_proj.bias"], p + "c_proj.b")
        blk.ff.layers[0].weight = _assign(blk.ff.layers[0].weight, params[p + "mlp.c_fc.weight"].T, p + "c_fc")
        blk.ff.layers[0].bias = _assign(blk.ff.layers[0].bias, params[p + "mlp.c_fc.bias"], p + "c_fc.b")
        blk.ff.layers[2].weight = _assign(blk.ff.layers[2].weight, params[p + "mlp.c_proj.weight"].T, p + "mlp.c_proj")
        blk.ff.layers[2].bias = _assign(blk.ff.layers[2].bias, params[p + "mlp.c_proj.bias"], p + "mlp.c_proj.b")
        blk.norm1.scale = _assign(blk.norm1.scale, params[p + "ln_1.weight"], p + "ln_1")
        blk.norm1.shift = _assign(blk.norm1.shift, params[p + "ln_1.bias"], p + "ln_1.b")
        blk.norm2.scale = _assign(blk.norm2.scale, params[p + "ln_2.weight"], p + "ln_2")
        blk.norm2.shift = _assign(blk.norm2.shift, params[p + "ln_2.bias"], p + "ln_2.b")
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["ln_f.weight"], "ln_f")
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["ln_f.bias"], "ln_f.b")
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte.weight"], "out_head")
    return gpt


def gpt_to_hf_params(gpt: GPTModel) -> dict[str, torch.Tensor]:
    """Inverse of ``load_weights_into_gpt`` (used by the tests to prove the mapping round-trips)."""
    out = {"wte.weight": gpt.tok_emb.weight.detach().clone(),
           "wpe.weight": gpt.pos_emb.weight.detach().clone()}
    for b, blk in enumerate(gpt.trf_blocks):
        p = f"h.{b}."
        a = blk.att
        out[p + "attn.c_attn.weight"] = torch.cat([a.W_query.weight.T, a.W_key.weight.T, a.W_value.weight.T], dim=-1).detach().clone()
        out[p + "attn.c_attn.bias"] = torch.cat([a.W_query.bias, a.W_key.bias, a.W_value.bias]).detach().clone()
        out[p + "attn.c_proj.weight"] = a.out_proj.weight.T.detach().clone()
        out[p + "attn.c_proj.bias"] = a.out_proj.bias.detach().clone()
        out[p + "mlp.c_fc.weight"] = blk.ff.layers[0].weight.T.detach().clone()
        out[p + "mlp.c_fc.bias"] = blk.ff.layers[0].bias.detach().clone()
        out[p + "mlp.c_proj.weight"] = blk.ff.layers[2].weight.T.detach().clone()
        out[p + "mlp.c_proj.bias"] = blk.ff.layers[2].bias.detach().clone()
        out[p + "ln_1.weight"] = blk.norm1.scale.detach().clone()
        out[p + "ln_1.bias"] = blk.norm1.shift.detach().clone()
        out[p + "ln_2.weight"] = blk.norm2.scale.detach().clone()
        out[p + "ln_2.bias"] = blk.norm2.shift.detach().clone()
    out["ln_f.weight"] = gpt.final_norm.scale.detach().clone()
    out["ln_f.bias"] = gpt.final_norm.shift.detach().clone()
    return out


def load_pretrained_gpt2(model_name: str = "gpt2-small (124M)", device="cpu",
                         models_dir: Path = MODELS_DIR, **cfg_overrides):
    """Download (once) and load OpenAI GPT-2 into our GPTModel. Returns (model, cfg)."""
    cfg = pretrained_config(model_name, **cfg_overrides)
    path = download_gpt2(model_name, models_dir)
    model = GPTModel(cfg)
    load_weights_into_gpt(model, load_hf_params(path))
    return model.to(device).eval(), cfg
