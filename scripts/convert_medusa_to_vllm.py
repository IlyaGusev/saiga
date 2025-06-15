from __future__ import annotations
import json, re, torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import fire
from tqdm import tqdm


def convert(
    src: str,
    out: str,
    original_lm_head: bool = False,
    truncated_vocab_size: int | None = None,
    token_map: str | None = None,
):
    raw = load_file(src)  # keys like "0.0.linear.weight"
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    head_re = re.compile(r"^(\d+)\.(\d+)\.linear\.weight$")
    proj_re = re.compile(r"^(\d+)\.\d+\.weight$")
    heads, blocks = set(), set()
    hidden_size = vocab_size = None

    print("RAW:", raw)
    for k, v in tqdm(raw.items(), desc="Infering config..."):
        m = head_re.match(k)
        if m:
            h, b = map(int, m.groups())
            heads.add(h)
            blocks.add(b)
            hidden_size = hidden_size or v.shape[0]
        elif ".linear." not in k and k.endswith(".weight"):
            vocab_size = vocab_size or v.shape[0]

    num_heads = max(heads) + 1
    num_hidden_layers = max(blocks) + 1

    if None in (hidden_size, vocab_size):
        raise RuntimeError("Could not infer hidden- or vocab-size from file.")

    new_state = {}
    for k, v in tqdm(raw.items(), desc="Renaming tensors"):
        m = head_re.match(k)
        if m:  # residual block weight/bias
            h, b = m.groups()
            #new_state[f"medusa_heads.{h}.{b}.linear.weight"] = v
            new_state[f"blocks.{h}.layers.{b}.weight"] = v
            bias_key = f"{h}.{b}.linear.bias"
            if bias_key in raw:
                #new_state[f"medusa_heads.{h}.{b}.linear.bias"] = raw[bias_key]
                new_state[f"blocks.{h}.layers.{b}.bias"] = raw[bias_key]
        elif ".linear." not in k and k.endswith(".weight"):
            h = k.split(".")[0]
            if not original_lm_head:
                new_state[f"lm_heads.{h}.weight"] = v

    # token-map for small-vocab Medusa
    if truncated_vocab_size is not None:
        if token_map is None:
            raise ValueError("token_map is required with truncated_vocab_size")
        new_state["token_map"] = torch.load(token_map)

    print("Saving...")
    save_file(new_state, out_dir / "model.safetensors")

    # ---------- write config.json ---------------------------------------------
    cfg = dict(
        architectures=["MedusaModel"],
        model_type="medusa",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_hidden_layers=num_hidden_layers,
        max_paths=64,
        topk=10,
        original_lm_head=original_lm_head,
        truncated_vocab_size=truncated_vocab_size,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"✅ Draft ready in {out_dir.resolve()}\n"
        f"   heads={num_heads}, layers/head={num_hidden_layers}, "
        f"hidden={hidden_size}, vocab={vocab_size}"
    )


if __name__ == "__main__":
    fire.Fire(convert)
