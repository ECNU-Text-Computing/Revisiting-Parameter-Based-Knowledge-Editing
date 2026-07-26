import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import os
import sys



# =========================
# utils
# =========================
def parse_list(arg, dtype=int):
    if arg is None or arg == "":
        return None
    return [dtype(x) for x in arg.split(",")]


# =========================
# collector (store raw hidden states per module)
# =========================
class Collector:

    def __init__(self):
        self.storage = defaultdict(list)

    def hook(self, name):

        def fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]  # [B, T, D]

            # store FULL sequence (IMPORTANT: no pooling here)
            self.storage[name].append(out.detach().cpu())

        return fn

    def clear(self):
        self.storage = defaultdict(list)


# =========================
# pooling (PAD-aware)
# =========================
def masked_pool(hidden, mask, pooling="mean"):

    # hidden: [B, T, D]
    # mask:   [B, T]

    mask = mask.unsqueeze(-1)  # [B, T, 1]

    if pooling == "mean":
        hidden = hidden * mask
        return hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    elif pooling == "last":
        idx = mask.squeeze(-1).sum(dim=1) - 1
        bsz = hidden.size(0)
        return hidden[torch.arange(bsz), idx]


# =========================
# register hooks
# =========================
def register_hooks(model, collector, config):

    hooks = []
    layers = model.model.layers

    for i, layer in enumerate(layers):

        if config["layers"] is not None and i not in config["layers"]:
            continue

        # -------- attention --------
        if "attn" in config["modules"]:
            attn = layer.self_attn

            for m in config["modules"]["attn"]:
                if m == "q_proj":
                    hooks.append(attn.q_proj.register_forward_hook(
                        collector.hook(f"layer{i}_q_proj")
                    ))
                if m == "k_proj":
                    hooks.append(attn.k_proj.register_forward_hook(
                        collector.hook(f"layer{i}_k_proj")
                    ))
                if m == "v_proj":
                    hooks.append(attn.v_proj.register_forward_hook(
                        collector.hook(f"layer{i}_v_proj")
                    ))
                if m == "o_proj":
                    hooks.append(attn.o_proj.register_forward_hook(
                        collector.hook(f"layer{i}_o_proj")
                    ))

        # -------- FFN --------
        if "mlp" in config["modules"]:
            mlp = layer.mlp

            for m in config["modules"]["mlp"]:
                if m == "gate_proj":
                    hooks.append(mlp.gate_proj.register_forward_hook(
                        collector.hook(f"layer{i}_gate_proj")
                    ))
                if m == "up_proj":
                    hooks.append(mlp.up_proj.register_forward_hook(
                        collector.hook(f"layer{i}_up_proj")
                    ))
                if m == "down_proj":
                    hooks.append(mlp.down_proj.register_forward_hook(
                        collector.hook(f"layer{i}_down_proj")
                    ))

        # -------- residual --------
        if config["modules"].get("residual", False):
            hooks.append(layer.register_forward_hook(
                collector.hook(f"layer{i}_residual")
            ))

    return hooks


# =========================
# forward
# =========================
@torch.no_grad()
def run(model, tokenizer, texts, config, pooling, max_len, batch_size, device):

    collector = Collector()
    hooks = register_hooks(model, collector, config)

    model.eval()

    all_results = defaultdict(list)

    for i in tqdm(range(0, len(texts), batch_size)):

        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        outputs = model(**inputs)
        attn_mask = inputs["attention_mask"].cpu()

        # =========================
        # apply pooling AFTER forward
        # =========================
        for name, feats in collector.storage.items():

            hidden = feats[-1]  # [B, T, D]

            pooled = masked_pool(hidden, attn_mask, pooling)

            all_results[name].append(pooled.cpu())

        collector.clear()

    # merge
    result = {}
    for k, v in all_results.items():
        result[k] = torch.cat(v, dim=0)

    for h in hooks:
        h.remove()

    return result


# =========================
# main
# =========================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["last", "mean"])

    parser.add_argument("--layers", type=str, default=None)

    parser.add_argument("--attn_modules", type=str, default=None)
    parser.add_argument("--mlp_modules", type=str, default=None)

    parser.add_argument("--use_residual", action="store_true")

    parser.add_argument("--save_path", type=str, default="activations.pt")

    args = parser.parse_args()

    # ------------------------
    # config
    # ------------------------
    config = {
        "layers": parse_list(args.layers),
        "modules": {}
    }

    if args.attn_modules:
        config["modules"]["attn"] = parse_list(args.attn_modules, str)

    if args.mlp_modules:
        config["modules"]["mlp"] = parse_list(args.mlp_modules, str)

    if args.use_residual:
        config["modules"]["residual"] = True

    print("\n===== CONFIG =====")
    print(config)
    print("Pooling:", args.pooling)

    if os.path.exists(args.save_path):
        print(f"⏭️ Skip exists: {args.save_path}")
        sys.exit(0)

    # ------------------------
    # model
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------
    # data (兼容纯文本和 JSONL)
    # ------------------------
    texts = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                texts.append(line)  # 纯文本直接使用

    # ------------------------
    # run
    # ------------------------
    outputs = run(
        model,
        tokenizer,
        texts,
        config,
        args.pooling,
        args.max_len,
        args.batch_size,
        device
    )

    torch.save(outputs, args.save_path)

    print("\nSaved to:", args.save_path)

    for k, v in outputs.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()