"""
EXP-5 Knockout: Evaluate Qwen2.5-VL-3B-Instruct on VL-ICL open_mi,
then knock out the top-5 induction heads and evaluate again.
Compares accuracy and CLA before/after to measure the causal impact
of induction heads on in-context learning.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import numpy as np

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("ERROR: transformers is not installed or does not have Qwen2.5-VL support.")
    sys.exit(1)

from mllm_head_analysis import (
    load_vl_icl_data,
    format_messages,
    prepare_inputs,
    VL_ICL_DATA_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention head knockout experiment on Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--subtask", type=str, default="open_mi")
    parser.add_argument("--data_dir", type=str, default=VL_ICL_DATA_ROOT)
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_eval", type=int, default=50,
                        help="Number of examples to evaluate")
    parser.add_argument("--heads_json", type=str, default="top_heads.json")
    parser.add_argument("--knockout_type", type=str, default="ih",
                        choices=["ih", "ph", "both"],
                        help="Which head type(s) to knock out")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_top_heads(json_path, top_k=5):
    with open(json_path, "r") as f:
        data = json.load(f)
    ph_heads = [(h["layer"], h["head"]) for h in data["ph_heads"][:top_k]]
    ih_heads = [(h["layer"], h["head"]) for h in data["ih_heads"][:top_k]]
    return ph_heads, ih_heads


def get_attention_layers(model):
    """Resolve the path to decoder layers, handling different model wrappers."""
    for attr_path in [
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "layers"),
    ]:
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        "Cannot locate decoder layers. Tried: "
        "model.model.language_model.layers, model.model.model.layers, "
        "model.language_model.layers"
    )


def verify_layer_structure(layers, heads_to_knockout, num_heads, head_dim):
    """Assert that o_proj exists and has the expected input dimension."""
    for layer_idx, head_idx in heads_to_knockout:
        assert layer_idx < len(layers), (
            f"Layer {layer_idx} out of range (model has {len(layers)} layers)"
        )
        attn = layers[layer_idx].self_attn
        assert hasattr(attn, "o_proj"), (
            f"Layer {layer_idx}.self_attn has no o_proj attribute"
        )
        expected_in = num_heads * head_dim
        actual_in = attn.o_proj.in_features
        assert actual_in == expected_in, (
            f"Layer {layer_idx} o_proj.in_features={actual_in}, "
            f"expected {expected_in} (num_heads={num_heads} x head_dim={head_dim})"
        )
        assert head_idx < num_heads, (
            f"Head {head_idx} out of range (model has {num_heads} heads)"
        )


def make_knockout_hook(head_indices, head_dim):
    """Create a forward pre-hook for o_proj that zeros out specific heads.

    ``head_indices`` is a list of head indices to knock out in this layer.
    The input to o_proj is ``[bsz, seq_len, num_heads * head_dim]``;
    head ``h`` occupies the slice ``[h*head_dim : (h+1)*head_dim]``.
    """
    def hook(module, args):
        x = args[0]
        for h in head_indices:
            start = h * head_dim
            end = start + head_dim
            x[:, :, start:end] = 0.0
        return (x,) + args[1:]
    return hook


def register_knockout_hooks(layers, heads_to_knockout, head_dim):
    """Register one hook per affected layer. Returns handles for later removal."""
    layer_to_heads = defaultdict(list)
    for layer_idx, head_idx in heads_to_knockout:
        layer_to_heads[layer_idx].append(head_idx)

    handles = []
    for layer_idx, head_indices in layer_to_heads.items():
        o_proj = layers[layer_idx].self_attn.o_proj
        hook_fn = make_knockout_hook(head_indices, head_dim)
        handle = o_proj.register_forward_pre_hook(hook_fn)
        handles.append(handle)
    return handles


def evaluate(model, processor, examples, device, data_dir):
    """Evaluate accuracy and CLA (no attention weight extraction needed)."""
    model.eval()
    correct = 0
    cla_correct = 0
    total = 0

    for i, example in enumerate(examples):
        messages = format_messages(example, data_dir=data_dir)
        try:
            inputs = prepare_inputs(processor, messages, device)
        except Exception:
            continue

        answer_text = example["answer"]
        answer_token_ids = processor.tokenizer.encode(
            answer_text, add_special_tokens=False
        )

        icl_label_first_tokens = set()
        for shot in example["shots"]:
            first_tok = processor.tokenizer.encode(
                shot["answer"], add_special_tokens=False
            )
            if first_tok:
                icl_label_first_tokens.add(first_tok[0])
            space_tok = processor.tokenizer.encode(
                " " + shot["answer"], add_special_tokens=False
            )
            if space_tok:
                icl_label_first_tokens.add(space_tok[0])

        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        pred_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
        if answer_token_ids and pred_token == answer_token_ids[0]:
            correct += 1
        if pred_token in icl_label_first_tokens:
            cla_correct += 1
        total += 1

        del outputs
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(examples)} ...", flush=True)

    accuracy = correct / total if total > 0 else 0.0
    cla = cla_correct / total if total > 0 else 0.0
    return accuracy, cla, total


def format_head_list(heads):
    return ", ".join(f"L{l}/H{h}" for l, h in heads)


def main():
    args = parse_args()
    device = args.device

    print(f"Loading top heads from {args.heads_json} ...")
    ph_heads, ih_heads = load_top_heads(args.heads_json)
    print(f"  PH heads: {format_head_list(ph_heads)}")
    print(f"  IH heads: {format_head_list(ih_heads)}")

    if args.knockout_type == "ih":
        heads_to_knockout = ih_heads
        knockout_label = "IH"
    elif args.knockout_type == "ph":
        heads_to_knockout = ph_heads
        knockout_label = "PH"
    else:
        heads_to_knockout = ih_heads + ph_heads
        knockout_label = "IH+PH"

    print(f"\nKnockout target ({knockout_label}): {format_head_list(heads_to_knockout)}")

    print(f"\nLoading model Qwen/Qwen2.5-VL-3B-Instruct on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    text_config = model.config if not hasattr(model.config, "text_config") else model.config.text_config
    num_heads = text_config.num_attention_heads
    head_dim = text_config.hidden_size // num_heads
    print(f"  num_heads={num_heads}, head_dim={head_dim}")

    layers = get_attention_layers(model)
    print(f"  Found {len(layers)} decoder layers")
    verify_layer_structure(layers, heads_to_knockout, num_heads, head_dim)
    print("  Layer structure verified OK")

    print(f"\nLoading VL-ICL data (subtask={args.subtask}, n_shot={args.n_shot}) ...")
    examples = load_vl_icl_data(
        args.subtask, args.num_eval,
        data_dir=args.data_dir, n_shot=args.n_shot, seed=args.seed,
    )
    print(f"  {len(examples)} eval examples loaded")

    # --- Baseline evaluation ---
    print("\n" + "=" * 60)
    print("  Baseline Evaluation (no knockout)")
    print("=" * 60)
    baseline_acc, baseline_cla, n = evaluate(
        model, processor, examples, device, args.data_dir
    )
    print(f"  accuracy={baseline_acc:.4f}  CLA={baseline_cla:.4f}  "
          f"({n} examples)")

    # --- Knockout evaluation ---
    print("\n" + "=" * 60)
    print(f"  {knockout_label} Knockout Evaluation "
          f"({len(heads_to_knockout)} heads zeroed)")
    print("=" * 60)
    handles = register_knockout_hooks(layers, heads_to_knockout, head_dim)
    print(f"  Registered {len(handles)} hooks")

    knockout_acc, knockout_cla, n = evaluate(
        model, processor, examples, device, args.data_dir
    )
    print(f"  accuracy={knockout_acc:.4f}  CLA={knockout_cla:.4f}  "
          f"({n} examples)")

    for h in handles:
        h.remove()

    # --- Comparison ---
    acc_diff = knockout_acc - baseline_acc
    cla_diff = knockout_cla - baseline_cla
    print("\n" + "=" * 60)
    print("  Comparison")
    print("=" * 60)
    print(f"  Baseline   — accuracy={baseline_acc:.4f}  CLA={baseline_cla:.4f}")
    print(f"  Knockout   — accuracy={knockout_acc:.4f}  CLA={knockout_cla:.4f}")
    print(f"  Difference — accuracy={acc_diff:+.4f}  CLA={cla_diff:+.4f}")
    print(f"  Knocked-out heads ({knockout_label}): "
          f"{format_head_list(heads_to_knockout)}")
    print("=" * 60)

    result = {
        "knockout_type": args.knockout_type,
        "heads": [{"layer": l, "head": h} for l, h in heads_to_knockout],
        "baseline": {"accuracy": baseline_acc, "cla": baseline_cla},
        "knockout": {"accuracy": knockout_acc, "cla": knockout_cla},
        "diff": {"accuracy": acc_diff, "cla": cla_diff},
        "num_eval": n,
    }
    out_path = os.path.join(
        os.path.dirname(args.heads_json) or ".",
        f"knockout_{args.knockout_type}_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
