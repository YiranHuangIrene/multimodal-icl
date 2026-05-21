"""
Identify top-5 previous-token heads and top-5 induction heads in
Qwen2.5-3B-Instruct (the text-only LLM backbone) using text ICL tasks
derived from VL-ICL open_mi data.  Then compare with the MLLM heads.
"""

import os
import sys
import json
import argparse
import random
import torch
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers is not installed.")
    sys.exit(1)

from mllm_head_analysis import (
    compute_ph_strength,
    compute_ih_strength,
    find_answer_positions,
    rank_heads,
    print_summary,
    NUM_LAYERS,
    NUM_HEADS,
    VL_ICL_DATA_ROOT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify top PH and IH heads in Qwen2.5-3B-Instruct (text-only LLM)"
    )
    parser.add_argument("--data_dir", type=str, default=VL_ICL_DATA_ROOT)
    parser.add_argument("--subtask", type=str, default="open_mi")
    parser.add_argument("--n_shot", type=int, default=2)
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--output_json", type=str, default="top_heads_llm.json")
    parser.add_argument("--mllm_heads_json", type=str, default="top_heads.json",
                        help="MLLM heads JSON for overlap comparison")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model verification
# ---------------------------------------------------------------------------

def verify_model_architecture(model, tokenizer):
    """Assert the LLM architecture matches the expected MLLM backbone
    (36 layers, 16 attention heads, head_dim=128).  Prints a summary
    and raises on mismatch."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    print("=" * 60)
    print("  Architecture Verification")
    print("-" * 60)
    print(f"  num_hidden_layers : {num_layers}  (expected {NUM_LAYERS})")
    print(f"  num_attention_heads: {num_heads}  (expected {NUM_HEADS})")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  hidden_size       : {hidden_size}")
    print(f"  head_dim          : {head_dim}  (expected 128)")
    print(f"  vocab_size        : {cfg.vocab_size}")

    ok = True
    if num_layers != NUM_LAYERS:
        print(f"  [FAIL] num_layers={num_layers} != {NUM_LAYERS}")
        ok = False
    if num_heads != NUM_HEADS:
        print(f"  [FAIL] num_heads={num_heads} != {NUM_HEADS}")
        ok = False
    if head_dim != 128:
        print(f"  [FAIL] head_dim={head_dim} != 128")
        ok = False

    if ok:
        print("  [PASS] Architecture matches MLLM backbone")
    else:
        print("=" * 60)
        raise AssertionError("Architecture mismatch -- aborting")
    print("=" * 60)
    return num_kv_heads


def verify_attention_output(model, tokenizer, device):
    """Run a tiny forward pass and verify attention output shapes."""
    dummy = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**dummy, output_attentions=True)

    attns = out.attentions
    n_layers = len(attns)
    shape0 = attns[0].shape

    print("=" * 60)
    print("  Attention Output Shape Check")
    print("-" * 60)
    print(f"  num layers returned : {n_layers}  (expected {NUM_LAYERS})")
    print(f"  layer 0 shape       : {list(shape0)}")
    print(f"  (batch, heads, seq, seq)")

    ok = True
    if n_layers != NUM_LAYERS:
        print(f"  [FAIL] got {n_layers} layers, expected {NUM_LAYERS}")
        ok = False
    if shape0[0] != 1:
        print(f"  [FAIL] batch dim = {shape0[0]}")
        ok = False

    if ok:
        print("  [PASS] Attention shapes correct")
    else:
        raise AssertionError("Attention output shape mismatch")
    print("=" * 60)

    del out, dummy
    torch.cuda.empty_cache()


def verify_answer_tokens(tokenizer, example, input_ids):
    """Print tokenization and verify answer positions are found."""
    answer_text = example["answer"]
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    matches = find_answer_positions(
        input_ids[0], answer_token_ids,
        tokenizer=tokenizer, answer_text=answer_text,
    )

    print("=" * 60)
    print("  Answer Token Sanity Check (first example)")
    print("-" * 60)
    print(f"  answer text   : '{answer_text}'")
    print(f"  answer tokens : {answer_token_ids}")
    print(f"  input length  : {input_ids.shape[1]} tokens")
    print(f"  matches found : {len(matches)}")
    for pos, length in matches:
        span = input_ids[0, pos:pos+length].tolist()
        decoded = tokenizer.decode(span)
        print(f"    pos {pos}..{pos+length-1}: ids={span} -> '{decoded}'")

    if not matches:
        print("  [FAIL] No answer positions found!")
        raise AssertionError("Answer tokens not found in prompt")
    print("  [PASS] Answer tokens located")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Data loading (text-only ICL from VL-ICL query.json)
# ---------------------------------------------------------------------------

def load_text_icl_data(subtask, num_examples, data_dir=VL_ICL_DATA_ROOT,
                       n_shot=2, seed=0):
    """Load VL-ICL query.json and build text-only ICL examples using
    real_name descriptions instead of images."""
    query_path = os.path.join(data_dir, subtask, "query.json")
    if not os.path.isfile(query_path):
        raise FileNotFoundError(f"query.json not found at {query_path}")

    with open(query_path, "r") as f:
        query_meta = json.load(f)

    rng = random.Random(seed)
    examples = []
    for query in query_meta[:num_examples]:
        shots = _select_text_demonstration(query, n_shot, rng)
        examples.append({
            "shots": shots,
            "query_description": f"This is a picture of a {query['real_name']}.",
            "query_question": "This is a ",
            "answer": query["answer"],
        })
    print(f"  Loaded {len(examples)} text ICL examples from query.json "
          f"(n_shot={n_shot})")
    return examples


def _select_text_demonstration(query, n_shot, rng):
    """Pick 2 classes and build n_shot rounds of text demonstrations
    with real_name descriptions."""
    query_class = query["answer"]
    other_class = rng.choice(
        [c for c in query["classes"] if c != query_class]
    )
    order_keys = [query_class, other_class]
    if rng.random() < 0.5:
        order_keys = order_keys[::-1]

    shots = []
    support = query["support"]
    for i in range(n_shot):
        for cls in order_keys:
            real_name = support[cls]["real_name"]
            shots.append({
                "description": f"This is a picture of a {real_name}.",
                "question": "This is a ",
                "answer": cls,
            })
    return shots


# ---------------------------------------------------------------------------
# Message formatting and tokenization (text-only)
# ---------------------------------------------------------------------------

def format_text_messages(example):
    """Build Qwen2.5 chat messages for a text-only ICL example."""
    task_instruction = "Answer the question with a single word or phase."
    messages = [{"role": "system", "content": task_instruction}]

    for shot in example["shots"]:
        messages.append({
            "role": "user",
            "content": f"{shot['description']} {shot['question']}",
        })
        messages.append({
            "role": "assistant",
            "content": shot["answer"],
        })

    messages.append({
        "role": "user",
        "content": f"{example['query_description']} {example['query_question']}",
    })
    return messages


def prepare_text_inputs(tokenizer, messages, device):
    """Tokenize text-only chat messages for the LLM."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Overlap comparison
# ---------------------------------------------------------------------------

def compare_heads(llm_path, mllm_path, k_values=(5, 10, 20)):
    """Load LLM and MLLM head JSONs and print overlap analysis at
    multiple top-K levels."""
    with open(llm_path) as f:
        llm = json.load(f)
    with open(mllm_path) as f:
        mllm = json.load(f)

    def _to_set(heads_list, k):
        return {(h["layer"], h["head"]) for h in heads_list[:k]}

    def _fmt(s):
        return ", ".join(f"L{l}/H{h}" for l, h in sorted(s))

    print("\n" + "=" * 70)
    print("  Head Overlap: LLM (Qwen2.5-3B) vs MLLM (Qwen2.5-VL-3B)")
    print("=" * 70)

    for head_type, key in [("PH", "ph_heads"), ("IH", "ih_heads")]:
        print(f"\n  --- {head_type} heads ---")
        for k in k_values:
            llm_avail = min(k, len(llm[key]))
            mllm_avail = min(k, len(mllm[key]))
            llm_set = _to_set(llm[key], k)
            mllm_set = _to_set(mllm[key], k)
            overlap = llm_set & mllm_set
            llm_only = llm_set - mllm_set
            mllm_only = mllm_set - llm_set

            print(f"\n  top-{k} (LLM has {llm_avail}, MLLM has {mllm_avail}):")
            print(f"    LLM  : {_fmt(llm_set)}")
            print(f"    MLLM : {_fmt(mllm_set)}")
            print(f"    Overlap  : {len(overlap)}/{min(llm_avail, mllm_avail)}  "
                  f"{_fmt(overlap) if overlap else '(none)'}")
            print(f"    LLM-only : {_fmt(llm_only) if llm_only else '(none)'}")
            print(f"    MLLM-only: {_fmt(mllm_only) if mllm_only else '(none)'}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device

    # ---- Load model ----
    print(f"Loading model Qwen/Qwen2.5-3B-Instruct on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Param count] Total: {total_params:,}")

    # ---- Verification step 1: architecture ----
    num_kv_heads = verify_model_architecture(model, tokenizer)

    # ---- Verification step 2: attention output shapes ----
    verify_attention_output(model, tokenizer, device)

    # ---- Load data ----
    print(f"\nLoading text ICL data (subtask={args.subtask}, "
          f"n_shot={args.n_shot}, num_examples={args.num_examples}) ...")
    examples = load_text_icl_data(
        args.subtask, args.num_examples,
        data_dir=args.data_dir, n_shot=args.n_shot, seed=args.seed,
    )

    # ---- Verification step 3: answer token sanity check (first example) ----
    first_msgs = format_text_messages(examples[0])
    first_inputs = prepare_text_inputs(tokenizer, first_msgs, device)
    verify_answer_tokens(tokenizer, examples[0], first_inputs["input_ids"])
    del first_inputs
    torch.cuda.empty_cache()

    # ---- Main analysis loop ----
    print(f"\nRunning PH/IH analysis on {len(examples)} examples ...\n")
    ph_scores_accum = np.zeros((NUM_LAYERS, NUM_HEADS))
    ih_scores_accum = np.zeros((NUM_LAYERS, NUM_HEADS))
    n_valid_ih = 0

    for i, example in enumerate(examples):
        print(f"  Processing example {i + 1}/{len(examples)} ...", end="", flush=True)

        messages = format_text_messages(example)
        try:
            inputs = prepare_text_inputs(tokenizer, messages, device)
        except Exception as e:
            print(f" skipped (input error: {e})")
            continue

        input_ids = inputs["input_ids"]
        answer_token_ids = tokenizer.encode(
            example["answer"], add_special_tokens=False
        )

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            print(" skipped (OOM)")
            torch.cuda.empty_cache()
            continue

        attn_weights = outputs.attentions

        ph_scores_accum += compute_ph_strength(attn_weights)

        ih = compute_ih_strength(
            attn_weights, input_ids, answer_token_ids,
            tokenizer=tokenizer, answer_text=example["answer"],
        )
        if ih.max() > 0:
            ih_scores_accum += ih
            n_valid_ih += 1

        del outputs, attn_weights
        torch.cuda.empty_cache()
        print(" done")

    n_examples = len(examples)
    if n_examples > 0:
        ph_scores_accum /= n_examples
    if n_valid_ih > 0:
        ih_scores_accum /= n_valid_ih

    ph_heads = rank_heads(ph_scores_accum, top_k=20)
    ih_heads = rank_heads(ih_scores_accum, top_k=20)

    result = {"ph_heads": ph_heads, "ih_heads": ih_heads}
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
    print_summary(ph_heads, ih_heads)

    # ---- Overlap comparison ----
    if os.path.isfile(args.mllm_heads_json):
        compare_heads(args.output_json, args.mllm_heads_json)
    else:
        print(f"\nMMLM heads file not found at {args.mllm_heads_json}, "
              f"skipping overlap comparison.")


if __name__ == "__main__":
    main()
