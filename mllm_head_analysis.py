"""
Phase 1 of EXP-5: Identify top-5 previous-token heads and top-5 induction
heads in Qwen2.5-VL-3B-Instruct via inference on VL-ICL Bench examples.
"""

import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("ERROR: transformers is not installed or does not have Qwen2.5-VL support.")
    sys.exit(1)

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False


NUM_LAYERS = 36
NUM_HEADS = 16


VL_ICL_DATA_ROOT = os.environ.get("VL_ICL_DATA_ROOT", "./data/VL-ICL")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify top PH and IH heads in Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--subtask", type=str, default="open_mi",
                        help="VL-ICL Bench subtask name")
    parser.add_argument("--data_dir", type=str, default=VL_ICL_DATA_ROOT,
                        help="Root directory of VL-ICL Bench data")
    parser.add_argument("--n_shot", type=int, default=2,
                        help="Number of shots per class for ICL")
    parser.add_argument("--num_examples", type=int, default=16,
                        help="Number of query examples to evaluate")
    parser.add_argument("--output_json", type=str, default="top_heads.json",
                        help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for demonstration selection")
    return parser.parse_args()


def load_vl_icl_data(subtask: str, num_examples: int,
                     data_dir: str = VL_ICL_DATA_ROOT,
                     n_shot: int = 2, seed: int = 0):
    """Load real VL-ICL Bench data.

    Tries the official pre-built ``{n_shot}-shot.json`` first (deterministic
    demonstrations from the benchmark).  Falls back to dynamic demonstration
    selection from ``query.json`` when the pre-built file is absent.
    """
    subtask_dir = os.path.join(data_dir, subtask)

    prebuilt = os.path.join(subtask_dir, f"{n_shot}-shot.json")
    if os.path.isfile(prebuilt):
        return _load_prebuilt(prebuilt, subtask_dir, data_dir,
                              n_shot, num_examples)

    query_path = os.path.join(subtask_dir, "query.json")
    if not os.path.isfile(query_path):
        raise FileNotFoundError(
            f"VL-ICL data not found at {subtask_dir}. "
            f"Make sure the VL-ICL dataset is at {data_dir}."
        )
    return _load_from_query_json(query_path, data_dir, n_shot,
                                 num_examples, seed)


def _load_prebuilt(path, subtask_dir, data_dir, n_shot, num_examples):
    """Load examples from an official pre-built ``{n_shot}-shot.json``."""
    with open(path, "r") as f:
        entries = json.load(f)

    shot_key = f"{n_shot}-shot"
    examples = []
    for entry in entries[:num_examples]:
        query = entry["query"]
        support_list = entry["support"][shot_key]
        shots = []
        for s in support_list:
            shots.append({
                "image": s["image"][0],
                "question": "This is a ",
                "answer": s["answer"],
            })
        examples.append({
            "shots": shots,
            "query_image": os.path.join(data_dir, query["image"][0]),
            "query_question": "This is a ",
            "answer": query["answer"],
        })
    print(f"  Loaded {len(examples)} VL-ICL examples from pre-built "
          f"{os.path.basename(path)} (n_shot={n_shot})")
    return examples


def _load_from_query_json(query_path, data_dir, n_shot, num_examples, seed):
    """Dynamically build demonstrations from ``query.json`` when no
    pre-built file is available."""
    with open(query_path, "r") as f:
        query_meta = json.load(f)

    rng = random.Random(seed)
    examples = []
    for query in query_meta[:num_examples]:
        shots = _select_demonstration_open_mi(query, n_shot, rng)
        examples.append({
            "shots": shots,
            "query_image": os.path.join(data_dir, query["image"][0]),
            "query_question": query["question"],
            "answer": query["answer"],
        })
    print(f"  Loaded {len(examples)} VL-ICL examples from query.json "
          f"(n_shot={n_shot}, dynamic selection)")
    return examples


def _select_demonstration_open_mi(query, n_shot, rng):
    """Replicate the VL-ICL ``select_demonstration`` for open_mi:
    pick 2 classes (query class + 1 random other), then for each of
    ``n_shot`` rounds add one image per class in random order."""
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
            shots.append({
                "image": support[cls]["images"][i],
                "question": "This is a ",
                "answer": cls,
            })
    return shots


def format_messages(example, data_dir=VL_ICL_DATA_ROOT):
    """Build Qwen2.5-VL multi-turn messages matching the VL-ICL codebase format.

    Produces:
      system: task instruction
      [for each shot]:  user: <image> + question  /  assistant: answer
      user: <query_image> + question
    """
    task_instruction = "Answer the question with a single word or phase."
    messages = [{"role": "system", "content": task_instruction}]

    for shot in example["shots"]:
        img_path = os.path.join(data_dir, shot["image"])
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": shot["question"]},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": shot["answer"]}],
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": example["query_image"]},
            {"type": "text", "text": example["query_question"]},
        ],
    })
    return messages


def prepare_inputs(processor, messages, device):
    """Tokenize and prepare model inputs from conversation messages."""
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

    if HAS_QWEN_VL_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_inputs.append(item["image"])
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _find_subsequence(seq, subseq):
    """Return all start positions where *subseq* occurs in *seq*."""
    positions = []
    n = len(subseq)
    if n == 0:
        return positions
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == subseq:
            positions.append(i)
    return positions


def find_answer_positions(input_ids, answer_token_ids, tokenizer=None, answer_text=None):
    """Find positions where the answer tokens appear in input_ids.

    BPE tokenizers produce different token IDs depending on whether a word
    has a leading space (e.g. " cat" vs "cat").  To handle this we search
    for *both* the bare encoding and the space-prefixed encoding.  If a
    ``tokenizer`` and the raw ``answer_text`` are provided we can do this
    automatically; otherwise we fall back to the bare token search only.

    Returns a list of ``(start_position, match_length)`` tuples so callers
    know exactly which token span each match covers."""
    seq = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    ans_bare = list(answer_token_ids)
    n_bare = len(ans_bare)

    matches = [(p, n_bare) for p in _find_subsequence(seq, ans_bare)]

    if tokenizer is not None and answer_text is not None:
        ans_space = tokenizer.encode(" " + answer_text, add_special_tokens=False)
        if ans_space != ans_bare:
            n_space = len(ans_space)
            matches.extend((p, n_space) for p in _find_subsequence(seq, ans_space))

    matches = sorted(set(matches))
    return matches


def compute_ph_strength(attn_weights):
    """Compute previous-token head strength for every (layer, head).
    attn_weights: tuple of (n_layers) tensors each of shape
                  [batch, n_heads, seq_len, seq_len].
    Returns np.ndarray of shape [n_layers, n_heads]."""
    scores = np.zeros((NUM_LAYERS, NUM_HEADS))
    for layer_idx, attn in enumerate(attn_weights):
        a = attn[0]  # [H, S, S]
        n_h = a.shape[0]
        diag = a.diagonal(offset=-1, dim1=-2, dim2=-1)  # [H, S-1]
        per_head = diag.mean(dim=-1).cpu().float().numpy()  # [H]

        if n_h == NUM_HEADS:
            scores[layer_idx] = per_head
        else:
            # GQA: expand KV heads to full head count
            repeat = NUM_HEADS // n_h
            scores[layer_idx] = np.repeat(per_head, repeat)
    return scores


def compute_ih_strength(attn_weights, input_ids, answer_token_ids,
                        tokenizer=None, answer_text=None):
    """Compute induction head strength for every (layer, head).

    Following the original definition in main_stage2.py, this is the
    *difference* between the mean attention the last token pays to answer
    positions and the mean attention it pays to non-answer positions.
    A head that attends uniformly scores ~0; a true induction head that
    selectively attends to correct-label positions scores positive.

    Returns np.ndarray of shape [n_layers, n_heads]."""
    scores = np.zeros((NUM_LAYERS, NUM_HEADS))
    matches = find_answer_positions(input_ids[0], answer_token_ids,
                                    tokenizer=tokenizer,
                                    answer_text=answer_text)
    if not matches:
        return scores

    answer_pos_set = set()
    for pos, length in matches:
        for offset in range(length):
            answer_pos_set.add(pos + offset)

    for layer_idx, attn in enumerate(attn_weights):
        a = attn[0]  # [H, S, S]
        n_h = a.shape[0]
        seq_len = a.shape[-1]
        last_token_attn = a[:, -1, :]  # [H, S]

        valid_answer_positions = [p for p in answer_pos_set if p < seq_len]
        non_answer_positions = [p for p in range(seq_len) if p not in answer_pos_set]

        if not valid_answer_positions:
            continue

        answer_idx = torch.tensor(valid_answer_positions, device=a.device)
        mean_answer_attn = last_token_attn[:, answer_idx].mean(dim=-1)  # [H]

        if non_answer_positions:
            non_answer_idx = torch.tensor(non_answer_positions, device=a.device)
            mean_non_answer_attn = last_token_attn[:, non_answer_idx].mean(dim=-1)
        else:
            mean_non_answer_attn = torch.zeros(n_h, device=a.device)

        per_head = (mean_answer_attn - mean_non_answer_attn).cpu().float().numpy()

        if n_h == NUM_HEADS:
            scores[layer_idx] = per_head
        else:
            repeat = NUM_HEADS // n_h
            scores[layer_idx] = np.repeat(per_head, repeat)
    return scores


def rank_heads(scores, top_k=5):
    """Return top-k (layer, head, score) tuples from a [layers, heads] array."""
    flat = scores.flatten()
    top_indices = np.argsort(flat)[::-1][:top_k]
    results = []
    for idx in top_indices:
        layer = int(idx // NUM_HEADS)
        head = int(idx % NUM_HEADS)
        results.append({"layer": layer, "head": head, "score": float(scores[layer, head])})
    return results


def print_summary(ph_heads, ih_heads):
    """Print a formatted summary table."""
    n_ph = len(ph_heads)
    n_ih = len(ih_heads)
    print("\n" + "=" * 60)
    print(f"  Top-{n_ph} Previous-Token (PH) Heads")
    print("-" * 60)
    print(f"  {'Rank':<6} {'Layer':<8} {'Head':<8} {'Score':<12}")
    print("-" * 60)
    for i, h in enumerate(ph_heads, 1):
        print(f"  {i:<6} {h['layer']:<8} {h['head']:<8} {h['score']:<12.6f}")

    print("\n" + "=" * 60)
    print(f"  Top-{n_ih} Induction (IH) Heads")
    print("-" * 60)
    print(f"  {'Rank':<6} {'Layer':<8} {'Head':<8} {'Score':<12}")
    print("-" * 60)
    for i, h in enumerate(ih_heads, 1):
        print(f"  {i:<6} {h['layer']:<8} {h['head']:<8} {h['score']:<12.6f}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    device = args.device

    print(f"Loading model Qwen/Qwen2.5-VL-3B-Instruct on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",  # required for full attention weight output
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Param count] Total: {total_params:,}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    print(f"Loading VL-ICL Bench data (subtask={args.subtask}, "
          f"n_shot={args.n_shot}, num_examples={args.num_examples}) ...")
    examples = load_vl_icl_data(
        args.subtask, args.num_examples,
        data_dir=args.data_dir, n_shot=args.n_shot, seed=args.seed,
    )

    ph_scores_accum = np.zeros((NUM_LAYERS, NUM_HEADS))
    ih_scores_accum = np.zeros((NUM_LAYERS, NUM_HEADS))
    n_valid_ih = 0

    for i, example in enumerate(examples):
        print(f"  Processing example {i + 1}/{len(examples)} ...", end="", flush=True)

        messages = format_messages(example, data_dir=args.data_dir)
        try:
            inputs = prepare_inputs(processor, messages, device)
        except Exception as e:
            print(f" skipped (input error: {e})")
            continue

        input_ids = inputs["input_ids"]

        answer_token_ids = processor.tokenizer.encode(
            example["answer"], add_special_tokens=False
        )

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            print(" skipped (OOM)")
            torch.cuda.empty_cache()
            continue

        attn_weights = outputs.attentions  # tuple of [1, H, S, S]

        ph_scores_accum += compute_ph_strength(attn_weights)

        ih = compute_ih_strength(attn_weights, input_ids, answer_token_ids,
                                 tokenizer=processor.tokenizer,
                                 answer_text=example["answer"])
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
    print(f"Results saved to {args.output_json}")

    print_summary(ph_heads, ih_heads)


if __name__ == "__main__":
    main()
