"""
Phase 2 of EXP-5: LoRA fine-tune Qwen2.5-VL-3B-Instruct on a VL-ICL subtask,
monitoring induction circuit dynamics (PH / IH head strength) during training.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("ERROR: transformers is not installed or does not have Qwen2.5-VL support.")
    sys.exit(1)
from peft import LoraConfig, get_peft_model


try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed. Logging will be print-only.")

from mllm_head_analysis import (
    load_vl_icl_data,
    format_messages,
    prepare_inputs,
    compute_ph_strength,
    compute_ih_strength,
    VL_ICL_DATA_ROOT,
    NUM_LAYERS,
    NUM_HEADS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: LoRA fine-tune Qwen2.5-VL-3B-Instruct with head monitoring"
    )
    parser.add_argument("--subtask", type=str, default="open_mi")
    parser.add_argument("--data_dir", type=str, default=VL_ICL_DATA_ROOT,
                        help="Root directory of VL-ICL Bench data")
    parser.add_argument("--n_shot", type=int, default=2,
                        help="Number of shots per class for ICL")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for demonstration selection")
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_eval", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str,
                        default="./outs_torch_icml_rebuttal/mllm_finetune/")
    parser.add_argument("--heads_json", type=str, default="top_heads.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_top_heads(json_path: str, top_k: int = 5):
    """Load top PH and IH heads from Phase 1 output JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    ph_heads = [(h["layer"], h["head"]) for h in data["ph_heads"][:top_k]]
    ih_heads = [(h["layer"], h["head"]) for h in data["ih_heads"][:top_k]]
    return ph_heads, ih_heads


def build_training_inputs(processor, example, device,
                          data_dir=VL_ICL_DATA_ROOT):
    """Build tokenised inputs *with* the answer appended, and return the
    number of prompt tokens so the loss mask covers only answer tokens.

    To avoid BPE boundary mismatches that arise from tokenizing the prompt
    separately, we compute ``prompt_len`` by subtracting the answer + EOS
    token count from the full sequence length."""
    messages = format_messages(example, data_dir=data_dir)

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    answer_text = example["answer"]
    eos = processor.tokenizer.eos_token
    full_text = prompt_text + answer_text + eos

    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_inputs.append(item["image"])

    inputs = processor(
        text=[full_text],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_with_eos_ids = processor.tokenizer.encode(
        answer_text + eos, add_special_tokens=False
    )
    n_answer_tokens = len(answer_with_eos_ids)
    full_len = inputs["input_ids"].shape[1]
    prompt_len = full_len - n_answer_tokens

    return inputs, prompt_len


def compute_loss(model, inputs, prompt_len):
    """Causal LM loss on answer tokens only."""
    outputs = model(**inputs)
    logits = outputs.logits  # [1, seq_len, vocab]
    input_ids = inputs["input_ids"]  # [1, seq_len]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    per_token_loss = per_token_loss.view(shift_labels.size())

    mask = torch.zeros_like(per_token_loss)
    answer_start = max(prompt_len - 1, 0)
    mask[:, answer_start:] = 1.0

    if mask.sum() == 0:
        return per_token_loss.mean()

    loss = (per_token_loss * mask).sum() / mask.sum()
    return loss


def eval_head_strengths(model, processor, examples, ph_heads, ih_heads,
                        device, data_dir=VL_ICL_DATA_ROOT):
    """Run evaluation: compute PH/IH strengths for tracked heads, ICL accuracy,
    and CLA (Copying from Labels Accuracy — whether the prediction is *any*
    in-context label, not necessarily the correct one)."""
    model.eval()

    ph_strengths = {(l, h): [] for l, h in ph_heads}
    ih_strengths = {(l, h): [] for l, h in ih_heads}
    correct = 0
    cla_correct = 0
    total = 0

    for example in examples:
        messages = format_messages(example, data_dir=data_dir)
        try:
            inputs = prepare_inputs(processor, messages, device)
        except Exception:
            continue

        input_ids = inputs["input_ids"]
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
                outputs = model(**inputs, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        attn_weights = outputs.attentions

        ph_scores = compute_ph_strength(attn_weights)
        for l, h in ph_heads:
            ph_strengths[(l, h)].append(ph_scores[l, h])

        ih_scores = compute_ih_strength(attn_weights, input_ids, answer_token_ids,
                                        tokenizer=processor.tokenizer,
                                        answer_text=answer_text)
        if ih_scores.max() > 0:
            for l, h in ih_heads:
                ih_strengths[(l, h)].append(ih_scores[l, h])

        pred_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
        if answer_token_ids and pred_token == answer_token_ids[0]:
            correct += 1
        if pred_token in icl_label_first_tokens:
            cla_correct += 1
        total += 1

        del outputs, attn_weights
        torch.cuda.empty_cache()

    model.train()

    mean_ph = {k: float(np.mean(v)) if v else 0.0 for k, v in ph_strengths.items()}
    mean_ih = {k: float(np.mean(v)) if v else 0.0 for k, v in ih_strengths.items()}
    accuracy = correct / total if total > 0 else 0.0
    cla = cla_correct / total if total > 0 else 0.0

    return mean_ph, mean_ih, accuracy, cla


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    if HAS_WANDB:
        wandb.init(
            project="ICL_torch",
            tags=["icml_rebuttal_mllm"],
            config=vars(args),
        )

    print(f"Loading top heads from {args.heads_json} ...")
    ph_heads, ih_heads = load_top_heads(args.heads_json)
    print(f"  PH heads: {ph_heads}")
    print(f"  IH heads: {ih_heads}")

    print(f"Loading model Qwen/Qwen2.5-VL-3B-Instruct on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    total_before_lora = sum(p.numel() for p in model.parameters())
    print(f"[Param count] Base model total: {total_before_lora:,}")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[Param count] After LoRA — Total: {total_params:,}  "
          f"Trainable: {trainable_params:,}  Frozen: {frozen_params:,}")
    if HAS_WANDB and wandb.run is not None:
        wandb.config.update({
            "base_model_params": total_before_lora,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        })

    print(f"Loading VL-ICL data (subtask={args.subtask}, n_shot={args.n_shot}) ...")
    total_needed = args.num_train + args.num_eval
    all_data = load_vl_icl_data(
        args.subtask, total_needed,
        data_dir=args.data_dir, n_shot=args.n_shot, seed=args.seed,
    )
    if len(all_data) < total_needed:
        print(f"WARNING: only {len(all_data)} examples available; "
              f"splitting proportionally.")
    train_data = all_data[: args.num_train]
    eval_data = all_data[args.num_train: args.num_train + args.num_eval]
    if not eval_data:
        eval_data = train_data[: args.num_eval]
    print(f"  Train examples: {len(train_data)}, Eval examples: {len(eval_data)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    print("\n--- Eval at step 0 (pre-training baseline) ---")
    mean_ph, mean_ih, accuracy, cla = eval_head_strengths(
        model, processor, eval_data, ph_heads, ih_heads, device,
        data_dir=args.data_dir,
    )
    avg_ph = float(np.mean(list(mean_ph.values()))) if mean_ph else 0.0
    avg_ih = float(np.mean(list(mean_ih.values()))) if mean_ih else 0.0
    metrics_0 = {
        "step": 0,
        "train_loss": 0.0,
        "eval_accuracy": accuracy,
        "eval_cla": cla,
        "avg_ph_strength": avg_ph,
        "avg_ih_strength": avg_ih,
    }
    for (l, h), v in mean_ph.items():
        metrics_0[f"ph_L{l}_H{h}"] = v
    for (l, h), v in mean_ih.items():
        metrics_0[f"ih_L{l}_H{h}"] = v
    print(f"  accuracy={accuracy:.4f}  CLA={cla:.4f}  "
          f"avg_PH={avg_ph:.6f}  avg_IH={avg_ih:.6f}")
    for (l, h), v in mean_ph.items():
        print(f"    PH L{l}/H{h}: {v:.6f}")
    for (l, h), v in mean_ih.items():
        print(f"    IH L{l}/H{h}: {v:.6f}")
    if HAS_WANDB:
        wandb.log(metrics_0)
    print("---\n")

    model.train()
    rng = np.random.RandomState(0)
    global_loss = 0.0

    print(f"Starting training for {args.steps} steps ...\n")

    for step in range(1, args.steps + 1):
        idx = rng.randint(0, len(train_data))
        example = train_data[idx]

        try:
            inputs, prompt_len = build_training_inputs(
                processor, example, device, data_dir=args.data_dir,
            )
        except Exception as e:
            print(f"  Step {step}: skipped (input error: {e})")
            continue

        loss = compute_loss(model, inputs, prompt_len)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_loss = loss.item()
        global_loss += step_loss

        del inputs, loss
        torch.cuda.empty_cache()

        if step % 10 == 0:
            steps_since_reset = step % args.eval_every or args.eval_every
            avg_loss = global_loss / steps_since_reset
            print(f"  Step {step}/{args.steps}  loss={step_loss:.4f}  "
                  f"avg_loss(recent)={avg_loss:.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n--- Eval at step {step} ---")
            mean_ph, mean_ih, accuracy, cla = eval_head_strengths(
                model, processor, eval_data, ph_heads, ih_heads, device,
                data_dir=args.data_dir,
            )

            avg_ph = float(np.mean(list(mean_ph.values()))) if mean_ph else 0.0
            avg_ih = float(np.mean(list(mean_ih.values()))) if mean_ih else 0.0

            metrics = {
                "step": step,
                "train_loss": step_loss,
                "eval_accuracy": accuracy,
                "eval_cla": cla,
                "avg_ph_strength": avg_ph,
                "avg_ih_strength": avg_ih,
            }
            for (l, h), v in mean_ph.items():
                metrics[f"ph_L{l}_H{h}"] = v
            for (l, h), v in mean_ih.items():
                metrics[f"ih_L{l}_H{h}"] = v

            print(f"  accuracy={accuracy:.4f}  CLA={cla:.4f}  "
                  f"avg_PH={avg_ph:.6f}  avg_IH={avg_ih:.6f}")
            for (l, h), v in mean_ph.items():
                print(f"    PH L{l}/H{h}: {v:.6f}")
            for (l, h), v in mean_ih.items():
                print(f"    IH L{l}/H{h}: {v:.6f}")

            if HAS_WANDB:
                wandb.log(metrics)

            print("---\n")

        if step % args.eval_every == 0:
            global_loss = 0.0

    save_path = os.path.join(args.output_dir, "lora_final")
    print(f"Saving LoRA checkpoint to {save_path} ...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print("Done.")

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
