# Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in Modern Transformers

Code for the ICML 2026 **SPOTLIGHT** paper.

Authors: **Yiran Huang, Karsten Roth, Quentin Bouniot, Wenjia Xu, Zeynep Akata.**

This repository contains a controlled, mechanistic study of how
transformer-based decoders acquire in-context learning (ICL) capabilities in
both unimodal and multimodal settings and how these findings generalize to real MLLMs. The codebase implements a three-stage
training pipeline on synthetic Gaussian-mixture data (and, optionally, on
Omniglot / Mini-ImageNet), together with the progress-measure metrics used in
the paper (PHStrength, IHStrength, TILA, TIIA, TLA).

## Setup

```bash
pip install torch numpy wandb tqdm einops torchvision scipy
# Only needed for the MLLM analysis (Sec. Qwen2.5-VL):
pip install transformers peft qwen-vl-utils
```

Tested with Python 3.10 and PyTorch 2.x on a single A100 GPU. Each `main_*.py`
takes positional CLI arguments and can be launched directly; for cluster runs
wrap the command in a SLURM submission script of your choice.

WandB logging is enabled by default (`WANDB = True` at the top of each
`main_*.py`). Set `WANDB_ENTITY` in your environment before running, or set
`WANDB = False` to disable.

## Data

| Source         | How to obtain                                                            | Used by                                     |
| -------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| Synthetic GMM  | Generated on the fly by `dataset.py`                                     | All stages                                  |
| Omniglot       | Auto-downloaded via `torchvision.datasets.Omniglot`                       | `main_encoder_vit_omniglot.py`, `main_mm_all_vit_omniglot.py` |
| Mini-ImageNet  | `python prepare_mini_imagenet.py --imagenet-train <PATH> --output-root <PATH>` | `main_encoder_vit_miniimagenet.py`, `main_mm_all_vit_miniimagenet.py` |
| VL-ICL Bench   | Clone <https://github.com/ys-zong/VL-ICL>; set `VL_ICL_DATA_ROOT`        | `mllm_head_analysis.py`, `mllm_knockout.py`, `mllm_finetune.py` |

Dataset roots are read from environment variables (`MINI_IMAGENET_ROOT`,
`VL_ICL_DATA_ROOT`) and default to `./data/<dataset>/`.

## Sequence Formats

- **Unimodal (Stage 1):** `[x_1, y_1, x_2, y_2, ..., x_N, y_N, x_query]` — length `2N + 1`. Loss is computed at the final position.
- **Multimodal (Stage 2 / Stage 3):** `[M1_1, M2_1, y_1, ..., M1_N, M2_N, y_N, M1_query, M2_query]` — length `3N + 2`. Each context triplet is `(M1_i, M2_slot_i, y_i)`; the trailing pair `(M1_query, M2_query)` is the query, and loss is computed at the final position. The M2 slots are populated by projected encoder features at runtime; `inputs_mm[..., 3*N+1]` is reserved for the projected M2 query.

When running multimodal training, `max_position_embeddings` must be set to `3*N + 2`; the provided `main_mm_*.py` scripts already do this.

## Three-Stage Pipeline

All training scripts take positional CLI arguments (no `argparse`). The
argument order is defined at the top of each `main_*.py`.

### Stage 1 — Unimodal decoder pretraining

```bash
python main_stage1.py K N D L alpha B p_B p_C eps no_repeats \
                      n_heads n_layers rope rope_theta alibi hybrid \
                      rms_norm batch_size optimizer save_ckpt \
                      progress_measure device
```

Variants:
- `main_stage1_larger_model.py` — width/depth scaling sweeps (Fig. 2a-b)

### Stage 2 — Projector + frozen decoder

```bash
python main_stage2.py K1 K2 N D1 D2 L1 L2 alpha1 alpha2 B p_B p_C eps1 eps2 \
                      no_repeats n_heads n_layers rope rope_theta rms_norm \
                      L_pos freeze_layers ckpt_path early_fusion batch_size \
                      optimizer save_ckpt progress_measure device
```

Variants:
- `main_stage2_larger_model.py` — scaling at Stage 2
- `main_stage2_joint_label.py` — joint-label setting (rebuttal Q1, dkFE)

### Stage 3 — Encoder pretraining + full MLLM

Encoder pretraining (choose one of):
- `main_encoder_mlp.py`, `main_encoder_CNN.py`, `main_encoder_transformer.py`,
  `main_encoder_ViT.py`
- `main_encoder_vit_omniglot.py`, `main_encoder_vit_miniimagenet.py`

Joint training (encoder + projector + decoder):
- `main_mm_all.py` — synthetic Gaussians (any encoder backbone)
- `main_mm_all_vit.py`, `main_mm_all_vit_omniglot.py` — Omniglot with ViT
- `main_mm_all_vit_miniimagenet.py` — Mini-ImageNet with ViT
- `main_mm_vit_omniglot.py`, `main_mm_large_projector.py`,
  `main_mm_all_no_encoder_weights.py` — ablations

## Figure / Table Reproduction

| Result                                | Entry point                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| Fig. 2a-b (scaling sweeps)            | `main_stage1_larger_model.py`                                |
| Fig. 2c-d (positional encodings)      | `main_stage1.py` (toggle `rope`)                             |
| Fig. 4 (multimodal asymmetry)         | `main_stage2.py` / `main_mm_all*.py`                         |
| App. Mini-ImageNet validation         | `main_mm_all_vit_miniimagenet.py`                            |
| App. Encoder representation vs. opt.  | `main_mm_all.py` (with `freeze_encoder` / `load_encoder_ckpt`) |
| App. Joint-label setting              | `main_stage2_joint_label.py`                                 |
| App. Qwen2.5-VL-3B head identification | `mllm_head_analysis.py`, `llm_head_analysis.py`             |
| App. Qwen2.5-VL-3B knockout           | `mllm_knockout.py`                                           |
| App. Qwen2.5-VL-3B fine-tuning        | `mllm_finetune.py`                                           |

The `top_heads.json` and `top_heads_llm.json` files in this repo are the head
rankings produced by `mllm_head_analysis.py` and `llm_head_analysis.py` on
Qwen2.5-VL-3B-Instruct / Qwen2.5-3B; downstream scripts (`mllm_knockout.py`,
`mllm_finetune.py`) consume them via `--heads_json`.

## Citation

```bibtex
@inproceedings{huang2026dissecting,
  title     = {Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in Modern Transformers},
  author    = {Huang, Yiran and Roth, Karsten and Bouniot, Quentin and Xu, Wenjia and Akata, Zeynep},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```

## License

Released under the MIT License unless otherwise noted.
