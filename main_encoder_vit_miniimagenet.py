import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset_img import MiniImageNetDatasetArgs, MiniImageNetDataset
from vit import ViTENcoderArgs, ViTEncoder
import sys
import os

WANDB = os.environ.get("WANDB_ENABLED", "0") == "1"

_state = {"logfile": None}

def log(msg):
    print(msg, flush=True)
    if _state["logfile"]:
        _state["logfile"].write(msg + "\n")
        _state["logfile"].flush()


def fast_augment(imgs_t):
    """Batch-level augmentation using torch ops."""
    B = imgs_t.shape[0]
    flip_mask = torch.rand(B, device=imgs_t.device) > 0.5
    imgs_t[flip_mask] = imgs_t[flip_mask].flip(-1)
    pad = 8
    imgs_t = F.pad(imgs_t, [pad]*4, mode='reflect')
    top = torch.randint(0, 2*pad, (1,)).item()
    left = torch.randint(0, 2*pad, (1,)).item()
    imgs_t = imgs_t[:, :, top:top+84, left:left+84]
    return imgs_t


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (np.clip(cx - cut_w//2, 0, W), np.clip(cy - cut_h//2, 0, H),
            np.clip(cx + cut_w//2, 0, W), np.clip(cy + cut_h//2, 0, H))


def mixup_cutmix(x, y, mixup_alpha=0.8, cutmix_alpha=1.0):
    B = x.size(0)
    index = torch.randperm(B, device=x.device)
    if np.random.rand() < 0.5:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(2), x.size(3), lam)
        x_out = x.clone()
        x_out[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(lam, 1 - lam)
        x_out = lam * x + (1 - lam) * x[index]
    y_out = lam * y + (1 - lam) * y[index]
    return x_out, y_out


def get_lr(step, peak_lr, warmup_steps, total_steps, min_lr=1e-6):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))


def train_model(model, dataloader, val_loader=None, lr=5e-4, weight_decay=0.05,
                device=None, prefix=None, print_every=100, val_every=100,
                save_ckpt=True, save_every=1000, label_smoothing=0.1, K=64):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    warmup_steps = 1500
    best_val = 0.0

    for n, batch in enumerate(dataloader):
        cur_lr = get_lr(n, lr, warmup_steps, niter)
        for pg in opt.param_groups:
            pg['lr'] = cur_lr

        model.train()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = fast_augment(inputs)
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + label_smoothing / K
        inputs, labels = mixup_cutmix(inputs, labels)
        opt.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if n % print_every == 0:
            preds = logits.argmax(dim=1)
            label_indices = labels.argmax(dim=1)
            acc = (preds == label_indices).sum().item() / len(preds)
            log(f"Iter {n:>5} -- loss: {loss.item():.4f}, acc: {acc:.4f}, lr: {cur_lr:.2e}")
            if WANDB:
                wandb.log({"train_loss": loss.item(), "train_acc": acc, "lr": cur_lr})
        if val_loader is not None and n % val_every == 0:
            model.eval()
            acc_val_all = 0
            loss_val_all = 0
            n_batches = 0
            with torch.no_grad():
                for inputs_val, labels_val in val_loader:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    logits_val = model(inputs_val)
                    loss_val = criterion(logits_val, labels_val)
                    loss_val_all += loss_val.item()
                    preds_val = logits_val.argmax(dim=1)
                    label_indices_val = labels_val.argmax(dim=1)
                    acc_val = (preds_val == label_indices_val).sum().item() / len(preds_val)
                    acc_val_all += acc_val
                    n_batches += 1
                loss_val = loss_val_all / n_batches
                acc_val = acc_val_all / n_batches
                best_val = max(best_val, acc_val)
                log(f"Iter {n:>5} Val -- loss: {loss_val:.4f}, acc: {acc_val:.4f}, best: {best_val:.4f}")
                if WANDB:
                    wandb.log({"val_loss": loss_val, "val_acc": acc_val, "best_val_acc": best_val})
        if save_ckpt and (n == niter - 1 or (n > 0 and (n+1) % save_every == 0)):
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            if not os.path.exists(f"{prefix}/seed_{SEED}"):
                os.makedirs(f"{prefix}/seed_{SEED}")
            torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{n}.pt")
            log(f"Model saved at iteration {n}")
    return model

if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.dirname(os.path.abspath(__file__))
    K = int(sys.argv[1])
    n_img_per_class = int(sys.argv[2])
    eps = float(sys.argv[3])
    alpha = float(sys.argv[4])
    augment = bool(int(sys.argv[5]))
    rotate = int(sys.argv[6])
    flip_h = float(sys.argv[7])
    flip_v = float(sys.argv[8])
    crop = float(sys.argv[9])
    img_size = 84
    patch_size = int(sys.argv[10])
    output_dim = int(sys.argv[11])
    depth = int(sys.argv[12])
    heads = int(sys.argv[13])
    dropout = float(sys.argv[14])
    l1_lambda = float(sys.argv[15])
    l2_lambda = float(sys.argv[16])
    niter = int(sys.argv[17])
    batch_size = int(sys.argv[18])
    save_ckpt = bool(int(sys.argv[19]))
    lr = 5e-4
    weight_decay = 0.05
    print_every = 100
    val_every = 100
    save_every = niter - 1 if save_ckpt else 0

    prefix = f"./outs_encoder_vit_miniimagenet/K_{K}_eps_{eps}_alpha_{alpha}_augment_{augment}_patch_size_{patch_size}_output_dim_{output_dim}_depth_{depth}_heads_{heads}_dropout_{dropout}_niter_{niter}"
    if WANDB:
        import wandb
        wandb.init(project="ICL_encoder_miniimagenet",
                   name=f"{prefix.split('/')[-1]}_seed_{SEED}",
                   tags=["icml_rebuttal_miniimagenet"],
                   config={
                       "K": K, "eps": eps, "alpha": alpha, "augment": augment,
                       "patch_size": patch_size, "output_dim": output_dim,
                       "depth": depth, "heads": heads, "dropout": dropout,
                       "niter": niter, "batch_size": batch_size,
                       "lr": lr, "weight_decay": weight_decay,
                       "label_smoothing": 0.1,
                       "augmentation": "fast_augment+mixup+cutmix",
                   })

    _state["logfile"] = open(f"./encoder_train_{patch_size}_{output_dim}_{depth}_{heads}.log", "w")

    model_args = ViTENcoderArgs(
        image_size=img_size, patch_size=patch_size, channels=3,
        num_classes=K, dim=output_dim, depth=depth, heads=heads, dropout=dropout,
    )
    model = ViTEncoder(model_args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[Param count] Total: {total_params:,}  Trainable: {trainable_params:,}")
    if WANDB:
        wandb.config.update({"total_params": total_params, "trainable_params": trainable_params})

    dataset_args = MiniImageNetDatasetArgs(K=K, eps=eps, alpha=alpha, augment=augment)
    dataset = MiniImageNetDataset(dataset_args, S=batch_size, datasize=niter, n_eval_per_class=20)
    val_dataset = dataset.generate_val()
    val_dataset = TensorDataset(val_dataset[0], val_dataset[1])
    train_loader = DataLoader(dataset, batch_size=None, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=10*batch_size, num_workers=1)

    train_model(model=model, dataloader=train_loader, val_loader=val_dataloader,
                lr=lr, weight_decay=weight_decay, device=device, prefix=prefix,
                print_every=print_every, val_every=val_every, save_ckpt=save_ckpt,
                save_every=save_every, K=K)
