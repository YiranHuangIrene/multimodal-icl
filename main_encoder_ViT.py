import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import SingleOmniglotDataset,FullOmniglotDataset, CIFAR100Dataset
from vit import  ViTENcoderArgs, ViTEncoder
import ast
import sys 
import os

WANDB = True

def train_model(model, dataloader, val_loader=None, lr=1e-3, weight_decay=1e-5, device=None, prefix=None, print_every=100, val_every=100, save_ckpt=True, save_every=1000):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    for n, batch in enumerate(dataloader):
        model.train()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        preds = logits.argmax(dim=1)
        label_indices = labels.argmax(dim=1)
        acc = (preds == label_indices).sum().item() / len(preds)
        if n % print_every == 0:
            print(f"Iter {n:>4} — Training - loss: {loss.item():.4f}, acc: {acc:.4f}")
            if WANDB:
                wandb.log({"train_loss": loss.item(), "train_acc": acc})
        if val_loader is not None and n % val_every == 0:
            model.eval()
            with torch.no_grad():
                inputs_val, labels_val = val_loader[0], val_loader[1]
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                logits_val = model(inputs_val)
                loss_val = criterion(logits_val, labels_val)
                preds_val = logits_val.argmax(dim=1)
                label_indices_val = labels_val.argmax(dim=1)
                acc_val = (preds_val == label_indices_val).sum().item() / len(preds_val)
                print(f"Iter {n:>4} Validation — loss: {loss_val.item():.4f}, acc: {acc_val:.4f}")
                if WANDB:
                    wandb.log({"val_loss": loss_val.item(), "val_acc": acc_val})
        if save_ckpt and n % save_every == 0 and n > 0 or n == niter - 1 and save_ckpt:
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            if not os.path.exists(f"{prefix}/seed_{SEED}"):
                os.makedirs(f"{prefix}/seed_{SEED}")
            torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{n}.pt")
            print(f"Model saved at iteration {n}")
    return model

if __name__ == "__main__":
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    datasetname = sys.argv[1]
    K = int(sys.argv[2])
    sample_method = sys.argv[3] 
    eps = float(sys.argv[4])
    image_size = int(sys.argv[5])
    patch_size = int(sys.argv[6])
    output_dim = int(sys.argv[7])
    depth = int(sys.argv[8])
    heads = int(sys.argv[9])
    niter = int(sys.argv[10])
    batch_size = int(sys.argv[11])
    save_ckpt = bool(int(sys.argv[12]))
    lr = 1e-3
    weight_decay = 1e-6 
    print_every = 500
    val_every = 500
    save_every = niter
    prefix = f"./outs_encoder_vit/dataset_{datasetname}_K{K}_sample{sample_method}_eps{eps}_output_dim{output_dim}_depth{depth}_heads{heads}_niter{niter}"
    if WANDB:
            import wandb
            wandb.init(project="ICL_encoder", 
                    name=f"run_{SEED}_{prefix.split('/')[-1]}",
                    config={
                "datasetname": datasetname,
                "K": K,
                "sample_method": sample_method,
                "eps": eps,
                "image_size": image_size,
                "patch_size": patch_size,
                "output_dim": output_dim,
                "depth": depth,
                "heads": heads,
                "niter": niter,
                "encoder": "vit"
            })
    model_args = ViTENcoderArgs(
        image_size=image_size,
        patch_size=patch_size,
        channels=1 if datasetname == "omniglot" else 3,
        num_classes=K,
        dim=output_dim,
        depth=depth,
        heads=heads,
    )
    model = ViTEncoder(model_args)
    print(f"Model: {model}")
    root = os.path.dirname(os.path.abspath(__file__))
    if datasetname == "omniglot":
        if sample_method == "full":
            dataset = FullOmniglotDataset(root=root, K=K, eps=eps, S=batch_size, datasize=niter)
            val_dataset = dataset.generate_val()
        elif sample_method == "single":
            dataset = SingleOmniglotDataset(root=root, K=K, eps=eps, S=batch_size, datasize=niter)
            val_dataset = dataset.generate_input()
    elif datasetname == "cifar100":
        dataset = CIFAR100Dataset(root=root, K=K, eps=eps, S=batch_size, datasize=niter)
        val_dataset = dataset.generate_val()
    
    train_loader = DataLoader(dataset, batch_size=None, num_workers=8)
    model = train_model(model, train_loader, val_loader=val_dataset, lr=lr, weight_decay=weight_decay, device=device, prefix=prefix, print_every=print_every, val_every=val_every, save_ckpt=save_ckpt, save_every=save_every)