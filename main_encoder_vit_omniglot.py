import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset_img import OmniglotDatasetArgs, OmniglotDataset
from vit import  ViTENcoderArgs, ViTEncoder
import ast
import sys 
import os

WANDB = True
def train_model(model, dataloader, val_loader=None, lr=1e-3, weight_decay=1e-5,l1_lambda=0.0,
                l2_lambda=0.0, device=None, prefix=None, print_every=100, val_every=100, save_ckpt=True, save_every=1000):
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
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
        if l2_lambda > 0:
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
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
            acc_val_all = 0
            loss_val_all = 0
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
                loss_val = loss_val_all / len(val_loader)
                acc_val = acc_val_all / len(val_loader)
                print(f"Iter {n:>4} Validation — loss: {loss_val:.4f}, acc: {acc_val:.4f}")
                if WANDB:
                    wandb.log({"val_loss": loss_val, "val_acc": acc_val})
        if save_ckpt and n+1 % save_every == 0 and n > 0 or n == niter - 1 and save_ckpt:
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
    # Parameters for the dataset
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
    img_size = 105
    patch_size = int(sys.argv[10])
    # Parameters for the model
    output_dim = int(sys.argv[11])
    depth = int(sys.argv[12])
    heads = int(sys.argv[13])
    # Parameters for the training
    dropout = float(sys.argv[14])
    l1_lambda = float(sys.argv[15])
    l2_lambda = float(sys.argv[16])
    niter = int(sys.argv[17])
    batch_size = int(sys.argv[18])
    save_ckpt = bool(int(sys.argv[19]))
    lr = 1e-3
    weight_decay = 1e-5
    print_every = 100
    val_every = 100
    save_every = niter-1 if save_ckpt else 0
    
    prefix = f"./outs_encoder_vit_omniglot/K_{K}_n_img_per_class_{n_img_per_class}_eps_{eps}_alpha_{alpha}_augment_{augment}_rotate_{rotate}_flip_h_{flip_h}_flip_v_{flip_v}_crop_{crop}_patch_size_{patch_size}_output_dim_{output_dim}_depth_{depth}_heads_{heads}_dropout_{dropout}_l1_lambda_{l1_lambda}_l2_lambda_{l2_lambda}_niter_{niter}"
    if WANDB:
        import wandb
        wandb.init(project="ICL_encoder_omniglot", 
                   name=f"{prefix.split('/')[-1]}_seed_{SEED}",
                   config={
                       "K": K,
                       "n_img_per_class": n_img_per_class,
                       "eps": eps,
                       "alpha": alpha,
                       "augment": augment,
                       "flip_h": flip_h,
                        "flip_v": flip_v,
                        "rotate": rotate,
                        "crop": crop,
                       "patch_size": patch_size,
                       "output_dim": output_dim,
                       "depth": depth,
                       "heads": heads,
                       "dropout": dropout,
                       "l1_lambda": l1_lambda,
                        "l2_lambda": l2_lambda,
                       "niter": niter,
                       "batch_size": batch_size,
                       "lr": lr,
                       "weight_decay": weight_decay,
                   })
    # Initialize the model
    model_args = ViTENcoderArgs(
        image_size=img_size,
        patch_size=patch_size,
        channels=1,
        num_classes=K,
        dim=output_dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
    )
    model = ViTEncoder(model_args)
    print(f"Model: {model}")
    
    # Initialize the dataset
    dataset_args = OmniglotDatasetArgs(root=root, K=K, n_img_per_class=n_img_per_class, eps=eps, alpha=alpha, augment=augment,rotate=rotate, flip_h=flip_h, flip_v=flip_v, crop=crop)
    dataset = OmniglotDataset(dataset_args, S=batch_size, datasize=niter)
    val_dataset = dataset.generate_val()
    val_dataset = TensorDataset(val_dataset[0], val_dataset[1])
    train_loader = DataLoader(dataset, batch_size=None, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=10*batch_size, num_workers=1)
    # Initialize the training
    train_model(model=model, dataloader=train_loader, val_loader=val_dataloader, lr=lr, weight_decay=weight_decay, device=device, prefix=prefix, print_every=print_every, val_every=val_every, save_ckpt=save_ckpt, save_every=save_every,l1_lambda=l1_lambda, l2_lambda=l2_lambda)
    
    
    