import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import generate_encoder_input, EncoderDataset,get_mm_mus_label_class
from model import TransformerEncoderArgs, TransformerEncoder
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
    
    K = int(sys.argv[1])
    eps = float(sys.argv[2])
    feat_dim = int(sys.argv[3])
    input_dim = int(sys.argv[4])
    output_dim = int(sys.argv[5])
    num_layers= int(sys.argv[6])
    num_heads = int(sys.argv[7])
    niter = int(sys.argv[8])
    batch_size = int(sys.argv[9])
    save_ckpt = bool(int(sys.argv[10]))
    
    lr = 1e-3
    weight_decay = 1e-6 
    print_every = 100
    val_every = 100
    save_every = 10000
        
    prefix = f"./outs_encoder_transformer/K{K}_eps{eps}_feat_dim{feat_dim}_input_dim{input_dim}_output_dim{output_dim}_num_layers{num_layers}_num_heads{num_heads}_niter{niter}"
    if WANDB:
        import wandb
        wandb.init(project="ICL_encoder", 
                   name=f"run_{SEED}_{prefix.split('/')[-1]}",
                   config={
            "K": K,
            "eps": eps,
            "feat_dim": feat_dim,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "niter": niter,
            "encoder": "transformer"
        })
        
    model_args = TransformerEncoderArgs(
        feat_dim=feat_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        num_classes=K,
        num_layers=num_layers,
        num_heads=num_heads
    )
    model = TransformerEncoder(model_args)
    print(f"Model: {model}")
    K1=8192
    L1=L2=32
    D1=64
    _, m1, _, _, mus_class, _, _ = get_mm_mus_label_class(K1=K1,K2=K,L1=L1,L2=L2,D1=D1,D2=feat_dim)
    train_dataset = EncoderDataset(mus_class=mus_class, eps=eps, S=batch_size, datasize=niter)
    train_loader = DataLoader(train_dataset, batch_size=None,num_workers=1)
    val_dataset = generate_encoder_input(mus_class=mus_class, eps=eps, S=1000)
    model = train_model(model=model,dataloader=train_loader,val_loader=val_dataset,lr=lr,weight_decay=weight_decay,device=device,prefix=prefix,print_every=print_every,val_every=val_every,save_ckpt=save_ckpt,save_every=save_every)