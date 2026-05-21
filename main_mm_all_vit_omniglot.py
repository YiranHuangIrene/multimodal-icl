import os
import sys
import time
import wandb
import torch
import numpy as np
from dataset_img import OmniglotMMDataset,OmniglotDatasetArgs
from dataset import get_mm_img_label_class
from model import ModelArgs, MLLMTransformer
from vit import  ViTENcoderArgs, ViTEncoder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
WANDB = True

def accuracy(outputs, labels, flip_labels=False, L2=None):
    predictions = F.softmax(outputs, dim=-1)
    predictions_inds = torch.argmax(predictions, dim=-1)
    labels_inds = torch.argmax(labels, dim=-1)
    if flip_labels:
        labels_inds = (torch.argmax(labels.int(), dim=-1) + 1) % L2
    return (predictions_inds == labels_inds).float().mean()


def evaluate(model, data, flip_labels=False, device=None,L2=None):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        inputs_mm, inputs_2, labels = data
        inputs_mm = inputs_mm.to(device)
        inputs_2 = inputs_2.to(device)
        labels = labels.to(device)
        outputs = model(inputs_mm, inputs_2)
        acc = accuracy(outputs, labels, flip_labels=flip_labels, L2=L2)
        if flip_labels:
            labels_inds = torch.argmax(labels.int(), dim=-1)
            labels_inds = (labels_inds + 1) % L2
            labels = torch.zeros_like(labels)
            labels.scatter_(1, labels_inds.unsqueeze(1), 1)
        loss = loss_criterion(outputs, labels)
        
        return loss, acc
    
def evaluate_transformer(model, data, flip_labels=False, device=None,L2=None):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        inputs_mm, inputs_2, labels = data
        inputs_mm_1, inputs_mm_2 = torch.chunk(inputs_mm, 2, dim=0)
        inputs_2_1, inputs_2_2 = torch.chunk(inputs_2, 2, dim=0)
        labels_1, labels_2 = torch.chunk(labels, 2, dim=0)
        
        inputs_mm_1, inputs_mm_2 = inputs_mm_1.to(device), inputs_mm_2.to(device)
        inputs_2_1, inputs_2_2 = inputs_2_1.to(device), inputs_2_2.to(device)
        labels_1, labels_2 = labels_1.to(device), labels_2.to(device)
        outputs_1 = model(inputs_mm_1, inputs_2_1)
        outputs_2 = model(inputs_mm_2, inputs_2_2)
        acc_1 = accuracy(outputs_1, labels_1, flip_labels=flip_labels, L2=L2)
        acc_2 = accuracy(outputs_2, labels_2, flip_labels=flip_labels, L2=L2)
        acc = (acc_1 + acc_2) / 2
        if flip_labels:
            labels_inds = torch.argmax(labels.int(), dim=-1)
            labels_inds = (labels_inds + 1) % L2
            labels = torch.zeros_like(labels)
            labels.scatter_(1, labels_inds.unsqueeze(1), 1)
            labels_1, labels_2 = torch.chunk(labels, 2, dim=0)
            labels_1, labels_2 = labels_1.to(device), labels_2.to(device)
        loss_1 = loss_criterion(outputs_1, labels_1)
        loss_2 = loss_criterion(outputs_2, labels_2)
        loss = (loss_1 + loss_2) / 2
        
        return loss, acc

def train(model,train_loader, test_data,  test_ic_data, test_ic2_data, test_iw_data,optimizer, device, encoder,print_every, ckpt_store_freq, prefix, niters, n_epochs, save_ckpt):
    model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    global_iter = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for n, batch in tqdm(enumerate(train_loader), total=niters):
            model.train()
            inputs_mm, inputs_2, labels = batch
            inputs_mm = inputs_mm.to(device)
            inputs_2 = inputs_2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs_mm, inputs_2)
            loss = loss_criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            if save_ckpt:
                if global_iter%ckpt_store_freq==0 and global_iter!=0:
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)
                    if not os.path.exists(f"{prefix}/seed_{SEED}"):
                        os.makedirs(f"{prefix}/seed_{SEED}")
                    torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{global_iter}.pt")
                
            # Evaluate  
            if global_iter%print_every==0:
                if encoder == "transformer" or encoder == "cnn" or encoder == "vit":
                    loss_test, acc_test = evaluate_transformer(model, test_data, flip_labels=False, device=device)
                    loss_ic, acc_ic =  evaluate_transformer(model, test_ic_data, flip_labels=False, device=device)
                    loss_ic2, acc_ic2 = evaluate_transformer(model, test_ic2_data, flip_labels=True, device=device,L2=L2)
                    loss_iw, acc_iw = evaluate_transformer(model, test_iw_data, flip_labels=False, device=device)
                else:
                    loss_test, acc_test = evaluate(model, test_data, flip_labels=False, device=device)
                    loss_ic, acc_ic = evaluate(model, test_ic_data, flip_labels=False, device=device)
                    loss_ic2, acc_ic2 = evaluate(model, test_ic2_data, flip_labels=True, device=device,L2=L2)
                    loss_iw, acc_iw = evaluate(model, test_iw_data, flip_labels=False, device=device)
                print(f"Epoch {epoch}, Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}")
                if WANDB:
                    wandb.log({"Epoch": epoch, "Iteration": n, "global_iter": global_iter, "Train_Loss": loss, "Train_Accuracy": acc, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw})
            global_iter += 1

        
if __name__ == "__main__":
    # Set up CUDA and random seeds
    device = torch.device(f"cuda:{int(sys.argv[32])}" if torch.cuda.is_available() else "cpu")
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Data Parameters
    K1 = int(sys.argv[1])
    K2 = int(sys.argv[2])
    N = int(sys.argv[3])  # Number of item-label pairs in the context
    D1 = int(sys.argv[4])
    D2 = int(sys.argv[5])
    L1 = int(sys.argv[6])
    L2 = int(sys.argv[7])
    alpha0 = float(sys.argv[8])
    alpha1 = float(sys.argv[9])
    alpha2 = float(sys.argv[10])  # Zipf's law exponent
    B = int(sys.argv[11])  # Burstiness
    p_B = float(sys.argv[12])  # Fraction of bursty sequences
    p_C = float(sys.argv[13])  # Fraction of OOD sequences
    eps0 = float(sys.argv[14])  # Within-class variance for pretraining the encoder
    eps1 = float(sys.argv[15])
    eps2 = float(sys.argv[16])  # Within-class variance
    # Image dataset parameters
    n_img_per_class = int(sys.argv[17])
    augment = bool(int(sys.argv[18]))
    img_size = 105
    patch_size = 15
    no_repeats = False
    
    
    S = 512  # Number of sequences in the test set
    Nmax = 32  # Maximum number of item-label pairs in the context
    P1 = 1.0/(np.arange(1,K1+1)**alpha1)
    P1 /= np.sum(P1)  # Normalized power law distribution
    P2 = 1.0/(np.arange(1,K2+1)**alpha2)
    P2 /= np.sum(P2)  # Normalized power law distribution
    
    # Model Parameters
    n_heads = int(sys.argv[19])  # Number of attention heads
    n_layers = int(sys.argv[20])  # Number of transformer layers
    rope = bool(int(sys.argv[21]))  # Whether to use RoPE
    rope_theta = int(sys.argv[22])  # Rope base
    rms_norm = bool(int(sys.argv[23])) # Whether to use RMS normalization
    L_pos = int(sys.argv[24])
    encoder = sys.argv[25] # which encoder to use
    freeze_layers = bool(int(sys.argv[26]))
    freeze_encoder = bool(int(sys.argv[27]))
    ckpt_path = sys.argv[28]
    root = os.path.dirname(os.path.realpath(__file__))
    if eps0 == 0 and augment == False and alpha0 == 0.:
        iterations = 5000
    elif n_img_per_class == 8 and eps0 != 0:
        iterations = 8000
    else:
        iterations = 12000
    ckpt_path_enc = f"{root}/outs_encoder_vit_omniglot_old/K_{K2}_n_img_per_class_{n_img_per_class}_eps_{eps0}_alpha_{alpha0}_augment_{augment}_patch_size_{patch_size}_output_dim_{D2}_depth_2_heads_1_niter_{iterations}/seed_0/ckpt_{iterations-1}.pt"

    # Training parameters
    niters = 50000  # Number of iterations
    n_epochs = 1  # Number of epochs
    batch_size = int(sys.argv[29])
    lr = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    optimizer = sys.argv[30]
    print_every = 100  # Print every n iterations
    ckpt_store_freq = 5000 # Store every n iterations
    save_ckpt = bool(int(sys.argv[31]))
    

    if rope:
        input_dim = D1
    else:
        input_dim = L_pos + D1
        
   # Initialize wandb
    prefix = f"./outs_torch/Omniglot_K2_{K2}_n_img_{n_img_per_class}_eps_{eps0}_a_{alpha0}_aug_{augment}_out_{D2}_MM_K1_{K1}_L1_{L1}_L2_{L2}_a1_{alpha1}_a2_{alpha2}_B{B}_pB{p_B}_pC{p_C}_eps1_{eps1}_eps2_{eps2}_encoder_{encoder}_freeze_layers{freeze_layers}_freeze_encoder{freeze_encoder}"
    if WANDB:
        wandb.init(project="ICL_torch_omniglot",
                name=f"run_{SEED}_{prefix.split('/')[-1]}",
                config={
                    "K1": K1,
                    "K2": K2,
                    "n_img_per_class": n_img_per_class,
                    "eps0": eps0,
                    "alpha0": alpha0,
                    "augment": augment,
                    "output_dim": D2,
                    "N": N,
                    "D1": D1,
                    "D2": D2,
                    "L1": L1,
                    "L2": L2,
                    "S": S,
                    "Nmax": Nmax,
                    "alpha1": alpha1,
                    "alpha2": alpha2,
                    "B": B,
                    "pB": p_B,
                    "pC": p_C,
                    "eps1": eps1,
                    "eps2": eps2,
                    "no_repeats": no_repeats,
                    "rope": rope,
                    "rope_theta": rope_theta,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "rms_norm": rms_norm,
                    "niters": niters,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    "seed": SEED,
                    "n_epochs": n_epochs,
                    "L_pos": L_pos,
                    "encoder": encoder,
                    "freeze_layers": freeze_layers,
                    "freeze_encoder": freeze_encoder,
                    "ckpt_path": ckpt_path,
                    "ckpt_path_enc":ckpt_path_enc,
                    "test_img": "val_img",
                    "train_img": "all_img"
                })

    # Initialize model
    model_args_mm = ModelArgs(
        m1_dim=D1,
        m2_dim=D2,
        dim=input_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_labels=L1,
        max_position_embeddings=3*N+2,
        rope_theta=rope_theta,
        mlp_bias=True,
        rms_norm=rms_norm,
        rope=rope,
        norm_eps=1e-5,
        L_pos=L_pos
    )
    model_args_enc =  ViTENcoderArgs(
            num_classes=K2,
            dim=D2,
        )
    Encoder = ViTEncoder(model_args_enc)
    Encoder.load_state_dict(torch.load(ckpt_path_enc), strict=True)
    for p in Encoder.classifier.parameters():
        p.requires_grad = False
    model = MLLMTransformer(model_args_mm)
    model.init_encoder(Encoder)
    model.load_state_dict(torch.load(ckpt_path),strict=False)
    print("Model structure:")
    print(model)
    if freeze_layers:
        for p in model.layers.parameters():
            p.requires_grad = False
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    #Initialize optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize datasets
    print("Generating training data...")
    dataset_args = OmniglotDatasetArgs(root=root, K=K2, n_img_per_class=n_img_per_class, eps=eps0, alpha=alpha0, augment=augment)
    mus_label_m1, mus_class_m1, labels_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1 = get_mm_img_label_class(K1=K1,K2=K2,L1=L1,L2=L2,D1=D1)
    train_dataset = OmniglotMMDataset(dataset_args, mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=batch_size,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B=B, p_B = p_B, p_C = p_C, no_repeats = no_repeats, datasize=niters)
    train_loader = DataLoader(train_dataset, batch_size=None,num_workers=16)
    print("Generating test data...")
    test_data = train_dataset.generate_test_sequence(S=S,B=B, p_B = p_B, p_C = p_C)
    test_ic_data = train_dataset.generate_test_sequence(S=S,B=B, p_B = 1, p_C = 1)
    test_ic2_data = train_dataset.generate_test_sequence(S=S,B=B, p_B = 1, p_C = 0, flip_labels=True)
    test_iw_data = train_dataset.generate_test_sequence(S=S,B=0, p_B = 0, p_C = 0)  
    print("Training model...")
    train(model=model, train_loader=train_loader, test_data=test_data,  test_ic_data=test_ic_data, test_ic2_data=test_ic2_data, test_iw_data=test_iw_data, optimizer=optimizer, device=device, encoder=encoder,print_every=print_every, ckpt_store_freq=ckpt_store_freq, prefix=prefix, niters=niters, n_epochs=n_epochs, save_ckpt=save_ckpt)
    