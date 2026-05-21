import os
import sys
import time
import wandb
import torch
import numpy as np
from model import ModelArgs, Transformer
from dataset import ICLDataset, get_mus_label_class, generate_input_seqs
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

WANDB = True

def accuracy(outputs, labels, flip_labels=False):
    predictions = F.softmax(outputs, dim=-1)
    predictions_inds = torch.argmax(predictions, dim=-1)
    labels_inds = torch.argmax(labels, dim=-1)
    if flip_labels:
        labels_inds = (torch.argmax(labels, dim=-1) + 1) % labels.shape[-1]
    return (predictions_inds == labels_inds).float().mean()

def compute_ih_strength(seq_labels,attn_weights):
    # Compute the attention difference the target paid to the correct labels  and in correct labels
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    unmatches = [(row[:-1] != row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][2*i+1] for i in m] for n,m in enumerate(matches)]
        attn_incorrect = [[layer_attn_weights[n][-1][2*i+1] for i in m] for n,m in enumerate(unmatches)]
        score_correct= [sum(attn_correct[i])/len(attn_correct[i]) if len(attn_correct[i])>0 else 0 for i in range(len(matches))]
        score_incorrect= [sum(attn_incorrect[i])/len(attn_incorrect[i]) if len(attn_incorrect[i])>0 else 0  for i in range(len(unmatches))]
        ih_strength = [score_correct[i] - score_incorrect[i] for i in range(len(score_correct))]
        ih_strength = sum(ih_strength)/len(ih_strength)
        scores.append(ih_strength)
    return scores

def compute_TILA(seq_labels,attn_weights):
    # Compute the attention the target paid to all the correct labels in the context
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][2*i+1] for i in m] for n,m in enumerate(matches)]
        score_correct= [sum(attn_correct[i])/len(attn_correct[i]) if len(attn_correct[i]) > 0 else 0  for i in range(len(matches))]
        score_correct = sum(score_correct)/len(score_correct)
        scores.append(score_correct)
    return scores

def compute_TIIA(seq_labels,attn_weights):
    # Compute the attention the target paid to all items with the correct label
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][2*i] for i in m] for n,m in enumerate(matches)]
        score_correct= [sum(attn_correct[i])/len(attn_correct[i]) if len(attn_correct[i]) > 0 else 0  for i in range(len(matches))]
        score_correct = sum(score_correct)/len(score_correct)
        scores.append(score_correct)
    return scores
    
    
def compute_TLA(seq_labels,attn_weights):
    # Compute the attention the target paid to all the labels in the context
    n_layers = len(attn_weights)
    bsz,_,seq_len,_ = attn_weights[0].shape
    n_pairs = (seq_len - 1) // 3
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1) # assume there's only one head
        tla = [[layer_attn_weights[n][-1][2*i+1] for i in range(n_pairs)] for n in range(bsz)]
        tla = [sum(tla[i]) for i in range(bsz)]
        tla = sum(tla)/len(tla)
        scores.append(tla)
    return scores

def compute_prev_1_attn(attn_weights):
    n_layers = len(attn_weights)
    prev_1_attns = []
    for i in range(n_layers):
        attn = attn_weights[i].squeeze(1)
        attn_to_prev = attn.diagonal(offset=-1, dim1=-2, dim2=-1)
        attn_to_prev = attn_to_prev.mean(dim=-1).mean(dim=-1)
        prev_1_attns.append(attn_to_prev)
    return prev_1_attns

def compute_prob_icl_labels(outputs, seq_labels):
    predictions = F.softmax(outputs, dim=-1)
    predictions_inds = torch.argmax(predictions, dim=-1)
    in_context = (predictions_inds.unsqueeze(1) == seq_labels[:,:-1]).any(dim=1)
    cla = in_context.float().mean().item()
    return cla

def evaluate(model, data, flip_labels=False, device=None, progress_measure=False,L=None):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        if not progress_measure:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            acc = accuracy(outputs, labels, flip_labels=flip_labels)
            if flip_labels:
                labels_inds = torch.argmax(labels.int(), dim=-1)
                labels_inds = (labels_inds + 1) % L
                labels = torch.zeros_like(labels)
                labels.scatter_(1, labels_inds.unsqueeze(1), 1)
            loss = loss_criterion(outputs, labels)
            return loss, acc
        else:
            inputs, labels, seq_labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            seq_labels = seq_labels.to(device)
            outputs, attn_weights = model(inputs, output_attn_weights=True)
            acc = accuracy(outputs, labels, flip_labels=flip_labels)
            if flip_labels:
                labels_inds = torch.argmax(labels.int(), dim=-1)
                labels_inds = (labels_inds + 1) % L
                labels = torch.zeros_like(labels)
                labels.scatter_(1, labels_inds.unsqueeze(1), 1)
            loss = loss_criterion(outputs, labels)
            ih_strengths = compute_ih_strength(seq_labels, attn_weights)
            tiias = compute_TIIA(seq_labels, attn_weights)
            tilas = compute_TILA(seq_labels, attn_weights)
            tlas = compute_TLA(seq_labels, attn_weights)
            prev_1_attns = compute_prev_1_attn(attn_weights)
            prob_icl_labels = compute_prob_icl_labels(outputs, seq_labels)
            return loss, acc, ih_strengths, tiias, tilas, tlas, prev_1_attns, prob_icl_labels


def train(model, train_loader, test_loader, test_ic_loader, test_ic2_loader, test_iw_loader, optimizer, device, print_every, save_ckpt, ckpt_store_freq, prefix, niters, n_epochs, progress_measure):
    model.to(device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    for n, batch in tqdm(enumerate(train_loader), total=niters):
        model.train()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        if save_ckpt:
            if n%ckpt_store_freq==0 and n!=0 or n==niters-1:
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                if not os.path.exists(f"{prefix}/seed_{SEED}"):
                    os.makedirs(f"{prefix}/seed_{SEED}")
                torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{n}.pt")
            
        # Evaluate  
        if n%print_every==0:
            loss_test, acc_test = evaluate(model, test_loader, flip_labels=False, device=device)
            metric_ic = evaluate(model, test_ic_loader, flip_labels=False, device=device, progress_measure=progress_measure)
            metric_ic2 = evaluate(model, test_ic2_loader, flip_labels=True, device=device, progress_measure=progress_measure, L=L)
            loss_iw, acc_iw = evaluate(model, test_iw_loader, flip_labels=False, device=device)
            if progress_measure:
                loss_ic, acc_ic, ih_strengths_ic, tiias_ic, tilas_ic, tlas_ic, prev_1_attns_ic, prob_icl_labels_ic = metric_ic
                loss_ic2, acc_ic2, ih_strengths_ic2, tiias_ic2, tilas_ic2, tlas_ic2, prev_1_attns_ic2, prob_icl_labels_ic2 = metric_ic2
                print(f"Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}, layer1 IH strength: {ih_strengths_ic[0]:.4f}, layer2 IH strength: {ih_strengths_ic[1]:.4f},layer1 TIIA: {tiias_ic[0]:.4f},layer2 TIIA: {tiias_ic[1]:.4f}, layer1 TILA: {tilas_ic[0]:.4f}, layer2 TILA: {tilas_ic[1]:.4f}, layer1 TLA: {tlas_ic[0]:.4f}, layer2 TLA: {tlas_ic[1]:.4f}, layer1 prev_1_attn: {prev_1_attns_ic[0]:.4f}, layer2 prev_1_attn: {prev_1_attns_ic[1]:.4f}, prob_icl_labels: {prob_icl_labels_ic:.4f}")
                if WANDB:
                    wandb.log({"Iteration": n,  "Train_Loss": loss, "Train_Accuracy": acc, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw, "IH_strength_layer1": ih_strengths_ic[0], "IH_strength_layer2": ih_strengths_ic[1], "TIIA_layer1": tiias_ic[0], "TIIA_layer2": tiias_ic[1],"TILA_layer1": tilas_ic[0], "TILA_layer2": tilas_ic[1], "TLA_layer1": tlas_ic[0], "TLA_layer2": tlas_ic[1], "prev_1_attn_layer1": prev_1_attns_ic[0], "prev_1_attn_layer2": prev_1_attns_ic[1], "prob_icl_labels": prob_icl_labels_ic})
            else:
                loss_ic, acc_ic = metric_ic
                loss_ic2, acc_ic2 = metric_ic2
                print(f" Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}")
                if WANDB:
                    wandb.log({"Iteration": n, "Train_Loss": loss, "Train_Accuracy": acc, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw})

        
if __name__ == "__main__":
    # Set up CUDA and random seeds
    # print all received arguments
    print(f"Received arguments: {sys.argv}")
    device = torch.device(f"cuda:{int(sys.argv[22])}" if torch.cuda.is_available() else "cpu")
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Data Parameters
    K = int(sys.argv[1])  # Number of classes
    N = int(sys.argv[2])  # Number of item-label pairs in the context
    D = int(sys.argv[3])  # Feature dimension
    L = int(sys.argv[4])  # Number of labels
    alpha = float(sys.argv[5])  # Zipf's law exponent
    B = int(sys.argv[6])  # Burstiness
    p_B = float(sys.argv[7])  # Fraction of bursty sequences
    p_C = float(sys.argv[8])  # Fraction of OOD sequences
    eps = float(sys.argv[9])  # Within-class variance
    no_repeats = bool(int(sys.argv[10]))  # Whether repeated items are allowed in the context
    
    S = 1000  # Number of sequences in the test set
    Nmax = 32  # Maximum number of item-label pairs in the context
    P = 1.0/(np.arange(1,K+1)**alpha)
    P /= np.sum(P)  # Normalized power law distribution

    # Model Parameters
    n_heads = int(sys.argv[11])  # Number of attention heads
    n_layers = int(sys.argv[12])  # Number of transformer layers
    rope = bool(int(sys.argv[13]))  # Whether to use RoPE
    rope_theta = int(sys.argv[14])  # Rope base
    alibi = bool(int(sys.argv[15]))
    hybrid = bool(int(sys.argv[16]))
    rms_norm = bool(int(sys.argv[17])) # Whether to use RMS normalization

    # Training parameters
    niters = 300000  # Number of iterations
    n_epochs = 1
    batch_size = int(sys.argv[18])
    lr = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    optimizer = sys.argv[19]
    print_every = 1000  # Print every n iterations
    ckpt_store_freq = 100000 # Store every n iterations
    save_ckpt = bool(int(sys.argv[20]))
    progress_measure = bool(int(sys.argv[21]))
    if progress_measure:
        seq_labels = True
    else:
        seq_labels = False
    
    if hybrid:
        input_dim = 2*Nmax + D + 2 
    elif rope or alibi:
        input_dim = D
    else:
        input_dim = 2*Nmax + 1 + D    
    
    # Initialize wandb
    prefix = f"./outs_torch_rebuttal/K{K}_N{N}_D{D}_alpha{alpha}_B{B}_pB{p_B}_pC{p_C}_eps{eps}_no_repeats{no_repeats}_rope{rope}_rope_theta{rope_theta}_alibi{alibi}_hybrid{hybrid}_n_heads{n_heads}_n_layers{n_layers}_rms_norm{rms_norm}_optimizer{optimizer}_niters{niters}_progress_mesure"
    if WANDB:
        wandb.init(project="ICL_torch",
                name=f"run_{SEED}_{prefix.split('/')[-1]}",
                config={
                    "K": K,
                    "N": N,
                    "D": D,
                    "L": L,
                    "S": S,
                    "Nmax": Nmax,
                    "alpha": alpha,
                    "B": B,
                    "pB": p_B,
                    "pC": p_C,
                    "eps": eps,
                    "no_repeats": no_repeats,
                    "rope": rope,
                    "rope_theta": rope_theta,
                    "alibi": alibi,
                    "hybrid": hybrid,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "rms_norm": rms_norm,
                    "niters": niters,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    "seed": SEED,
                    "progress_measure": progress_measure,
                    "normalize_progress_measure": True
                },
                tags=["rebuttal"]
                )

    if hybrid:
        max_position_embeddings = input_dim
    else:
        max_position_embeddings = 2*N+1
    # Initialize model
    model_args = ModelArgs(
        dim=input_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_labels=L,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        mlp_bias=True,
        rms_norm=rms_norm,
        rope=rope,
        norm_eps=1e-5,
        use_alibi=alibi,
        use_hybrid=hybrid
    )
    model = Transformer(model_args)
    print("Model structure:")
    print(model)

    # Initialize optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize datasets
    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)
    train_dataset = ICLDataset(
        mus_label=mus_label, mus_class=mus_class, labels_class=labels_class,
        N=N, S=batch_size, Nmax=Nmax,
        eps=eps, B=B, p_B=p_B, p_C=p_C, P=P, datasize=niters,
        no_repeats=no_repeats, rope=(rope or alibi), hybrid=hybrid
    )

    # test_dataset = ICLDataset(
    #     mus_label=mus_label, mus_class=mus_class, labels_class=labels_class,
    #     N=N, S=S, Nmax=Nmax,
    #     eps=eps, B=B, p_B=p_B, p_C=p_C, P=P, datasize=1,
    #     no_repeats=no_repeats, rope=rope
    # )
    # test_dataset = [sample for sample in test_dataset]

    # test_ic_dataset = ICLDataset(
    #     mus_label=mus_label, mus_class=mus_class, labels_class=labels_class,
    #     N=N, S=S, Nmax=Nmax,
    #     eps=eps, B=B, p_B=1, p_C=1, P=P, datasize=1,        
    #     no_repeats=no_repeats, rope=rope
    # )
    # test_ic_dataset = [sample for sample in test_ic_dataset]
    # test_ic2_dataset = ICLDataset(
    #     mus_label=mus_label, mus_class=mus_class, labels_class=labels_class,
    #     N=N, S=S, Nmax=Nmax,
    #     eps=eps, B=B, p_B=1, p_C=0, P=P, datasize=1,        
    #     flip_labels=True, no_repeats=no_repeats, rope=rope
    # )
    # test_ic2_dataset = [sample for sample in test_ic2_dataset]
    # test_iw_dataset = ICLDataset(
    #     mus_label=mus_label, mus_class=mus_class, labels_class=labels_class,
    #     N=N, S=S, Nmax=Nmax,
    #     eps=eps, B=0, p_B=0, p_C=0, P=P, datasize=1,        
    #     no_repeats=no_repeats, rope=rope
    # )
    # test_iw_dataset = [sample for sample in test_iw_dataset]
    
    # Load the datasets with dataloader
    train_loader = DataLoader(train_dataset, batch_size=None,num_workers=1)
    test_data  = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = p_B, p_C = p_C, no_repeats = no_repeats, rope=(rope or alibi),hybrid=hybrid)
    test_ic_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats, rope=(rope or alibi),seq_labels=seq_labels,hybrid=hybrid)
    test_ic2_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats, rope=(rope or alibi),seq_labels=seq_labels,hybrid=hybrid)
    test_iw_data = generate_input_seqs(mus_label,mus_class,labels_class,N,S, Nmax,eps = eps, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats, rope=(rope or alibi),hybrid=hybrid)

    
    train(model=model, train_loader=train_loader, test_loader=test_data, test_ic_loader=test_ic_data, test_ic2_loader=test_ic2_data, test_iw_loader=test_iw_data, optimizer=optimizer, device=device, print_every=print_every, save_ckpt=save_ckpt, ckpt_store_freq=ckpt_store_freq, prefix=prefix, niters=niters, n_epochs=n_epochs, progress_measure=progress_measure)