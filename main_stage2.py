import os
import sys
import time
import wandb
import torch
import numpy as np
from model import ModelArgs, Transformer, MMTransformer
from dataset import ICLDataset, MMDataset, get_mus_label_class, generate_input_seqs, generate_input_seqs_mm_v1,get_mm_mus_label_class
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

def compute_ih_strength(seq_labels,attn_weights):
    # Compute the attention difference the target paid to the correct labels  and in correct labels
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    unmatches = [(row[:-1] != row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][3*i+2] for i in m] for n,m in enumerate(matches)]
        attn_incorrect = [[layer_attn_weights[n][-1][3*i+2] for i in m] for n,m in enumerate(unmatches)]
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
        attn_correct = [[layer_attn_weights[n][-1][3*i+2] for i in m] for n,m in enumerate(matches)]
        score_correct= [sum(attn_correct[i])/len(attn_correct[i]) if len(attn_correct[i]) > 0 else 0  for i in range(len(matches))]
        score_correct = sum(score_correct)/len(score_correct)
        scores.append(score_correct)
    return scores

def compute_TIIA_m1(seq_labels,attn_weights):
    # Compute the attention the target paid to all items with the correct label
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][3*i] for i in m] for n,m in enumerate(matches)]
        score_correct= [sum(attn_correct[i])/len(attn_correct[i]) if len(attn_correct[i]) > 0 else 0  for i in range(len(matches))]
        score_correct = sum(score_correct)/len(score_correct)
        scores.append(score_correct)
    return scores

def compute_TIIA_m2(seq_labels,attn_weights):
    # Compute the attention the target paid to all items with the correct label
    matches = [(row[:-1] == row[-1]).nonzero(as_tuple=True)[0].tolist() for row in seq_labels]
    n_layers = len(attn_weights)
    scores = []
    for i in range(n_layers):
        layer_attn_weights = attn_weights[i].squeeze(1)
        attn_correct = [[layer_attn_weights[n][-1][3*i+1] for i in m] for n,m in enumerate(matches)]
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
        tla = [[layer_attn_weights[n][-1][3*i+2] for i in range(n_pairs)] for n in range(bsz)]
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

def compute_prev_2_attn(attn_weights):
    n_layers = len(attn_weights)
    prev_2_attns = []
    for i in range(n_layers):
        attn = attn_weights[i].squeeze(1)
        attn_to_prev = attn.diagonal(offset=-2, dim1=-2, dim2=-1)
        attn_to_prev = attn_to_prev.mean(dim=-1).mean(dim=-1)
        prev_2_attns.append(attn_to_prev)
    return prev_2_attns

def compute_prob_icl_labels(outputs, seq_labels):
    predictions = F.softmax(outputs, dim=-1)
    predictions_inds = torch.argmax(predictions, dim=-1)
    in_context = (predictions_inds.unsqueeze(1) == seq_labels[:,:-1]).any(dim=1)
    cla = in_context.float().mean().item()
    return cla

def evaluate(model, data, flip_labels=False, device=None, L2=None, progress_measure=False):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        if not progress_measure:
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
        else:
            inputs_mm, inputs_2, labels, seq_labels = data
            inputs_mm = inputs_mm.to(device)
            inputs_2 = inputs_2.to(device)
            labels = labels.to(device)
            seq_labels = seq_labels.to(device)
            outputs, attn_weights = model(inputs_mm, inputs_2, output_attn_weights=True)
            acc = accuracy(outputs, labels, flip_labels=flip_labels, L2=L2)
            if flip_labels:
                labels_inds = torch.argmax(labels.int(), dim=-1)
                labels_inds = (labels_inds + 1) % L2
                labels = torch.zeros_like(labels)
                labels.scatter_(1, labels_inds.unsqueeze(1), 1)
            loss = loss_criterion(outputs, labels)
            ih_strengths = compute_ih_strength(seq_labels, attn_weights)
            tiias_m1 = compute_TIIA_m1(seq_labels, attn_weights)
            tiias_m2 = compute_TIIA_m2(seq_labels, attn_weights)
            tilas = compute_TILA(seq_labels, attn_weights)
            tlas = compute_TLA(seq_labels, attn_weights)
            prev_1_attns = compute_prev_1_attn(attn_weights)
            prev_2_attns = compute_prev_2_attn(attn_weights)
            prob_icl_labels = compute_prob_icl_labels(outputs, seq_labels)
            return loss, acc, ih_strengths, tiias_m1, tiias_m2, tilas, tlas, prev_1_attns, prev_2_attns, prob_icl_labels

def train(model,train_loader, test_data,  test_ic_data, test_ic2_data, test_iw_data,optimizer, device, print_every, ckpt_store_freq, prefix, niters, n_epochs, save_ckpt, progress_measure):
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
                if global_iter%ckpt_store_freq==0 and global_iter!=0 or global_iter==niters-1:
                    if not os.path.exists(prefix):
                        os.makedirs(prefix)
                        if not os.path.exists(f"{prefix}/seed_{SEED}"):
                            os.makedirs(f"{prefix}/seed_{SEED}")
                    torch.save(model.state_dict(), f"{prefix}/seed_{SEED}/ckpt_{global_iter}.pt")
                    
            # Evaluate  
            if global_iter%print_every==0:
                loss_test, acc_test = evaluate(model, test_data, flip_labels=False, device=device)
                metric_ic = evaluate(model, test_ic_data, flip_labels=False, device=device, progress_measure=progress_measure)
                metric_ic2 = evaluate(model, test_ic2_data, flip_labels=True, device=device,L2=L2, progress_measure=progress_measure)
                loss_iw, acc_iw = evaluate(model, test_iw_data, flip_labels=False, device=device)
                if progress_measure:
                    loss_ic, acc_ic, ih_strengths_ic, tiias_m1_ic, tiias_m2_ic, tilas_ic, tlas_ic, prev_1_attns_ic, prev_2_attns_ic, prob_icl_labels_ic = metric_ic
                    loss_ic2, acc_ic2, ih_strengths_ic2, tiias_m1_ic2, tiias_m2_ic2, tilas_ic2, tlas_ic2, prev_1_attns_ic2, prev_2_attns_ic2, prob_icl_labels_ic2 = metric_ic2
                    print(f"Epoch {epoch}, Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}, layer1 IH strength: {ih_strengths_ic[0]:.4f}, layer2 IH strength: {ih_strengths_ic[1]:.4f}, layer1 TIIA m1: {tiias_m1_ic[0]:.4f}, layer2 TIIA m1: {tiias_m1_ic[1]:.4f}, layer1 TIIA m2: {tiias_m2_ic[0]:.4f}, layer2 TIIA m2: {tiias_m2_ic[1]:.4f}, layer1 TILA: {tilas_ic[0]:.4f}, layer2 TILA: {tilas_ic[1]:.4f}, layer1 TLA: {tlas_ic[0]:.4f}, layer2 TLA: {tlas_ic[1]:.4f}, layer1 prev_1_attn: {prev_1_attns_ic[0]:.4f}, layer2 prev_1_attn: {prev_1_attns_ic[1]:.4f}, layer1 prev_2_attn: {prev_2_attns_ic[0]:.4f}, layer2 prev_2_attn: {prev_2_attns_ic[1]:.4f}, prob_icl_labels: {prob_icl_labels_ic:.4f}")
                    if WANDB:
                        wandb.log({"Epoch": epoch, "Iteration": n, "global_iter": global_iter, "Train_Loss": loss, "Train_Accuracy": acc, 
                                   "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, 
                                   "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw, 
                                   "IH_strength_layer1": ih_strengths_ic[0], "IH_strength_layer2": ih_strengths_ic[1], 
                                   "TIIA_m1_layer1": tiias_m1_ic[0], "TIIA_m1_layer2": tiias_m1_ic[1], "TIIA_m2_layer1": tiias_m2_ic[0], "TIIA_m2_layer2": tiias_m2_ic[1],
                                   "TILA_layer1": tilas_ic[0], "TILA_layer2": tilas_ic[1], "TLA_layer1": tlas_ic[0], "TLA_layer2": tlas_ic[1], 
                                   "prev_1_attn_layer1": prev_1_attns_ic[0], "prev_1_attn_layer2": prev_1_attns_ic[1], "prev_2_attn_layer1": prev_2_attns_ic[0], "prev_2_attn_layer2": prev_2_attns_ic[1], 
                                   "prob_icl_labels": prob_icl_labels_ic})
                else:
                    loss_ic, acc_ic = metric_ic
                    loss_ic2, acc_ic2 = metric_ic2
                    print(f"Epoch {epoch}, Iteration {n}: Train loss: {loss:.4f}, Train acc: {acc:.4f}, Test loss: {loss_test:.4f}, Test acc: {acc_test:.4f}, IC loss: {loss_ic:.4f}, IC acc: {acc_ic:.4f}, IC2 loss: {loss_ic2:.4f}, IC2 acc: {acc_ic2:.4f}, IW loss: {loss_iw:.4f}, IW acc: {acc_iw:.4f}")
                    if WANDB:
                        wandb.log({"Epoch": epoch, "Iteration": n, "global_iter": global_iter, "Train_Loss": loss, "Train_Accuracy": acc, "Test_Loss": loss_test, "Test_Accuracy": acc_test, "IC_Loss": loss_ic, "IC_Accuracy": acc_ic, "IC2_Loss": loss_ic2, "IC2_Accuracy": acc_ic2, "IW_Loss": loss_iw, "IW_Accuracy": acc_iw})
            global_iter += 1

        
if __name__ == "__main__":
    # Set up CUDA and random seeds
    device = torch.device(f"cuda:{int(sys.argv[29])}" if torch.cuda.is_available() else "cpu")
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
    alpha1 = float(sys.argv[8])
    alpha2 = float(sys.argv[9])  # Zipf's law exponent
    B = int(sys.argv[10])  # Burstiness
    p_B = float(sys.argv[11])  # Fraction of bursty sequences
    p_C = float(sys.argv[12])  # Fraction of OOD sequences
    eps1 = float(sys.argv[13])
    eps2 = float(sys.argv[14])  # Within-class variance
    no_repeats = bool(int(sys.argv[15]))  # Whether repeated items are allowed in the context
    
    S = 1000  # Number of sequences in the test set
    Nmax = 32  # Maximum number of item-label pairs in the context
    P1 = 1.0/(np.arange(1,K1+1)**alpha1)
    P1 /= np.sum(P1)  # Normalized power law distribution
    P2 = 1.0/(np.arange(1,K2+1)**alpha2)
    P2 /= np.sum(P2)  # Normalized power law distribution
    
    # Model Parameters
    n_heads = int(sys.argv[16])  # Number of attention heads
    n_layers = int(sys.argv[17])  # Number of transformer layers
    rope = bool(int(sys.argv[18]))  # Whether to use RoPE
    rope_theta = int(sys.argv[19])  # Rope base
    rms_norm = bool(int(sys.argv[20])) # Whether to use RMS normalization
    L_pos = int(sys.argv[21])
    freeze_layers = bool(int(sys.argv[22]))
    ckpt_path = sys.argv[23]
    early_fusion = bool(int(sys.argv[24]))
    
    # Training parameters
    niters = 300000  # Number of iterations 80000 for normal 300000 for early fusion
    n_epochs = 1  # Number of epochs
    batch_size = int(sys.argv[25])
    lr = 1e-3  # Learning rate
    weight_decay = 1e-6  # Weight decay
    optimizer = sys.argv[26]
    print_every = 100  # Print every n iterations
    ckpt_store_freq = 40000 # Store every n iterations
    save_ckpt = bool(int(sys.argv[27]))
    progress_measure = bool(int(sys.argv[28]))
    if progress_measure:
        seq_labels = True
    else:
        seq_labels = False
    if rope:
        input_dim = D1
    else:
        input_dim = L_pos + D1
        
   # Initialize wandb
    prefix = f"./outs_torch/K1_{K1}_K2_{K2}_N{N}_D1_{D1}_D2_{D2}_L1_{L1}_L2_{L2}_alpha1_{alpha1}_alpha2_{alpha2}_B{B}_pB{p_B}_pC{p_C}_eps1_{eps1}_eps2_{eps2}_no_repeats{no_repeats}_rope_{rope}_rope_theta{rope_theta}_freeze_layers{freeze_layers}_early_fusion{early_fusion}_n_heads{n_heads}_n_layers{n_layers}_rms_norm{rms_norm}_optimizer{optimizer}_niters{niters}_n_epochs{n_epochs}"
    if WANDB:
        wandb.init(project="ICL_torch",
                name=f"run_{SEED}_{prefix.split('/')[-1]}",
                config={
                    "K1": K1,
                    "K2": K2,
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
                    "freeze_layers": freeze_layers,
                    "ckpt_path": ckpt_path,
                    "early_fusion": early_fusion,
                    "progress_measure": True,
                    "normalize_progress_measure": True,
                },
                tags=["early_fusion_rebuttal"]
                )

    # Initialize model
    model_args = ModelArgs(
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
    model = MMTransformer(model_args)
    print("Model structure:")
    print(model)
    # weights = "/shared-local/aoq609/MLLM_ICL/ICL/outs_torch/K8192_N8_D64_alpha0.0_B2_pB1.0_pC0.0_eps0.1_no_repeatsFalse_rope_theta10000_n_heads1_n_layers2_rms_normTrue_optimizerSGD_niters30000_n_epochs10/seed_0/ckpt_150000.pt"
    if not early_fusion:
        weights = ckpt_path
        model.load_state_dict(torch.load(weights),strict=False)
    if freeze_layers:
        for p in model.layers.parameters():
            p.requires_grad = False
    #Initialize optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize datasets
    mus_label_m1, mus_class_m1, labels_class_m1, mus_label_m2, mus_class_m2, labels_class_m2, mapping_m2_to_m1 = get_mm_mus_label_class(K1=K1,K2=K2,L1=L1,L2=L2,D1=D1,D2=D2)
    train_dataset = MMDataset(mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, mus_class_m2=mus_class_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=batch_size,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B=B, p_B = p_B, p_C = p_C, no_repeats = no_repeats, datasize=niters)
    train_loader = DataLoader(train_dataset, batch_size=None,num_workers=1)
    print("Generating test data...")
    test_data  = generate_input_seqs_mm_v1(mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, mus_class_m2=mus_class_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=S,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B=B, p_B = p_B, p_C = p_C, no_repeats = no_repeats)
    test_ic_data = generate_input_seqs_mm_v1(mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, mus_class_m2=mus_class_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=S,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats,seq_labels=seq_labels)
    test_ic2_data = generate_input_seqs_mm_v1(mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, mus_class_m2=mus_class_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=S,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats,seq_labels=seq_labels)
    test_iw_data = generate_input_seqs_mm_v1(mus_label_m1=mus_label_m1, mus_class_m1=mus_class_m1, mus_label_m2=mus_label_m2, mus_class_m2=mus_class_m2, labels_class_m2=labels_class_m2, mapping_m2_to_m1=mapping_m2_to_m1, N=N,S=S,eps1=eps1,eps2=eps2, P1 = P1, P2 = P2, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)
    
    print("Training model...")
    train(model=model, train_loader=train_loader, test_data=test_data,  test_ic_data=test_ic_data, test_ic2_data=test_ic2_data, test_iw_data=test_iw_data, optimizer=optimizer, device=device, print_every=print_every, ckpt_store_freq=ckpt_store_freq, prefix=prefix, niters=niters, n_epochs=n_epochs, save_ckpt=save_ckpt,progress_measure=progress_measure)
    