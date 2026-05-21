import numpy as np
import math
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.datasets import Omniglot, CIFAR100

def get_mus_label_class(K, L, D):
    """Generate label and class embeddings"""
    mus_label = np.random.normal(size=(L, D))/np.sqrt(D)
    mus_class = np.random.normal(size=(K, D))/np.sqrt(D)
    
    if K < L or K%L != 0:
        raise ValueError("K > L and K%L == 0 is required")
        
    labels_class = np.tile(np.arange(L), int(K/L))
    
    return mus_label, mus_class, labels_class

def get_mm_mus_label_class(K1,K2,L1,L2,D1,D2):
    """
    Generate label and class embeddings for two distributions
    K1: number of classes for distribution 1
    K2: number of classes for distribution 2
    L1: number of labels for distribution 1
    L2: number of labels for distribution 2 (The labels for distribution 2 is part of the labels for distribution 1)
    D1: dimension of the label embeddings for distribution 1
    D2: dimension of the label embeddings for distribution 2
    """
    mus_label_m1 = np.random.normal(size=(L1, D1))/np.sqrt(D1)
    mus_label_m2 = mus_label_m1[:L2] # The labels for distribution 2 is part of the labels for distribution 1
    mus_class_m1 = np.random.normal(size=(K1, D1))/np.sqrt(D1)
    mus_class_m2 = np.random.normal(size=(K2, D2))/np.sqrt(D2)
    
    labels_class_m1 = np.tile(np.arange(L1), int(K1/L1))
    labels_class_m2 = np.tile(np.arange(L2), int(K2/L2))
    
    # Create mapping from distribution 2 to distribution 1 classes with same labels
    mapping_m2_to_m1 = []
    for k2 in range(K2):
        label2 = labels_class_m2[k2]
        # Find all classes in distribution 1 with matching label
        matching_classes = np.where(labels_class_m1 == label2)[0]
        # Store list of all matching class indices from distribution 1
        mapping_m2_to_m1.append(matching_classes)
    mapping_m2_to_m1 = np.array(mapping_m2_to_m1)
    return mus_label_m1, mus_class_m1, labels_class_m1, mus_label_m2, mus_class_m2, labels_class_m2, mapping_m2_to_m1

def get_mm_img_label_class(K1,K2,L1,L2,D1):
    """
    Generate label and class embeddings for two distributions
    K1: number of classes for distribution 1
    K2: number of classes for distribution 2
    L1: number of labels for distribution 1
    L2: number of labels for distribution 2 (The labels for distribution 2 is part of the labels for distribution 1)
    D1: dimension of the label embeddings for distribution 1
    """
    mus_label_m1 = np.random.normal(size=(L1, D1))/np.sqrt(D1)
    mus_label_m2 = mus_label_m1[:L2] # The labels for distribution 2 is part of the labels for distribution 1
    mus_class_m1 = np.random.normal(size=(K1, D1))/np.sqrt(D1)
    
    labels_class_m1 = np.tile(np.arange(L1), int(K1/L1))
    labels_class_m2 = np.tile(np.arange(L2), int(K2/L2))
    
    # Create mapping from distribution 2 to distribution 1 classes with same labels
    mapping_m2_to_m1 = []
    for k2 in range(K2):
        label2 = labels_class_m2[k2]
        # Find all classes in distribution 1 with matching label
        matching_classes = np.where(labels_class_m1 == label2)[0]
        # Store list of all matching class indices from distribution 1
        mapping_m2_to_m1.append(matching_classes)
    mapping_m2_to_m1 = np.array(mapping_m2_to_m1)
    return mus_label_m1, mus_class_m1, labels_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1

def generate_encoder_input(mus_class,eps,S):
    # Generate input data for the encoder
    # mus_class: (K,D) the class embeddings for the K classes
    # eps: the within-class variance
    # Returns:
    # inputs: (S,D) the input data for the encoder
    # labels: (S) the labels for the input data
    K = mus_class.shape[0]
    D = mus_class.shape[1]
    inputs = np.zeros((S,D))
    e_fac = 1/np.sqrt(1+eps**2)
    choices = np.random.choice(K, size = (S,))
    inputs = e_fac*(mus_class[choices] + eps*np.random.normal(size = (S,D))/np.sqrt(D))
    labels = np.zeros((S,K),dtype=bool)
    labels[np.arange(S),choices] = True
    return torch.FloatTensor(inputs), torch.FloatTensor(labels)

def generate_input_seqs(mus_label, mus_class, labels_class, N,S, Nmax, eps= 0.1, B = 0, p_B = 0, P = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, rope = True, hybrid = False):
    e_fac = 1/np.sqrt(1+eps**2)
    
    L = mus_label.shape[0]
    K = mus_class.shape[0]
    D = mus_label.shape[1]


    K_c = 128
    mus_class_new = np.random.normal(size = (K_c,D))/np.sqrt(D)

    if K_c < L or K_c%L != 0:
        print("K > L and K%L == 0 is required")
        return 0
    labels_class_new =  np.tile(np.arange(L),int(K_c/L))
    
    if hybrid:
        input_dim = 2*Nmax + 2 + D
    elif rope:
        input_dim = D
    else:
        input_dim = 2*Nmax+1 + D
    inputs = np.zeros((S,2*N+1,input_dim))

    if P is None or len(P) != K:
        P = np.ones(K)/K

    #N has to be divisible by B as we will have N/B copies of each label in the context. 
    if (B > 0 and N%B != 0) or B >= N:
        print("N is not divisible by B or N/B is not even or B >= N")
        return 0,0

    if B == 0:
        B = int(N/2)
        p_B = 0

    choices = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K/L)), size = (int(N/B)))
            choices[s] = pos_choices*L + label_choices
    else:
        choices = np.random.choice(np.arange(K), size = (S,int(N/B)), p=P)
    choices = np.tile(choices,B)
    [np.random.shuffle(x) for x in choices]

    choices_c = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K_c/L)), size = (int(N/B)))
            choices_c[s] = pos_choices*L + label_choices
            #print(choices_c[s], labels_class_new[choices_c[s]],label_choices, pos_choices)
    else:
        choices_c = np.random.choice(np.arange(K_c), size = (S,int(N/B)))
    choices_c = np.tile(choices_c,B)
    [np.random.shuffle(x) for x in choices_c]

    targets_ind = np.random.choice(choices.shape[1], size = (choices.shape[0],)) # select the target from the context
    targets = choices[np.arange(choices.shape[0]),targets_ind]

    targets_c_ind = np.random.choice(choices_c.shape[1], size = (choices_c.shape[0],))
    targets_c = choices_c[np.arange(choices_c.shape[0]),targets_c_ind]

    filt_B = np.random.uniform(size = S) > p_B

    choices[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),N), p = P)
    targets[filt_B] = np.random.choice(K,size  = (np.sum(filt_B),), p = P)

    filt_C = np.random.uniform(size = S) > p_C

    #print(np.arange(S)[~filt_C])
    #print(np.arange(S)[~filt_B])
    if hybrid:
        start_ind = 2*Nmax + 2
    elif rope:
        start_ind = 0
    else:
        start_ind = 2*Nmax+1
    inputs[filt_C,:-1:2,start_ind:] = (e_fac*(mus_class[choices] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[filt_C]  # Context item embeddings sampled from the bursty sequences
    
    if flip_labels:
        wrong_label = (labels_class + 1)%L
        inputs[filt_C,1:-1:2,start_ind:] = ((mus_label[wrong_label])[choices])[filt_C]
    else:
        inputs[filt_C,1:-1:2,start_ind:] = ((mus_label[labels_class])[choices])[filt_C]  # Label item embeddings sampled from the bursty sequences

    inputs[filt_C,-1,start_ind:] = ((e_fac*(mus_class[targets] + eps*np.random.normal(size = (S,D))/np.sqrt(D))))[filt_C]  # Target item embedding sampled from the bursty sequences

    inputs[~filt_C,:-1:2,start_ind:] = (e_fac*(mus_class_new[choices_c] + eps*np.random.normal(size = (S,N,D))/np.sqrt(D)))[~filt_C] # Context item embeddings sampled from the OOD sequences
    inputs[~filt_C,1:-1:2,start_ind:] = ((mus_label[labels_class_new])[choices_c])[~filt_C]  # Label item embeddings sampled from the OOD sequences
    inputs[~filt_C,-1,start_ind:] = (e_fac*(mus_class_new[targets_c] + eps*np.random.normal(size = (S,D))/np.sqrt(D)))[~filt_C]   # Target item embedding sampled from the OOD sequences

    shifts = np.random.choice((2*Nmax + 1) - (2*N + 1) + 1, size = (S))
    
    labels = np.zeros((S,L),dtype= bool)
    target_classes = np.zeros(S, dtype = int)
    label_sequences = np.zeros((S, N+1), dtype=int)

    for s in range(S):
        if filt_C[s]:
            labels[s,labels_class[targets[s]]] = True
            target_classes[s]= targets[s]
            label_sequences[s] = np.append(labels_class[choices[s]],labels_class[targets[s]])
             
        else:
            labels[s,labels_class_new[targets_c[s]]] = True
            target_classes[s] = -1
            label_sequences[s] = np.append(labels_class_new[choices_c[s]],labels_class_new[targets_c[s]])
        if hybrid:
            inputs[s,:,shifts[s]:shifts[s] + 2*N+1] = np.identity(2*N+1)
        elif not rope:
            inputs[s,:,shifts[s]:shifts[s] + 2*N+1] = np.identity(2*N+1)

    if seq_labels:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
    elif output_target_labels:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.LongTensor(target_classes)
    else:
        return torch.FloatTensor(inputs), torch.FloatTensor(labels)
    
def generate_input_seqs_mm_v1(mus_label_m1, mus_class_m1, mus_label_m2, mus_class_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False):
    # Multimodal sequence layout (length 3N+2):
    #   [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
    # Position 3N (= -2) holds the M1 query embedding; position 3N+1 (= -1)
    # is reserved for the projected M2 query (filled in combine_mm_input_seqs_v1).
    L1 = mus_label_m1.shape[0]
    L2 = mus_label_m2.shape[0]
    K1 = mus_class_m1.shape[0]
    K2 = mus_class_m2.shape[0]
    D1 = mus_class_m1.shape[1]
    D2 = mus_class_m2.shape[1]

    inputs_mm = np.zeros((S,3*N+2,D1))
    inputs_2 = np.zeros((S,N+1,D2))
    
    # check parameters
    if P1 is None or len(P1) != K1:
        P1 = np.ones(K1)/K1
    if P2 is None or len(P2) != K2:
        P2 = np.ones(K2)/K2
    if (B > 0 and N%B != 0) or B >= N:
        print("N is not divisible by B or N/B is not even or B >= N")
        return 0
    if B == 0:
        B = int(N/2)
        p_B = 0
    
    # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
    K_c1 = 128  
    K_c2 = int(K_c1 * K2 / K1)
    K_c2 = K_c2 // L2 * L2
    if K_c2 < L2:
        K_c2 = L2
    mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
    mus_class_new2 = np.random.normal(size = (K_c2,D2))/np.sqrt(D2)
    if K_c1 < L1 or K_c1%L1 != 0:
        print("K_c1 > L1 and K_c1%L1 == 0 is required")
        return 0
    if K_c2 < L2 or K_c2%L2 != 0:
        print("K_c2 > L2 and K_c2%L2 == 0 is required")
        return 0
    labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
    labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
    mapping_m2_to_m1_new = []
    for k2 in range(K_c2):
        label2 = labels_class_new2[k2]
        matching_classes = np.where(labels_class_new1 == label2)[0]
        mapping_m2_to_m1_new.append(matching_classes)
    mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
    # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
    choices_2 = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
            choices_2[s] = pos_choices*L2 + label_choices
    else:
        choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
    choices_1 = np.zeros((S,int(N/B)), dtype = int)
    for s in range(S):
        choices = []
        for k in choices_2[s]:
            Pk=P1[mapping_m2_to_m1[k]]
            Pk/=np.sum(Pk)
            choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
        choices_1[s] = np.array(choices)
    # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
    choices = np.zeros((S, 2*int(N/B)), dtype=int)
    choices[:,0::2] = choices_1  # Even indices get choices_1 
    choices[:,1::2] = choices_2  # Odd indices get choices_2

    choices = np.tile(choices,B)
    # Shuffle choices pairwise (keeping x,y pairs together)
    for s in range(S):
        # Reshape into pairs, shuffle pairs, then flatten back
        pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
        np.random.shuffle(pairs)  # Shuffle along first dimension
        choices[s] = pairs.flatten()  # Back to 1D array
    choices_1 = choices[:,0::2]
    choices_2 = choices[:,1::2]
    
    # Select target from distribution 2
    targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
    targets = choices_2[np.arange(S),targets_ind]
    # Paired M1 target (sampled from the M1 classes that share the M2 target's label)
    targets_m1 = choices_1[np.arange(S), targets_ind]

    # Choose OOD context items classes for distribution 2
    choices_c2 = np.zeros((S,int(N/B)), dtype = int)
    if no_repeats:
        for s in range(S):
            label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
            pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
            choices_c2[s] = pos_choices*L2 + label_choices
    else:
        choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
    choices_c1 = np.zeros((S,int(N/B)), dtype = int)
    for s in range(S):
        choices_c = []
        for k in choices_c2[s]:
            choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
        choices_c1[s] = np.array(choices_c)
    choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
    choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
    choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
    choices_c = np.tile(choices_c,B)
    # Shuffle choices pairwise (keeping x,y pairs together)
    for s in range(S):
        # Reshape into pairs, shuffle pairs, then flatten back
        pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
        np.random.shuffle(pairs_c)  # Shuffle along first dimension
        choices_c[s] = pairs_c.flatten()  # Back to 1D array
    choices_c1 = choices_c[:,0::2]
    choices_c2 = choices_c[:,1::2]
    
    # Select OOD target from distribution 2
    targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
    targets_c = choices_c2[np.arange(S),targets_c_ind]
    targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]

    # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
    filt_B = np.random.uniform(size = S) > p_B
    choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
    choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
    targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
    targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)

    filt_C = np.random.uniform(size = S) > p_C

    # Context M1 items at positions 0, 3, 6, ..., 3N-3
    e_fac1 = 1/np.sqrt(1+eps1**2)
    e_fac2 = 1/np.sqrt(1+eps2**2)
    inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
    inputs_2[filt_C,:-1,:] = (e_fac2*(mus_class_m2[choices_2] + eps2*np.random.normal(size = (S,N,D2))/np.sqrt(D2)))[filt_C]
    # Context labels at positions 2, 5, 8, ..., 3N-1
    if flip_labels:
        wrong_label = (labels_class_m2 + 1)%L2
        inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
    else:
        inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
    # M1 query at position 3N (= -2)
    inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
    # M2 query embedding (the M2 slot at position 3N+1 is filled by combine_mm_input_seqs_v1)
    inputs_2[filt_C,-1,:] = (e_fac2*(mus_class_m2[targets] + eps2*np.random.normal(size = (S,D2))/np.sqrt(D2)))[filt_C]

    # OOD context M1 items
    inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
    inputs_2[~filt_C,:-1,:] = (e_fac2*(mus_class_new2[choices_c2] + eps2*np.random.normal(size = (S,N,D2))/np.sqrt(D2)))[~filt_C]
    # OOD context labels
    inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
    # OOD M1 query at position 3N
    inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
    # OOD M2 query embedding
    inputs_2[~filt_C,-1,:] = (e_fac2*(mus_class_new2[targets_c] + eps2*np.random.normal(size = (S,D2))/np.sqrt(D2)))[~filt_C]
    
    labels = np.zeros((S,L1),dtype= bool)
    target_classes = np.zeros(S, dtype = int)
    label_sequences = np.zeros((S, N+1), dtype=int)
    
    for s in range(S):
        if filt_C[s]:
            labels[s,labels_class_m2[targets[s]]] = True
            target_classes[s] = targets[s]
            label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
        else:
            labels[s,labels_class_new2[targets_c[s]]] = True
            target_classes[s] = -1
            label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
    if seq_labels:
        return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
    elif output_target_labels:
        return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
    else:
        return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
            
            
def combine_mm_input_seqs_v1(inputs_mm, inputs_2, L_pos, N, rope = True):
    # Combine the input sequences from the two distributions.
    """
    inputs_mm: (S, 3N+2, D1) -- M1 items + label items + M1 query at position 3N,
              M2 query slot at position 3N+1.
    inputs_2:  (S, N+1, D1)  -- projected M2 features for the N context items
              plus the M2 query.

    The M2 context slots are positions 1, 4, ..., 3N-2 and the M2 query slot is
    position 3N+1 (the last position).
    """
    inputs_mm[:,1:-2:3,:] = inputs_2[:,:-1,:]
    inputs_mm[:,-1,:] = inputs_2[:,-1,:]
    if rope:
        inputs = inputs_mm
    else:
        inputs = np.zeros((inputs_mm.shape[0], inputs_mm.shape[1], L_pos + inputs_mm.shape[2]))
        inputs[:,:,L_pos:] = inputs_mm
        shifts = np.random.choice(L_pos - (3*N + 2) + 1, size = (inputs_mm.shape[0]))
        for s in range(inputs_mm.shape[0]):
            inputs[s,:,shifts[s]:shifts[s] + 3*N + 2] = np.identity(3*N + 2)
    return inputs

    
class ICLDataset(IterableDataset):
    def __init__(
        self,
        mus_label: np.ndarray,
        mus_class: np.ndarray,
        labels_class: np.ndarray,
        N: int,  # Number of item-label pairs in context
        S: int,  # Number of sequences
        Nmax: int,  # Maximum context length
        eps: float = 0.1,  # Within-class variance
        B: int = 0,  # Burstiness 
        p_B: float = 0,  # Fraction of bursty sequences
        p_C: float = 0,  # Fraction of OOD sequences
        P = None,  # Class probability distribution
        datasize: int = 10,  # Number of sequences to generate
        flip_labels: bool = False,  # Whether to flip labels
        no_repeats: bool = False,  # Whether to allow repeated items
        hybrid: bool = False,  # Whether to use hybrid position embeddings
        rope: bool = False,  # Whether using rotary position embeddings
        output_target_labels: bool = False,  # Whether to output target labels
        seq_labels: bool = False,  # Whether to output sequence labels
    ):
        super().__init__()
        self.L = mus_label.shape[0]
        self.K = mus_class.shape[0]
        self.D = mus_label.shape[1]
        self.mus_label = mus_label
        self.mus_class = mus_class
        self.labels_class = labels_class
        self.N = N
        self.S = S
        self.Nmax = Nmax
        self.eps = eps
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.no_repeats = no_repeats
        self.hybrid = hybrid
        self.rope = rope
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        # Set up class distribution if not provided
        if P is None or len(P) != self.K:
            self.P = np.ones(self.K)/self.K
        else:
            self.P = P
            
        # Set up OOD class embeddings
        self.K_c = 128
        self.mus_class_new = None
        self.labels_class_new = None
        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0

    def generate_sequence(self):
        """Generate a single sequence"""
        e_fac = 1/np.sqrt(1 + self.eps**2)
        if self.hybrid:
            input_dim = 2*self.Nmax + 2 + self.D
        elif self.rope:
            input_dim = self.D
        else:
            input_dim = 2*self.Nmax + 1 + self.D
        inputs = np.zeros((self.S, 2*self.N + 1, input_dim))
        
        # Generate context items
        n_bursts = int(self.N/self.B)
        choices = np.zeros((self.S, n_bursts), dtype=int)
        if self.no_repeats:
            for s in range(self.S):
                label_choices = np.random.choice(np.arange(self.L), size=n_bursts, replace=False)
                pos_choices = np.random.choice(np.arange(int(self.K/self.L)), size=n_bursts)
                choices[s] = pos_choices*self.L + label_choices
        else:
            choices = np.random.choice(np.arange(self.K), size=(self.S, n_bursts), p=self.P)
        
        choices = np.tile(choices, self.B)
        [np.random.shuffle(x) for x in choices]
        
        # Generate OOD context items
        self.mus_class_new = np.random.normal(size=(self.K_c, self.D))/np.sqrt(self.D)
        self.labels_class_new = np.tile(np.arange(self.L), int(self.K_c/self.L))
        choices_c = np.zeros((self.S, n_bursts), dtype=int)
        if self.no_repeats:
            for s in range(self.S):
                label_choices = np.random.choice(np.arange(self.L), size=n_bursts, replace=False)
                pos_choices = np.random.choice(np.arange(int(self.K_c/self.L)), size=n_bursts)
                choices_c[s] = pos_choices*self.L + label_choices
        else:
            choices_c = np.random.choice(np.arange(self.K_c), size=(self.S, n_bursts))
            
        choices_c = np.tile(choices_c, self.B)
        [np.random.shuffle(x) for x in choices_c]
        
        # Select target
        target_ind = np.random.choice(choices.shape[1], size=(choices.shape[0],))
        target = choices[np.arange(choices.shape[0]), target_ind]
        
        target_c_ind = np.random.choice(choices_c.shape[1], size=(choices_c.shape[0],))
        target_c = choices_c[np.arange(choices_c.shape[0]), target_c_ind]
        
        # Determine if sequence is bursty and/or OOD
        is_bursty = np.random.uniform(size=self.S) <= self.p_B
        is_ood = np.random.uniform(size=self.S) <= self.p_C
        
        choices[~is_bursty] = np.random.choice(self.K, size=(np.sum(~is_bursty), self.N), p=self.P)
        target[~is_bursty] = np.random.choice(self.K, size=(np.sum(~is_bursty),), p=self.P)
            
        if self.hybrid:
            start_ind = 2*self.Nmax + 2
        elif self.rope:
            start_ind = 0
        else:
            start_ind = 2*self.Nmax + 1
        # Context item embeddings sampled from the existing classes
        inputs[~is_ood, :-1:2, start_ind:] =(e_fac*(self.mus_class[choices] + 
                self.eps*np.random.normal(size=(self.S,self.N, self.D))/np.sqrt(self.D)))[~is_ood]  
        # Label item embeddings sampled from the existing classes with wrong/correct labels
        if self.flip_labels:
            wrong_label = (self.labels_class + 1)%self.L
            inputs[~is_ood, 1:-1:2, start_ind:] = ((self.mus_label[wrong_label])[choices])[~is_ood] 
        else:
            inputs[~is_ood, 1:-1:2, start_ind:] = ((self.mus_label[self.labels_class])[choices])[~is_ood] 
        # Target item embedding sampled from the existing classes
        inputs[~is_ood, -1, start_ind:] = ((e_fac*(self.mus_class[target] + 
                self.eps*np.random.normal(size=(self.S,self.D))/np.sqrt(self.D)))[~is_ood]) 
        
        # Context item embeddings sampled from the OOD classes  
        inputs[is_ood,:-1:2,start_ind:] = (e_fac*(self.mus_class_new[choices_c] + 
                self.eps*np.random.normal(size=(self.S,self.N, self.D))/np.sqrt(self.D)))[is_ood] 
        # Label item embeddings sampled from the OOD classes
        inputs[is_ood,1:-1:2,start_ind:] = ((self.mus_label[self.labels_class_new])[choices_c])[is_ood] 
        # Target item embedding sampled from the OOD classes
        inputs[is_ood,-1,start_ind:] = (e_fac*(self.mus_class_new[target_c] + 
                self.eps*np.random.normal(size=(self.S,self.D))/np.sqrt(self.D)))[is_ood] 
        
        # Vanilla Position Embeddings
        shifts = np.random.choice((2*self.Nmax + 1) - (2*self.N + 1) + 1, size=(self.S))
        
        labels = np.zeros((self.S,self.L),dtype= bool)
        target_classes = np.zeros(self.S, dtype = int)
        label_sequences = np.zeros((self.S, self.N+1), dtype=int)
        
        for s in range(self.S):
            if is_ood[s]:
                labels[s,self.labels_class_new[target_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(self.labels_class_new[choices_c[s]],self.labels_class_new[target_c[s]])
            else:
                labels[s,self.labels_class[target[s]]] = True
                target_classes[s] = target[s]
                label_sequences[s] = np.append(self.labels_class[choices[s]],self.labels_class[target[s]])
            if self.hybrid:
                inputs[s,:,shifts[s]:shifts[s] + 2*self.N+1] = np.identity(2*self.N+1)  
            elif not self.rope:
                inputs[s,:,shifts[s]:shifts[s] + 2*self.N+1] = np.identity(2*self.N+1)  
                
        if self.seq_labels:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif self.output_target_labels:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels), torch.FloatTensor(target_classes)
        else:
            return torch.FloatTensor(inputs), torch.FloatTensor(labels)
      
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_sequence()
            yield imgs, labels
            n += 1

class MMDataset(IterableDataset):
    def __init__(self, mus_label_m1, mus_class_m1, mus_label_m2, mus_class_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, datasize = 0):
        super().__init__()
        self.mus_label_m1 = mus_label_m1
        self.mus_class_m1 = mus_class_m1
        self.mus_label_m2 = mus_label_m2
        self.mus_class_m2 = mus_class_m2
        self.labels_class_m2 = labels_class_m2
        self.mapping_m2_to_m1 = mapping_m2_to_m1
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        L1 = mus_label_m1.shape[0]
        L2 = mus_label_m2.shape[0]
        K1 = mus_class_m1.shape[0]
        K2 = mus_class_m2.shape[0]
        D1 = mus_class_m1.shape[1]
        D2 = mus_class_m2.shape[1]
        
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1
        self.D2 = D2
            
        # Set up class distribution if not provided
        if P1 is None or len(P1) != K1:
            self.P1 = np.ones(K1)/K1
        else:
            self.P1 = P1    
        if P2 is None or len(P2) != K2:
            self.P2 = np.ones(K2)/K2
        else:
            self.P2 = P2
            
        # Set up OOD class embeddings
        self.K_c1 = 128
        self.K_c2 = int(self.K_c1 * K2 / K1)
        self.K_c2 = self.K_c2 // L2 * L2
        if self.K_c2 < L2:
            self.K_c2 = L2
        if self.K_c1 < L1 or self.K_c1%L1 != 0:
            print("K_c1 > L1 and K_c1%L1 == 0 is required")
            raise ValueError("K_c1 > L1 and K_c1%L1 == 0 is required")
        if self.K_c2 < L2 or self.K_c2%L2 != 0:
            print("K_c2 > L2 and K_c2%L2 == 0 is required")
            raise ValueError("K_c2 > L2 and K_c2%L2 == 0 is required")
        
        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0
        
        
    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = self.D2
        K_c1 = self.K_c1
        K_c2 = self.K_c2
        mus_class_m1 = self.mus_class_m1
        mus_class_m2 = self.mus_class_m2
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        
        # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        mus_class_new2 = np.random.normal(size = (K_c2,D2))/np.sqrt(D2)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C

        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        e_fac2 = 1/np.sqrt(1+eps2**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        inputs_2[filt_C,:-1,:] = (e_fac2*(mus_class_m2[choices_2] + eps2*np.random.normal(size = (S,N,D2))/np.sqrt(D2)))[filt_C]
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
        # Target item embeddings sampled from distribution 2
        inputs_2[filt_C,-1,:] = (e_fac2*(mus_class_m2[targets] + eps2*np.random.normal(size = (S,D2))/np.sqrt(D2)))[filt_C]
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        inputs_2[~filt_C,:-1,:] = (e_fac2*(mus_class_new2[choices_c2] + eps2*np.random.normal(size = (S,N,D2))/np.sqrt(D2)))[~filt_C]
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        inputs_2[~filt_C,-1,:] = (e_fac2*(mus_class_new2[targets_c] + eps2*np.random.normal(size = (S,D2))/np.sqrt(D2)))[~filt_C]
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1


def get_mm_mus_label_class_joint(K1, K2, G1, G2, D1, D2):
    """
    Generate label and class embeddings for joint-label multimodal ICL.
    Labels depend on coarse groups from BOTH modalities:
      label = group_m1[k1] * G2 + group_m2[k2]
    Total number of labels L = G1 * G2.

    K1: number of fine-grained classes for modality 1
    K2: number of fine-grained classes for modality 2
    G1: number of coarse groups for modality 1
    G2: number of coarse groups for modality 2
    D1: embedding dimension for modality 1
    D2: embedding dimension for modality 2
    """
    if K1 % G1 != 0:
        raise ValueError(f"K1 ({K1}) must be divisible by G1 ({G1})")
    if K2 % G2 != 0:
        raise ValueError(f"K2 ({K2}) must be divisible by G2 ({G2})")

    L = G1 * G2
    mus_label = np.random.normal(size=(L, D1)) / np.sqrt(D1)
    mus_class_m1 = np.random.normal(size=(K1, D1)) / np.sqrt(D1)
    mus_class_m2 = np.random.normal(size=(K2, D2)) / np.sqrt(D2)

    group_m1 = np.tile(np.arange(G1), K1 // G1)
    group_m2 = np.tile(np.arange(G2), K2 // G2)

    return mus_label, mus_class_m1, group_m1, mus_class_m2, group_m2


def generate_input_seqs_mm_joint(mus_label, mus_class_m1, group_m1, mus_class_m2, group_m2,
                                  G2, N, S, eps1=0.1, eps2=0.1, B=0, p_B=0, P1=None, P2=None,
                                  p_C=0, flip_labels=False, no_repeats=False, seq_labels=False):
    """
    Generate joint-label multimodal ICL sequences.
    M1 and M2 are sampled independently; the label for each (k1, k2) pair
    is group_m1[k1] * G2 + group_m2[k2].

    Sequence format (length 3N+2):
      [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
    Position 3N = M1 query, position 3N+1 = placeholder for projected M2 query.

    Returns:
      inputs_mm: (S, 3N+2, D1)
      inputs_2:  (S, N+1, D2)
      labels:    (S, L)   where L = G1 * G2
    Optionally also returns label_sequences: (S, N+1) if seq_labels=True.
    """
    L = mus_label.shape[0]
    G1 = int(L / G2)
    K1 = mus_class_m1.shape[0]
    K2 = mus_class_m2.shape[0]
    D1 = mus_class_m1.shape[1]
    D2 = mus_class_m2.shape[1]

    inputs_mm = np.zeros((S, 3 * N + 2, D1))
    inputs_2 = np.zeros((S, N + 1, D2))

    if P1 is None or len(P1) != K1:
        P1 = np.ones(K1) / K1
    if P2 is None or len(P2) != K2:
        P2 = np.ones(K2) / K2
    if (B > 0 and N % B != 0) or B >= N:
        print("N is not divisible by B or B >= N")
        return 0
    if B == 0:
        B = int(N / 2)
        p_B = 0

    # OOD classes
    K_c1 = 128
    K_c2 = int(K_c1 * K2 / K1)
    K_c2 = max(K_c2 // G2 * G2, G2)
    K_c1 = max(K_c1 // G1 * G1, G1)
    mus_class_new1 = np.random.normal(size=(K_c1, D1)) / np.sqrt(D1)
    mus_class_new2 = np.random.normal(size=(K_c2, D2)) / np.sqrt(D2)
    group_new1 = np.tile(np.arange(G1), K_c1 // G1)
    group_new2 = np.tile(np.arange(G2), K_c2 // G2)

    # Sample M1 and M2 independently for bursty context
    n_unique = int(N / B)
    choices_m2 = np.random.choice(K2, size=(S, n_unique), p=P2)
    choices_m1 = np.random.choice(K1, size=(S, n_unique), p=P1)

    # Interleave as (m1, m2) pairs, tile for burstiness, shuffle pairs
    choices = np.zeros((S, 2 * n_unique), dtype=int)
    choices[:, 0::2] = choices_m1
    choices[:, 1::2] = choices_m2
    choices = np.tile(choices, B)
    for s in range(S):
        pairs = choices[s].reshape(-1, 2)
        np.random.shuffle(pairs)
        choices[s] = pairs.flatten()
    choices_m1 = choices[:, 0::2]
    choices_m2 = choices[:, 1::2]

    # Target: pick one context pair for target
    targets_ind = np.random.choice(choices_m2.shape[1], size=(S,))
    targets_m2 = choices_m2[np.arange(S), targets_ind]
    targets_m1 = choices_m1[np.arange(S), targets_ind]

    # OOD context (independent sampling)
    choices_c_m2 = np.random.choice(K_c2, size=(S, n_unique))
    choices_c_m1 = np.random.choice(K_c1, size=(S, n_unique))
    choices_c = np.zeros((S, 2 * n_unique), dtype=int)
    choices_c[:, 0::2] = choices_c_m1
    choices_c[:, 1::2] = choices_c_m2
    choices_c = np.tile(choices_c, B)
    for s in range(S):
        pairs_c = choices_c[s].reshape(-1, 2)
        np.random.shuffle(pairs_c)
        choices_c[s] = pairs_c.flatten()
    choices_c_m1 = choices_c[:, 0::2]
    choices_c_m2 = choices_c[:, 1::2]

    targets_c_ind = np.random.choice(choices_c_m2.shape[1], size=(S,))
    targets_c_m2 = choices_c_m2[np.arange(S), targets_c_ind]
    targets_c_m1 = choices_c_m1[np.arange(S), targets_c_ind]

    # Non-bursty override
    filt_B = np.random.uniform(size=S) > p_B
    choices_m2[filt_B] = np.random.choice(K2, size=(np.sum(filt_B), N), p=P2)
    choices_m1[filt_B] = np.random.choice(K1, size=(np.sum(filt_B), N), p=P1)
    targets_m2[filt_B] = np.random.choice(K2, size=(np.sum(filt_B),), p=P2)
    targets_m1[filt_B] = np.random.choice(K1, size=(np.sum(filt_B),), p=P1)

    filt_C = np.random.uniform(size=S) > p_C

    e_fac1 = 1 / np.sqrt(1 + eps1 ** 2)
    e_fac2 = 1 / np.sqrt(1 + eps2 ** 2)

    # In-distribution context: M1 items at positions 0,3,6,...,3(N-1)
    inputs_mm[filt_C, :-2:3, :] = (e_fac1 * (mus_class_m1[choices_m1] + eps1 * np.random.normal(size=(S, N, D1)) / np.sqrt(D1)))[filt_C]
    inputs_2[filt_C, :-1, :] = (e_fac2 * (mus_class_m2[choices_m2] + eps2 * np.random.normal(size=(S, N, D2)) / np.sqrt(D2)))[filt_C]

    # Joint labels for context pairs
    joint_labels_ctx = group_m1[choices_m1] * G2 + group_m2[choices_m2]  # (S, N)
    if flip_labels:
        joint_labels_ctx = (joint_labels_ctx + 1) % L
    inputs_mm[filt_C, 2:-2:3, :] = mus_label[joint_labels_ctx][filt_C]

    # M1 query at position -2 (= 3N)
    inputs_mm[filt_C, -2, :] = (e_fac1 * (mus_class_m1[targets_m1] + eps1 * np.random.normal(size=(S, D1)) / np.sqrt(D1)))[filt_C]
    # M2 query in inputs_2
    inputs_2[filt_C, -1, :] = (e_fac2 * (mus_class_m2[targets_m2] + eps2 * np.random.normal(size=(S, D2)) / np.sqrt(D2)))[filt_C]

    # OOD context
    inputs_mm[~filt_C, :-2:3, :] = (e_fac1 * (mus_class_new1[choices_c_m1] + eps1 * np.random.normal(size=(S, N, D1)) / np.sqrt(D1)))[~filt_C]
    inputs_2[~filt_C, :-1, :] = (e_fac2 * (mus_class_new2[choices_c_m2] + eps2 * np.random.normal(size=(S, N, D2)) / np.sqrt(D2)))[~filt_C]
    joint_labels_ctx_ood = group_new1[choices_c_m1] * G2 + group_new2[choices_c_m2]
    inputs_mm[~filt_C, 2:-2:3, :] = mus_label[joint_labels_ctx_ood][~filt_C]
    inputs_mm[~filt_C, -2, :] = (e_fac1 * (mus_class_new1[targets_c_m1] + eps1 * np.random.normal(size=(S, D1)) / np.sqrt(D1)))[~filt_C]
    inputs_2[~filt_C, -1, :] = (e_fac2 * (mus_class_new2[targets_c_m2] + eps2 * np.random.normal(size=(S, D2)) / np.sqrt(D2)))[~filt_C]

    # Target labels (one-hot over L = G1*G2)
    labels = np.zeros((S, L), dtype=bool)
    label_sequences = np.zeros((S, N + 1), dtype=int)
    for s in range(S):
        if filt_C[s]:
            target_label = group_m1[targets_m1[s]] * G2 + group_m2[targets_m2[s]]
            if flip_labels:
                target_label = (target_label + 1) % L
            labels[s, target_label] = True
            ctx_labels = group_m1[choices_m1[s]] * G2 + group_m2[choices_m2[s]]
            if flip_labels:
                ctx_labels = (ctx_labels + 1) % L
            label_sequences[s] = np.append(ctx_labels, target_label)
        else:
            target_label = group_new1[targets_c_m1[s]] * G2 + group_new2[targets_c_m2[s]]
            labels[s, target_label] = True
            ctx_labels = group_new1[choices_c_m1[s]] * G2 + group_new2[choices_c_m2[s]]
            label_sequences[s] = np.append(ctx_labels, target_label)

    if seq_labels:
        return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
    else:
        return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)


class MMDatasetJoint(IterableDataset):
    """IterableDataset for joint-label multimodal ICL sequences."""

    def __init__(self, mus_label, mus_class_m1, group_m1, mus_class_m2, group_m2,
                 G2, N, S, eps1=0.1, eps2=0.1, B=0, p_B=0, P1=None, P2=None,
                 p_C=0, flip_labels=False, no_repeats=False, seq_labels=False, datasize=0):
        super().__init__()
        self.mus_label = mus_label
        self.mus_class_m1 = mus_class_m1
        self.group_m1 = group_m1
        self.mus_class_m2 = mus_class_m2
        self.group_m2 = group_m2
        self.G2 = G2
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize

        self.L = mus_label.shape[0]
        self.G1 = int(self.L / G2)
        self.K1 = mus_class_m1.shape[0]
        self.K2 = mus_class_m2.shape[0]
        self.D1 = mus_class_m1.shape[1]
        self.D2 = mus_class_m2.shape[1]

        if P1 is None or len(P1) != self.K1:
            self.P1 = np.ones(self.K1) / self.K1
        else:
            self.P1 = P1
        if P2 is None or len(P2) != self.K2:
            self.P2 = np.ones(self.K2) / self.K2
        else:
            self.P2 = P2

        K_c1 = 128
        K_c2 = int(K_c1 * self.K2 / self.K1)
        self.K_c1 = max(K_c1 // self.G1 * self.G1, self.G1)
        self.K_c2 = max(K_c2 // G2 * G2, G2)

        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N / 2)
            self.p_B = 0

    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        D1 = self.D1
        D2 = self.D2
        G1 = self.G1
        G2 = self.G2
        L = self.L
        K_c1 = self.K_c1
        K_c2 = self.K_c2
        mus_class_m1 = self.mus_class_m1
        mus_class_m2 = self.mus_class_m2
        mus_label = self.mus_label
        group_m1 = self.group_m1
        group_m2 = self.group_m2
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels

        inputs_mm = np.zeros((S, 3 * N + 2, D1))
        inputs_2 = np.zeros((S, N + 1, D2))

        # OOD classes
        mus_class_new1 = np.random.normal(size=(K_c1, D1)) / np.sqrt(D1)
        mus_class_new2 = np.random.normal(size=(K_c2, D2)) / np.sqrt(D2)
        group_new1 = np.tile(np.arange(G1), K_c1 // G1)
        group_new2 = np.tile(np.arange(G2), K_c2 // G2)

        n_unique = int(N / B)
        choices_m2 = np.random.choice(K2, size=(S, n_unique), p=P2)
        choices_m1 = np.random.choice(K1, size=(S, n_unique), p=P1)

        choices = np.zeros((S, 2 * n_unique), dtype=int)
        choices[:, 0::2] = choices_m1
        choices[:, 1::2] = choices_m2
        choices = np.tile(choices, B)
        for s in range(S):
            pairs = choices[s].reshape(-1, 2)
            np.random.shuffle(pairs)
            choices[s] = pairs.flatten()
        choices_m1 = choices[:, 0::2]
        choices_m2 = choices[:, 1::2]

        targets_ind = np.random.choice(choices_m2.shape[1], size=(S,))
        targets_m2 = choices_m2[np.arange(S), targets_ind]
        targets_m1 = choices_m1[np.arange(S), targets_ind]

        # OOD context
        choices_c_m2 = np.random.choice(K_c2, size=(S, n_unique))
        choices_c_m1 = np.random.choice(K_c1, size=(S, n_unique))
        choices_c = np.zeros((S, 2 * n_unique), dtype=int)
        choices_c[:, 0::2] = choices_c_m1
        choices_c[:, 1::2] = choices_c_m2
        choices_c = np.tile(choices_c, B)
        for s in range(S):
            pairs_c = choices_c[s].reshape(-1, 2)
            np.random.shuffle(pairs_c)
            choices_c[s] = pairs_c.flatten()
        choices_c_m1 = choices_c[:, 0::2]
        choices_c_m2 = choices_c[:, 1::2]

        targets_c_ind = np.random.choice(choices_c_m2.shape[1], size=(S,))
        targets_c_m2 = choices_c_m2[np.arange(S), targets_c_ind]
        targets_c_m1 = choices_c_m1[np.arange(S), targets_c_ind]

        # Non-bursty override
        filt_B = np.random.uniform(size=S) > p_B
        choices_m2[filt_B] = np.random.choice(K2, size=(np.sum(filt_B), N), p=P2)
        choices_m1[filt_B] = np.random.choice(K1, size=(np.sum(filt_B), N), p=P1)
        targets_m2[filt_B] = np.random.choice(K2, size=(np.sum(filt_B),), p=P2)
        targets_m1[filt_B] = np.random.choice(K1, size=(np.sum(filt_B),), p=P1)

        filt_C = np.random.uniform(size=S) > p_C

        e_fac1 = 1 / np.sqrt(1 + eps1 ** 2)
        e_fac2 = 1 / np.sqrt(1 + eps2 ** 2)

        # In-distribution context
        inputs_mm[filt_C, :-2:3, :] = (e_fac1 * (mus_class_m1[choices_m1] + eps1 * np.random.normal(size=(S, N, D1)) / np.sqrt(D1)))[filt_C]
        inputs_2[filt_C, :-1, :] = (e_fac2 * (mus_class_m2[choices_m2] + eps2 * np.random.normal(size=(S, N, D2)) / np.sqrt(D2)))[filt_C]

        joint_labels_ctx = group_m1[choices_m1] * G2 + group_m2[choices_m2]
        if flip_labels:
            joint_labels_ctx = (joint_labels_ctx + 1) % L
        inputs_mm[filt_C, 2:-2:3, :] = mus_label[joint_labels_ctx][filt_C]

        inputs_mm[filt_C, -2, :] = (e_fac1 * (mus_class_m1[targets_m1] + eps1 * np.random.normal(size=(S, D1)) / np.sqrt(D1)))[filt_C]
        inputs_2[filt_C, -1, :] = (e_fac2 * (mus_class_m2[targets_m2] + eps2 * np.random.normal(size=(S, D2)) / np.sqrt(D2)))[filt_C]

        # OOD context
        inputs_mm[~filt_C, :-2:3, :] = (e_fac1 * (mus_class_new1[choices_c_m1] + eps1 * np.random.normal(size=(S, N, D1)) / np.sqrt(D1)))[~filt_C]
        inputs_2[~filt_C, :-1, :] = (e_fac2 * (mus_class_new2[choices_c_m2] + eps2 * np.random.normal(size=(S, N, D2)) / np.sqrt(D2)))[~filt_C]
        joint_labels_ctx_ood = group_new1[choices_c_m1] * G2 + group_new2[choices_c_m2]
        inputs_mm[~filt_C, 2:-2:3, :] = mus_label[joint_labels_ctx_ood][~filt_C]
        inputs_mm[~filt_C, -2, :] = (e_fac1 * (mus_class_new1[targets_c_m1] + eps1 * np.random.normal(size=(S, D1)) / np.sqrt(D1)))[~filt_C]
        inputs_2[~filt_C, -1, :] = (e_fac2 * (mus_class_new2[targets_c_m2] + eps2 * np.random.normal(size=(S, D2)) / np.sqrt(D2)))[~filt_C]

        # Target labels
        labels = np.zeros((S, L), dtype=bool)
        label_sequences = np.zeros((S, N + 1), dtype=int)
        for s in range(S):
            if filt_C[s]:
                target_label = group_m1[targets_m1[s]] * G2 + group_m2[targets_m2[s]]
                if flip_labels:
                    target_label = (target_label + 1) % L
                labels[s, target_label] = True
                ctx_labels = group_m1[choices_m1[s]] * G2 + group_m2[choices_m2[s]]
                if flip_labels:
                    ctx_labels = (ctx_labels + 1) % L
                label_sequences[s] = np.append(ctx_labels, target_label)
            else:
                target_label = group_new1[targets_c_m1[s]] * G2 + group_new2[targets_c_m2[s]]
                labels[s, target_label] = True
                ctx_labels = group_new1[choices_c_m1[s]] * G2 + group_new2[choices_c_m2[s]]
                label_sequences[s] = np.append(ctx_labels, target_label)

        if self.seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1


class EncoderDataset(IterableDataset):
    def __init__(self, mus_class, eps, S, datasize):
        """
        mus_class: (K, D) array of class means
        eps: float, within-class noise scale
        samples_per_epoch: int or None, how many samples to emit per epoch
        infinite: if True, ignore samples_per_epoch and loop forever
        """
        super().__init__()
        self.mus_class = mus_class
        self.eps = eps
        self.S = S
        self.datasize = datasize
        self.n_samples = 0

        # precompute constants
        self.K, self.D = mus_class.shape
        self.e_fac = 1.0 / np.sqrt(1.0 + eps**2)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_input()
            yield imgs, labels
            n += 1

    def generate_input(self):
        inputs = np.zeros((self.S, self.D))
        choices = np.random.choice(self.K, size=(self.S,))
        inputs = self.e_fac * (self.mus_class[choices] + self.eps * np.random.normal(size=(self.S, self.D)) / np.sqrt(self.D))
        labels = np.zeros((self.S, self.K), dtype=bool)
        labels[np.arange(self.S), choices] = True
        return inputs, labels
        

class SingleOmniglotDataset(IterableDataset):
    def __init__(self, root, K, S, eps, datasize):
        """
        IMAGE_SIZE = 105
        N_CHARACTER_CLASSES = 1623 (964 + 659)
        N_EXEMPLARS_PER_CLASS = 20
        """
        self.K = K
        self.eps = eps
        self.S = S
        self.datasize = datasize
        self.n_samples = 0
        
        
        train_dataset = Omniglot(root, background=True)
        eval_dataset = Omniglot(root, background=False)

        # Merge the two datasets, only leave 100 classes as hold-out classes
        self.data_train = []
        labels_train = []
        for img,label in train_dataset:
            if label not in labels_train:
                labels_train.append(label)
                self.data_train.append(np.array(img).astype(np.float32) / 255.0)
        len_train = len(self.data_train)
        eval_list = list(eval_dataset)
        for img,label in eval_list[:10600]:
            new_label = label + len_train
            if new_label not in labels_train:
                labels_train.append(new_label)
                self.data_train.append(np.array(img).astype(np.float32) / 255.0)
        self.data_train = np.array(self.data_train[:K])
        self.labels_train = np.array(labels_train[:K])

        self.data_test = []
        labels_test = []
        label_0 = eval_list[10620][1]
        for img,label in eval_list[10620:]:
            new_label = label - label_0
            if new_label not in labels_test:
                labels_test.append(new_label)
                self.data_test.append(np.array(img).astype(np.float32) / 255.0)
        self.data_test = np.array(self.data_test)
        self.labels_test = np.array(labels_test)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_input()
            yield imgs, labels
            n += 1
            
    def generate_input(self):
        img_idxs = np.random.choice(len(self.data_train), size=(self.S,))
        imgs = self.data_train[img_idxs]
        noise = np.random.normal(0, self.eps, imgs.shape)
        imgs = imgs + noise.astype(np.float32)
        labels = np.zeros((self.S, len(self.data_train)), dtype=bool)
        label_index = self.labels_train[img_idxs]
        labels[np.arange(self.S), label_index] = True
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
    
    
class FullOmniglotDataset(IterableDataset):
    def __init__(self, root, K, S, eps, datasize):
        """
        IMAGE_SIZE = 105
        N_CHARACTER_CLASSES = 1623 (964 + 659)
        N_EXEMPLARS_PER_CLASS = 20
        """
        self.K = K
        self.eps = eps
        self.S = S
        self.datasize = datasize
        self.n_samples = 0
        
        from collections import defaultdict
        char_to_imgs = defaultdict(list)
        
        train_dataset = Omniglot(root, background=True)
        eval_dataset = Omniglot(root, background=False)
        
        for img, label in list(train_dataset):
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
        len_train = len(char_to_imgs)
        for img, label in list(eval_dataset):
            label = label + len_train
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
            
        all_labels = list(char_to_imgs.keys())
        selected_labels_train = all_labels[:K]
        self.class_to_imgs_train = {i: np.array(char_to_imgs[label])
                              for i, label in enumerate(selected_labels_train)}
        selected_labels_test = all_labels[-128:]
        self.class_to_imgs_test = {i: np.array(char_to_imgs[label]) for i, label in enumerate(selected_labels_test)}
       

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_input()
            yield imgs, labels
            n += 1
            
            
    def generate_input(self):
        class_idxs = np.random.choice(self.K, size=(self.S,), replace=True)
        batch_imgs = []
        for c in class_idxs:
            img_idx = np.random.choice(20, size=(1,))
            img = self.class_to_imgs_train[c][img_idx]
            batch_imgs.append(img)
        imgs = np.stack(batch_imgs, axis=0).squeeze(1)
        noise = np.random.normal(0, self.eps, size=imgs.shape).astype(np.float32)
        imgs = imgs + noise
        labels = np.zeros((self.S, self.K), dtype=bool)
        labels[np.arange(self.S), class_idxs] = True
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
    
    def generate_val(self):
        class_idxs = np.arange(self.K)
        batch_imgs = []
        for c in class_idxs:
            img_idx = np.random.choice(20, size=(1,))
            img = self.class_to_imgs_train[c][img_idx]
            batch_imgs.append(img)
        imgs = np.stack(batch_imgs, axis=0).squeeze(1)
        noise = np.random.normal(0, self.eps, size=imgs.shape).astype(np.float32)
        imgs = imgs + noise
        labels = np.zeros((self.K, self.K), dtype=bool)
        labels[np.arange(self.K), class_idxs] = True
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
        

class SingleOmniglotMMDataset(IterableDataset):
    def __init__(self, root, K2, mus_label_m1, mus_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, datasize = 0):
        super().__init__()
        self.mus_label_m1 = mus_label_m1
        self.mus_class_m1 = mus_class_m1
        self.mus_label_m2 = mus_label_m2
        self.labels_class_m2 = labels_class_m2
        self.mapping_m2_to_m1 = mapping_m2_to_m1
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        L1 = mus_label_m1.shape[0]
        L2 = mus_label_m2.shape[0]
        K1 = mus_class_m1.shape[0]
        D1 = mus_class_m1.shape[1]
        
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1

            
        # Set up class distribution if not provided
        if P1 is None or len(P1) != K1:
            self.P1 = np.ones(K1)/K1
        else:
            self.P1 = P1    
        if P2 is None or len(P2) != K2:
            self.P2 = np.ones(K2)/K2
        else:
            self.P2 = P2
            
        # Set up OOD class embeddings
        self.K_c1 = 128
        if self.K_c1 < L1 or self.K_c1%L1 != 0:
            print("K_c1 > L1 and K_c1%L1 == 0 is required")
            raise ValueError("K_c1 > L1 and K_c1%L1 == 0 is required")

        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0
            
        # Read omniglot dataset
        train_dataset = Omniglot(root, background=True)
        eval_dataset = Omniglot(root, background=False)
        
        self.data_train = []
        labels_train = []
        for img,label in train_dataset:
            if label not in labels_train:
                labels_train.append(label)
                self.data_train.append(np.array(img).astype(np.float32) / 255.0)
        len_train = len(self.data_train)
        eval_list = list(eval_dataset)
        for img,label in eval_list[:10600]:
            new_label = label + len_train
            if new_label not in labels_train:
                labels_train.append(new_label)
                self.data_train.append(np.array(img).astype(np.float32) / 255.0)
        self.data_train = np.array(self.data_train[:K2])

        self.data_test = []
        labels_test = []
        label_0 = eval_list[10620][1]
        for img,label in eval_list[10620:]:
            new_label = label - label_0
            if new_label not in labels_test:
                labels_test.append(new_label)
                self.data_test.append(np.array(img).astype(np.float32) / 255.0)
        self.data_test = np.array(self.data_test)

        
    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = self.data_train.shape[-1]
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        
        # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(self.data_train[choices_2][filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        inputs_2[filt_C,-1,:] = add_noise_to_img(self.data_train[targets][filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(self.data_test[choices_c2][~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        inputs_2[~filt_C,-1] = add_noise_to_img(self.data_test[targets_c][~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def generate_test_sequence(self,S,B,p_B,p_C,flip_labels=False):
        N = self.N
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = self.data_train.shape[-1]
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        eps1 = self.eps1
        eps2 = self.eps2
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        if B == 0:
            B = int(N/2)
            p_B = 0
        
        # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(self.data_train[choices_2][filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        inputs_2[filt_C,-1,:] = add_noise_to_img(self.data_train[targets][filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(self.data_test[choices_c2][~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        inputs_2[~filt_C,-1] = add_noise_to_img(self.data_test[targets_c][~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1
    
    
class FullOmniglotMMDataset(IterableDataset):
    def __init__(self, root, K2, mus_label_m1, mus_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, datasize = 0):
        super().__init__()
        self.mus_label_m1 = mus_label_m1
        self.mus_class_m1 = mus_class_m1
        self.mus_label_m2 = mus_label_m2
        self.labels_class_m2 = labels_class_m2
        self.mapping_m2_to_m1 = mapping_m2_to_m1
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        L1 = mus_label_m1.shape[0]
        L2 = mus_label_m2.shape[0]
        K1 = mus_class_m1.shape[0]
        D1 = mus_class_m1.shape[1]
        
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1

            
        # Set up class distribution if not provided
        if P1 is None or len(P1) != K1:
            self.P1 = np.ones(K1)/K1
        else:
            self.P1 = P1    
        if P2 is None or len(P2) != K2:
            self.P2 = np.ones(K2)/K2
        else:
            self.P2 = P2
            
        # Set up OOD class embeddings
        self.K_c1 = 128
        if self.K_c1 < L1 or self.K_c1%L1 != 0:
            print("K_c1 > L1 and K_c1%L1 == 0 is required")
            raise ValueError("K_c1 > L1 and K_c1%L1 == 0 is required")

        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0
            
        from collections import defaultdict
        char_to_imgs = defaultdict(list)
        
        train_dataset = Omniglot(root, background=True)
        eval_dataset = Omniglot(root, background=False)
        
        for img, label in list(train_dataset):
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
        len_train = len(char_to_imgs)
        for img, label in list(eval_dataset):
            label = label + len_train
            img_arr = np.array(img).astype(np.float32) / 255.0
            char_to_imgs[label].append(img_arr)
            
        all_labels = list(char_to_imgs.keys())
        selected_labels_train = all_labels[:K2]
        self.class_to_imgs_train = {i: np.array(char_to_imgs[label])
                              for i, label in enumerate(selected_labels_train)}
        selected_labels_test = all_labels[-128:]
        self.class_to_imgs_test = {i: np.array(char_to_imgs[label]) for i, label in enumerate(selected_labels_test)}
        
    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = next(iter(self.class_to_imgs_train.values())).shape[-1]
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        
        # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        n_per_class = next(iter(self.class_to_imgs_train.values())).shape[0]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def generate_test_sequence(self,S,B,p_B,p_C,flip_labels=False):
        N = self.N
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = next(iter(self.class_to_imgs_train.values())).shape[-1]
        K_c1 = self.K_c1
        K_c2 = 128
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        eps1 = self.eps1
        eps2 = self.eps2
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        if B == 0:
            B = int(N/2)
            p_B = 0
        
         # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        n_per_class = next(iter(self.class_to_imgs_train.values())).shape[0]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1
    
    
class CIFAR100Dataset(IterableDataset):
    def __init__(self, root, K, S=16, eps=0.0, datasize=10000,):
        """
        Iterable Dataset for CIFAR-100 with reserved novel classes.

        Args:
            root (str): Path to CIFAR-100 data.
            train (bool): If True, use training split excluding novel classes; else use test split of novel classes.
            novel_classes (int): Number of classes held out as novel (default=20).
            S (int): Number of samples per generated batch.
            eps (float): Gaussian noise sigma added to images.
            datasize (int): Total number of iterations (samples) per epoch.
            download (bool): Whether to download CIFAR-100 if missing.
        """
        super().__init__()
        self.S = S
        self.eps = eps
        self.datasize = datasize
        self.K = K
    

        # Load the full dataset once
        full_train = CIFAR100(root=root, train=True)
        full_test  = CIFAR100(root=root, train=False)

        # Group images by label
        from collections import defaultdict
        self.label_to_imgs = defaultdict(list)
        for img, label in full_train:
            arr = np.array(img).astype(np.float32) / 255.0
            self.label_to_imgs[label].append(arr)
        for img, label in full_test:
            arr = np.array(img).astype(np.float32) / 255.0
            self.label_to_imgs[label].append(arr)

        all_labels = sorted(self.label_to_imgs.keys())
        # Reserve last novel_classes as novel
        novel_classes = 10 
        self.test_labels = all_labels[-novel_classes:]
        self.train_labels = all_labels[:K]

        self.class_to_imgs_train = {
            i:  np.transpose(np.stack(self.label_to_imgs[label], axis=0), (0,3,1,2))
            for i, label in enumerate(self.train_labels)
        }
        self.class_to_imgs_test = {
            i: np.transpose(np.stack(self.label_to_imgs[label], axis=0), (0,3,1,2))
            for i, label in enumerate(self.test_labels)
        }

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            imgs, labels = self.generate_input()
            yield imgs, labels
            n += 1

    def generate_input(self):
        # Randomly choose S classes with replacement
        class_idxs = np.random.choice(self.K, size=(self.S,), replace=True)
        batch_imgs = []
        for c in class_idxs:
            # Randomly choose one image from the class
            imgs = self.class_to_imgs_train[c]
            idx = np.random.randint(len(imgs))
            batch_imgs.append(imgs[idx])

        imgs = np.stack(batch_imgs, axis=0)
        noise = np.random.normal(0, self.eps, size=imgs.shape).astype(np.float32)
        imgs = imgs + noise
        
        # One-hot labels
        labels = np.zeros((self.S, self.K), dtype=bool)
        labels[np.arange(self.S), class_idxs] = True
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
    
    def generate_val(self):
        class_idxs = np.arange(self.K)
        batch_imgs = []
        for c in class_idxs:
            imgs = self.class_to_imgs_train[c]
            idx = np.random.randint(len(imgs))
            batch_imgs.append(imgs[idx])
        imgs = np.stack(batch_imgs, axis=0)
        noise = np.random.normal(0, self.eps, size=imgs.shape).astype(np.float32)
        imgs = imgs + noise
        labels = np.zeros((self.K, self.K), dtype=bool)
        labels[np.arange(self.K), class_idxs] = True
        return torch.FloatTensor(imgs), torch.FloatTensor(labels)
        

class CIFAR100MMDataset(IterableDataset):
    def __init__(self, root, K2, mus_label_m1, mus_class_m1, mus_label_m2, labels_class_m2, mapping_m2_to_m1, N,S, eps1= 0.1, eps2 = 0.1, B= 0, p_B = 0, P1 = None, P2 = None, p_C = 0, flip_labels = False, output_target_labels = False, no_repeats = False, seq_labels = False, datasize = 0):
        super().__init__()
        self.mus_label_m1 = mus_label_m1
        self.mus_class_m1 = mus_class_m1
        self.mus_label_m2 = mus_label_m2
        self.labels_class_m2 = labels_class_m2
        self.mapping_m2_to_m1 = mapping_m2_to_m1
        self.N = N
        self.S = S
        self.eps1 = eps1
        self.eps2 = eps2
        self.B = B
        self.p_B = p_B
        self.p_C = p_C
        self.flip_labels = flip_labels
        self.output_target_labels = output_target_labels
        self.no_repeats = no_repeats
        self.seq_labels = seq_labels
        self.datasize = datasize
        self.n_samples = 0
        
        L1 = mus_label_m1.shape[0]
        L2 = mus_label_m2.shape[0]
        K1 = mus_class_m1.shape[0]
        D1 = mus_class_m1.shape[1]
        
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1

            
        # Set up class distribution if not provided
        if P1 is None or len(P1) != K1:
            self.P1 = np.ones(K1)/K1
        else:
            self.P1 = P1    
        if P2 is None or len(P2) != K2:
            self.P2 = np.ones(K2)/K2
        else:
            self.P2 = P2
            
        # Set up OOD class embeddings
        self.K_c1 = 128
        if self.K_c1 < L1 or self.K_c1%L1 != 0:
            print("K_c1 > L1 and K_c1%L1 == 0 is required")
            raise ValueError("K_c1 > L1 and K_c1%L1 == 0 is required")

        # Validate burstiness parameter
        if B > 0 and N % B != 0:
            raise ValueError("N must be divisible by B when B > 0")
        if B >= N:
            raise ValueError("B must be less than N")
        if B == 0:
            self.B = int(N/2)
            self.p_B = 0
            
        from collections import defaultdict
        self.label_to_imgs = defaultdict(list)
        
        full_train = CIFAR100(root=root, train=True)
        full_test  = CIFAR100(root=root, train=False)
        
        for img, label in full_train:
            arr = np.array(img).astype(np.float32) / 255.0
            self.label_to_imgs[label].append(arr)
        for img, label in full_test:
            arr = np.array(img).astype(np.float32) / 255.0
            self.label_to_imgs[label].append(arr)

        all_labels = sorted(self.label_to_imgs.keys())
        # Reserve last novel_classes as novel
        novel_classes = 10 
        self.test_labels = all_labels[-novel_classes:]
        self.train_labels = all_labels[:K]

        self.class_to_imgs_train = {
            i:  np.transpose(np.stack(self.label_to_imgs[label], axis=0), (0,3,1,2))
            for i, label in enumerate(self.train_labels)
        }
        self.class_to_imgs_test = {
            i: np.transpose(np.stack(self.label_to_imgs[label], axis=0), (0,3,1,2))
            for i, label in enumerate(self.test_labels)
        }

    def generate_sequence(self):
        S = self.S
        N = self.N
        B = self.B
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = next(iter(self.class_to_imgs_train.values())).shape[-1]
        K_c1 = self.K_c1
        K_c2 = 10
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        p_B = self.p_B
        p_C = self.p_C
        eps1 = self.eps1
        eps2 = self.eps2
        flip_labels = self.flip_labels
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        
        # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        n_per_class = next(iter(self.class_to_imgs_train.values())).shape[0]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_test.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def generate_test_sequence(self,S,B,p_B,p_C,flip_labels=False):
        N = self.N
        P1 = self.P1
        P2 = self.P2
        K1 = self.K1
        K2 = self.K2
        L1 = self.L1
        L2 = self.L2
        D1 = self.D1
        D2 = next(iter(self.class_to_imgs_train.values())).shape[-1]
        K_c1 = self.K_c1
        K_c2 = 10
        mus_class_m1 = self.mus_class_m1
        mus_label_m2 = self.mus_label_m2
        labels_class_m2 = self.labels_class_m2
        mapping_m2_to_m1 = self.mapping_m2_to_m1
        eps1 = self.eps1
        eps2 = self.eps2
        output_target_labels = self.output_target_labels
        no_repeats = self.no_repeats
        seq_labels = self.seq_labels
        if B == 0:
            B = int(N/2)
            p_B = 0
        
         # Decide in final input dimension
        # Layout (length 3N+2): [M1_1, M2_slot_1, y_1, ..., M1_N, M2_slot_N, y_N, M1_query, M2_query_slot]
        inputs_mm = np.zeros((S,3*N+2,D1))
        inputs_2 = np.zeros((S,N+1,D2,D2))
        
        # Generate OOD classes for distribution 1 (modality 1, here means llm) and distribution 2 (modality 2, here means vision)
        mus_class_new1 = np.random.normal(size = (K_c1,D1))/np.sqrt(D1)
        labels_class_new1 =  np.tile(np.arange(L1),int(K_c1/L1))
        labels_class_new2 =  np.tile(np.arange(L2),int(K_c2/L2))
        mapping_m2_to_m1_new = []
        for k2 in range(K_c2):
            label2 = labels_class_new2[k2]
            matching_classes = np.where(labels_class_new1 == label2)[0]
            mapping_m2_to_m1_new.append(matching_classes)
        mapping_m2_to_m1_new = np.array(mapping_m2_to_m1_new)
        # Choose the context-item classes for distribution 2, then choose the context-item classes for distribution 1 with the same labels
        choices_2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K2/L2)), size = (int(N/B)))
                choices_2[s] = pos_choices*L2 + label_choices
        else:
            choices_2 = np.random.choice(np.arange(K2), size = (S,int(N/B)), p = P2)
        choices_1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices = []
            for k in choices_2[s]:
                Pk=P1[mapping_m2_to_m1[k]]
                Pk/=np.sum(Pk)
                choices.append(np.random.choice(mapping_m2_to_m1[k], p = Pk))
            choices_1[s] = np.array(choices)
        # Interleave choices_1 and choices_2 into alternating pattern (x1,y1,x2,y2,...)
        choices = np.zeros((S, 2*int(N/B)), dtype=int)
        choices[:,0::2] = choices_1  # Even indices get choices_1 
        choices[:,1::2] = choices_2  # Odd indices get choices_2

        choices = np.tile(choices,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs = choices[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs)  # Shuffle along first dimension
            choices[s] = pairs.flatten()  # Back to 1D array
        choices_1 = choices[:,0::2]
        choices_2 = choices[:,1::2]
        
        # Select target from distribution 2
        targets_ind = np.random.choice(choices_2.shape[1], size = (S,))
        targets = choices_2[np.arange(S),targets_ind]
        # Paired M1 target (M1 class that shares the M2 target's label)
        targets_m1 = choices_1[np.arange(S), targets_ind]
    
        # Choose OOD context items classes for distribution 2 
        choices_c2 = np.zeros((S,int(N/B)), dtype = int)
        if no_repeats:
            for s in range(S):
                label_choices = np.random.choice(np.arange(L2), size = (int(N/B)), replace = False)
                pos_choices = np.random.choice(np.arange(int(K_c2/L2)), size = (int(N/B)))
                choices_c2[s] = pos_choices*L2 + label_choices
        else:
            choices_c2 = np.random.choice(np.arange(K_c2), size = (S,int(N/B)))
        choices_c1 = np.zeros((S,int(N/B)), dtype = int)
        for s in range(S):
            choices_c = []
            for k in choices_c2[s]:
                choices_c.append(np.random.choice(mapping_m2_to_m1_new[k]))
            choices_c1[s] = np.array(choices_c)
        choices_c = np.zeros((S, 2*int(N/B)), dtype=int)
        choices_c[:,0::2] = choices_c1  # Even indices get choices_c1 
        choices_c[:,1::2] = choices_c2  # Odd indices get choices_c2
        choices_c = np.tile(choices_c,B)
        # Shuffle choices pairwise (keeping x,y pairs together)
        for s in range(S):
            # Reshape into pairs, shuffle pairs, then flatten back
            pairs_c = choices_c[s].reshape(-1, 2)  # Shape: (N/B, 2)
            np.random.shuffle(pairs_c)  # Shuffle along first dimension
            choices_c[s] = pairs_c.flatten()  # Back to 1D array
        choices_c1 = choices_c[:,0::2]
        choices_c2 = choices_c[:,1::2]
        
        # Select OOD target from distribution 2
        targets_c_ind = np.random.choice(choices_c2.shape[1], size = (S,))
        targets_c = choices_c2[np.arange(S),targets_c_ind]
        targets_c_m1 = choices_c1[np.arange(S), targets_c_ind]
        
        # Determine if sequence is bursty and/or OOD [filt_B] means non-bursty,[~filt_B] means bursty, [filt_C] means in-distribution,[~filt_C] means ood
        filt_B = np.random.uniform(size = S) > p_B
        choices_2[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),N), p = P2)
        choices_1[filt_B] = np.random.choice(K1,size  = (np.sum(filt_B),N), p = P1)
        targets[filt_B] = np.random.choice(K2,size  = (np.sum(filt_B),), p = P2)
        targets_m1[filt_B] = np.random.choice(K1, size = (np.sum(filt_B),), p = P1)
        
        filt_C = np.random.uniform(size = S) > p_C
        
        def add_noise_to_img(imgs):
            noise = np.random.normal(0, self.eps2, imgs.shape)
            imgs = imgs + noise.astype(np.float32)
            return imgs
        # Context item embeddings sampled from distribution 1
        e_fac1 = 1/np.sqrt(1+eps1**2)
        inputs_mm[filt_C,:-2:3,:] = (e_fac1*(mus_class_m1[choices_1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[filt_C]
        n_per_class = next(iter(self.class_to_imgs_train.values())).shape[0]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_train.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_train[c][idx]
        inputs_2[filt_C,:-1,:] = add_noise_to_img(sampled[filt_C])
        # Label item embeddings sampled from distribution 2
        if flip_labels:
            wrong_label = (labels_class_m2 + 1)%L2
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[wrong_label][choices_2][filt_C]
        else:
            inputs_mm[filt_C,2:-2:3,:] = mus_label_m2[labels_class_m2][choices_2][filt_C]
        # M1 query at position 3N (= -2)
        inputs_mm[filt_C,-2,:] = (e_fac1*(mus_class_m1[targets_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[filt_C]
     
        # Target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_train[c][idx] for c, idx in zip(targets, sample_idxs)], axis=0)
        inputs_2[filt_C,-1,:] = add_noise_to_img(sampled[filt_C])
        
        # OOD context item embeddings sampled from distribution 1
        inputs_mm[~filt_C,:-2:3,:] = (e_fac1*(mus_class_new1[choices_c1] + eps1*np.random.normal(size = (S,N,D1))/np.sqrt(D1)))[~filt_C]
        img_size = next(iter(self.class_to_imgs_test.values())).shape[-1]
        sample_idxs = np.random.randint(0, n_per_class, size=(S, N))
        sampled = np.empty((S, N, img_size, img_size),dtype=next(iter(self.class_to_imgs_test.values())).dtype)
        for i in range(S):
            for j in range(N):
                c = choices_c2[i, j]
                idx = sample_idxs[i, j]
                sampled[i, j] = self.class_to_imgs_test[c][idx]
        inputs_2[~filt_C,:-1] =  add_noise_to_img(sampled[~filt_C])
        # OOD label item embeddings sampled from distribution 2
        inputs_mm[~filt_C,2:-2:3,:] = mus_label_m2[labels_class_new2][choices_c2][~filt_C]
        # OOD M1 query at position 3N (= -2)
        inputs_mm[~filt_C,-2,:] = (e_fac1*(mus_class_new1[targets_c_m1] + eps1*np.random.normal(size = (S,D1))/np.sqrt(D1)))[~filt_C]
        # OOD target item embeddings sampled from distribution 2
        sample_idxs = np.random.randint(0, n_per_class, size=(S,))
        sampled = np.stack([self.class_to_imgs_test[c][idx] for c, idx in zip(targets_c, sample_idxs)], axis=0)
        inputs_2[~filt_C,-1] = add_noise_to_img(sampled[~filt_C])
        
        labels = np.zeros((S,L1),dtype= bool)
        target_classes = np.zeros(S, dtype = int)
        label_sequences = np.zeros((S, N+1), dtype=int)
        
        for s in range(S):
            if filt_C[s]:
                labels[s,labels_class_m2[targets[s]]] = True
                target_classes[s] = targets[s]
                label_sequences[s] = np.append(labels_class_m2[choices_2[s]],labels_class_m2[targets[s]])
            else:
                labels[s,labels_class_new2[targets_c[s]]] = True
                target_classes[s] = -1
                label_sequences[s] = np.append(labels_class_new2[choices_c2[s]],labels_class_new2[targets_c[s]])
        if seq_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.FloatTensor(label_sequences)
        elif output_target_labels:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels), torch.LongTensor(target_classes)
        else:
            return torch.FloatTensor(inputs_mm), torch.FloatTensor(inputs_2), torch.FloatTensor(labels)
        
    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            start, end = 0, self.datasize
        else:
            per_worker = int(math.ceil(self.datasize / worker.num_workers))
            start = worker.id * per_worker
            end   = min(start + per_worker, self.datasize)

        n = 0
        while n < (end - start):
            inputs_m, inputs_2, labels = self.generate_sequence()
            yield inputs_m, inputs_2, labels
            n += 1