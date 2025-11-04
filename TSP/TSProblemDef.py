  
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.autograd import Variable


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems 


def get_edge_node_problems(problems, num_neighbors, knn_node_edge):
    batch_size = problems.shape[0]
    problem_size = problems.shape[1]
    # -------------------------
    instances_temp = problems.unsqueeze(1)
    distance_matrixs_temp = instances_temp - instances_temp.transpose(1, 2)
    distance_matrixs = torch.norm(distance_matrixs_temp, dim=-1)
    W_val_batch = distance_matrixs  # [B, N+1, N+1]

    W_val_batch_clone = W_val_batch.clone()
    if num_neighbors == -1:     
        W_batch = torch.ones((batch_size, problem_size, problem_size))  # Graph is fully connected
        if knn_node_edge == 0: 
            knn_node_feature = None
            knn_node_indice = None
        else:   
            knn_edge, knn_indices = torch.topk(W_val_batch_clone, k=knn_node_edge+1, dim=-1, largest=False)
            # node feature 
            knn_node_feature = knn_edge[:, :, 1: knn_node_edge+1]   
            knn_node_indice = knn_indices[:, :, 1: knn_node_edge+1]

    else: 
        # Initialize the adjacency matrix with zeros
        W_batch = torch.zeros((batch_size, problem_size, problem_size)) 
        knn_edge, knn_indices = torch.topk(W_val_batch_clone, k=max(knn_node_edge+1, num_neighbors+1), dim=-1, largest=False) 
        knn_edge_indices = knn_indices[:, :, 1:num_neighbors+1] 

        # Set connections for k-nearest neighbors
        batch_idx = torch.arange(batch_size)[:, None, None]
        problem_idx = torch.arange(problem_size)[None, :, None]
        W_batch[batch_idx, problem_idx, knn_edge_indices] = 1

        if knn_node_edge == 0: 
            knn_node_feature = None
            knn_node_indice = None
        else: 
            # node feature 
            knn_node_feature = knn_edge[:, :, 1: knn_node_edge+1]  
            knn_node_indice = knn_indices[:, :, 1: knn_node_edge+1]

    # Set the diagonal elements to 2 for self-connections
    torch.diagonal(W_batch, dim1=-2, dim2=-1).fill_(2)

    batch_edges = torch.tensor(W_batch, dtype=torch.int)  
    batch_edges_values = W_val_batch      

    batch_node_indices = knn_node_indice   
    batch_node_features = knn_node_feature  # node_features 

    return batch_edges, batch_edges_values, batch_node_indices, batch_node_features

 
def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)
    return aug_problems

