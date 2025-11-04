# EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems

The PyTorch Implementation of IJCAI 2025 -- EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems

EFormer, an Edge-based Transformer model that uses edge as the sole input for VRPs. Our approach employs a precoder module with a mixed-score attention mechanism to convert edge information into temporary node embeddings. 
We also present a parallel encoding strategy characterized by a graph encoder and a node encoder, each responsible for processing graph and node embeddings in distinct feature spaces, respectively. 
This design yields a more comprehensive representation of the global relationships among edges. In the decoding phase, parallel context embedding and multi-query integration are used to compute separate attention mechanisms over the two encoded embeddings, facilitating efficient path construction. We train EFormer using reinforcement learning in an autoregressive manner.

## Overview

## Poster
 
## Dependencies


## How to Run

## Acknowledgments

## Citation
 
