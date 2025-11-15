# EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems

The PyTorch Implementation of IJCAI 2025 -- [EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems](https://www.ijcai.org/proceedings/2025/954)

EFormer, an Edge-based Transformer model that uses edge as the sole input for VRPs. Our approach employs a precoder module with a mixed-score attention mechanism to convert edge information into temporary node embeddings. 
We also present a parallel encoding strategy characterized by a graph encoder and a node encoder, each responsible for processing graph and node embeddings in distinct feature spaces, respectively. 
This design yields a more comprehensive representation of the global relationships among edges. In the decoding phase, parallel context embedding and multi-query integration are used to compute separate attention mechanisms over the two encoded embeddings, facilitating efficient path construction. We train EFormer using reinforcement learning in an autoregressive manner.

## Overview

<img width="1379" height="401" alt="image" src="https://github.com/user-attachments/assets/95f5c4d8-3d1e-4e9f-8da6-4a10e66314e7" />


## Download datasets and models

Download `datasets` and `models` from [Hugging Face](https://huggingface.co/datasets/Regina921/EFormer/tree/main). 

Unzip `TSP-results.zip` and `CVRP-results.zip`, and organize the files in the project directory as follows:

```bash

EFormer
├─ TSP
│  ├─ data
│  └─ results
└─ CVRP
   ├─ data
   └─ results

```


## Dependencies

```bash
Python >= 3.8
Pytorch >= 2.0.1
numpy==1.24.4
matplotlib==3.5.2 
tqdm==4.67.1
```
 

## Citation
 If this repository is helpful for your research, please cite our paper:
 
```bash
 @inproceedings{ijcai2025p954,
  title     = {EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems},
  author    = {Meng, Dian and Cao, Zhiguang and Wu, Yaoxin and Hou, Yaqing and Ge, Hongwei and Zhang, Qiang},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {8582--8590},
  year      = {2025},
  month     = {8},
  doi       = {10.24963/ijcai.2025/954},
  url       = {https://doi.org/10.24963/ijcai.2025/954},
}
```


## Acknowledgments

* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/yd-kwon/POMO
* https://github.com/yd-kwon/MatNet 
 

