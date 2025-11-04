
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import ResidualGatedGCNLayer, MLP


class TSPModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.node_encoder = TSP_NodeEncoder(**model_params)
        self.graph_encoder = TSP_GraphEncoder(**model_params)

        self.node_encodingblock = Node_EncodingBlock(**model_params)  
        self.decoder = TSP_Decoder(**model_params)
        
        self.encoded_nodes = None
        self.encoded_graph = None
        # shape: (batch, problem, EMBEDDING_DIM)
        self.node_feature = None
        self.method = self.model_params['method']  

    def pre_forward(self, reset_state, x_edges, x_edges_values, x_node_indices, x_nodes_false):
        if self.method == "edge_node" or self.method == "KNN_edge":
            self.node_feature = None
        else: 
            self.node_feature = self.node_encodingblock(x_edges_values)  
            # shape：(batch, node, embedding)

        self.encoded_graph = self.graph_encoder(reset_state.problems, x_edges, x_edges_values, x_node_indices, x_nodes_false, self.node_feature)
        self.encoded_nodes = self.node_encoder(reset_state.problems, x_nodes_false, self.node_feature)
        # shape: (batch, problem, EMBEDDING_DIM)

        self.decoder.set_kv(self.encoded_graph, self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            encoded_first_graph = _get_encoding(self.encoded_graph, selected)
            # shape: (batch, pomo, embedding)

            self.decoder.set_q1(encoded_first_node, encoded_first_graph)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            encoded_last_graph = _get_encoding(self.encoded_graph, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, encoded_last_graph, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
    return picked_nodes


###################################### 
## Precoder
########################################
class Node_EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']   # 256
        self.row_encoding_block = EncodingBlock(**model_params)    

    def forward(self, x_edges_values):
        batch_size = x_edges_values.size(0)
        problem_size = x_edges_values.size(1)
        # [A]row_emb=0
        row_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim)) 
        # emb.shape: (batch, node, embedding)

        # [B]col_emb=one-hot
        col_emb = torch.zeros(size=(batch_size, problem_size, self.embedding_dim)) 
        # shape: (batch, node, embedding)

        if problem_size <= self.embedding_dim:  
            seed_cnt = problem_size 
            rand = torch.rand(batch_size, seed_cnt)
            batch_rand_perm = rand.argsort(dim=1)  
        else:   
            seed_cnt = self.embedding_dim
            rand = torch.rand(batch_size, seed_cnt)
            batch_rand_perm = rand.argsort(dim=1)

            if self.embedding_dim <= problem_size < self.embedding_dim*2:  
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm), dim=-1)
            elif self.embedding_dim * 2 <= problem_size < self.embedding_dim * 3:  
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm, batch_rand_perm), dim=-1)
            elif self.embedding_dim*3 <= problem_size < self.embedding_dim*4: 
                batch_rand_perm = torch.cat((batch_rand_perm, batch_rand_perm, batch_rand_perm, batch_rand_perm), dim=-1)
            else:
                raise NotImplementedError

        rand_idx = batch_rand_perm[:, : problem_size]
        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, problem_size)
        n_idx = torch.arange(problem_size)[None, :].expand(batch_size, problem_size)
        col_emb[b_idx, n_idx, rand_idx] = 1

        # row_encoding_block
        row_emb_out = self.row_encoding_block(row_emb, col_emb, x_edges_values)  
        return row_emb_out  # (batch, node, embedding)


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)   
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)
        
        self.add_n_normalization_1 = Add_And_Batch_Normalization(**model_params)  
        self.add_n_normalization_2 = Add_And_Batch_Normalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)
        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        return out3
        # shape: (batch, row_cnt, embedding)

########################################
# DOUBLE ENCODER
########################################
 
# TSP_GraphEncoder
class TSP_GraphEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.method = self.model_params['method']  
        self.embedding_dim = self.model_params['embedding_dim']  
        self.hidden_dim = self.model_params['hidden_dim'] 
        self.num_layers = self.model_params['GCN_dim']     
        self.mlp_layers = self.model_params['mlp_layers']  
        self.aggregation = self.model_params['aggregation']  
        self.knn_node_edge = self.model_params['knn_node_edge']  

        self.XE_embedding = nn.Linear(2, self.embedding_dim)    
        self.KNN_embedding = nn.Linear(self.knn_node_edge, self.embedding_dim)  
        # Node and edge embedding layers/lookups
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)  
        self.edges_embedding = nn.Embedding(3, self.hidden_dim // 2)         

        # GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):  
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # MLP classifiers
        self.mlp_nodes = MLP(self.hidden_dim, self.embedding_dim, self.mlp_layers)   

    def forward(self, data, x_edges, x_edges_values, x_node_indices, x_nodes_false, node_feature):

        if self.method == "edge_node":
            x = self.XE_embedding(data)

        elif self.method == "KNN_edge":
            x = self.KNN_embedding(x_nodes_false)

        elif self.method == "AB_edge":  
            x = node_feature            

        elif self.method == "Gknn_Nab":
            x = self.KNN_embedding(x_nodes_false)

        elif self.method == "Gab_Nknn":
            x = node_feature      
        else:
            raise NotImplementedError("Unknown search method")

        # [2]edge-embedding 
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  
        e_tags = self.edges_embedding(x_edges)             
        e = torch.cat((e_vals, e_tags), dim=3)     # [B N N H]
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # [B,N,H] [B,N,N,H]

        # MLP classifier
        y_pred_nodes = self.mlp_nodes(x)       # [B,N,H]
        return y_pred_nodes 


# NODE-encoder
class TSP_NodeEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.method = self.model_params['method']    
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        knn_node_edge = self.model_params['knn_node_edge']   

        self.XE_embedding = nn.Linear(2, embedding_dim)    
        self.KNN_embedding = nn.Linear(knn_node_edge, embedding_dim)  
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, x_nodes_coord, node_feature):
        # data.shape: (batch, problem, 2)
        if self.method == "edge_node":
            embedded_input = self.XE_embedding(data)       
        elif self.method == "KNN_edge":
            embedded_input = self.KNN_embedding(x_nodes_coord)  
        elif self.method == "AB_edge":
            embedded_input = node_feature    
        elif self.method == "Gknn_Nab":
            embedded_input = node_feature   
        elif self.method == "Gab_Nknn":
            embedded_input = self.KNN_embedding(x_nodes_coord)  
        else:
            raise NotImplementedError("Unknown search method")

        out = embedded_input
        for layer in self.layers:
            out = layer(out)     
        # shape: (batch, problem, embedding)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

        self.addAndNormalization1 = Add_And_Batch_Normalization(**model_params)
        self.addAndNormalization2 = Add_And_Batch_Normalization(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)
        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

########################################
# DECODER
########################################
class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        # [encoder1]
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        # [encoder2]
        self.Wq_first_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_e = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k_e = None   
        self.v_e = None  
        self.single_head_key_e = None
        self.q_first_e = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_graph, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2) # shape: (batch, embedding, problem)

        self.k_e = reshape_by_heads(self.Wk_e(encoded_graph), head_num=head_num)
        self.v_e = reshape_by_heads(self.Wv_e(encoded_graph), head_num=head_num)  # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key_e = encoded_graph.transpose(1, 2)

    def set_q1(self, encoded_q1, encoded_first_graph):   
        # encoded_q.shape: (batch, n, embedding)   # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)
        self.q_first_e = reshape_by_heads(self.Wq_first_e(encoded_first_graph), head_num=head_num)

    def forward(self, encoded_last_node, encoded_last_graph, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']
        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q_last_e = reshape_by_heads(self.Wq_last_e(encoded_last_graph), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q_x = self.q_first + q_last  
        q_e = self.q_first_e + q_last_e 
        q = q_x + q_e 

        out_concat1 = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out_concat2 = multi_head_attention(q, self.k_e, self.v_e, rank3_ninf_mask=ninf_mask)

        # shape: (batch, pomo, head_num*qkv_dim)
        mh_atten_out1 = self.multi_head_combine(out_concat1)
        mh_atten_out2 = self.multi_head_combine2(out_concat2)

        mh_atten_out = mh_atten_out1 + mh_atten_out2   
        single_head_key = self.single_head_key + self.single_head_key_e  
        # shape: (batch, pomo, embedding)
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)
        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################
def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    return out_concat


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))


# BatchNormalization  
class Add_And_Batch_Normalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim
    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)
        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)
        added = input1 + input2
        normalized = self.norm(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)
        return back_trans


###################################################
class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']  # 16,
        mix1_init = model_params['ms_layer1_init']     # (1/2)**(1/2),
        mix2_init = model_params['ms_layer2_init']     # (1/16)**(1/2),

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):  # (q, k, v, cost_mat=problem)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)
        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)
        ms1_activated = F.relu(ms1)
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)
        mixed_scores = ms2.transpose(1, 2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)
        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)
        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)
        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)
        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)
        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat
