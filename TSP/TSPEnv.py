
from dataclasses import dataclass
import torch
from TSProblemDef import get_random_problems, get_edge_node_problems, augment_xy_data_by_8_fold
from torch.autograd import Variable
from tqdm import tqdm
import os

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class TSPEnv:
    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
 
        self.num_neighbors = env_params['num_neighbors']   
        self.knn_node_edge = env_params['knn_node_edge']     
        self.data_path = env_params['data_path']        
        self.mode = env_params['mode']
        self.optimal_label = env_params['optimal_label']
        self.raw_pkl_nodes = None

        self.raw_data_nodes = None
        self.raw_data_tours = None
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        # self.offset = None
        self.solutions = None  
        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.x_edges = None         
        self.x_edges_values = None   
        self.x_node_indices = None  
        self.x_nodes_false = None   

        self.noaug_problems = None

    def load_problems(self, episode, batch_size, aug_factor=1):
        self.batch_size = batch_size
        if self.mode == 'train':
            self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            # test
            if os.path.splitext(self.data_path)[1] == '.txt':  # uniform
                self.problems, self.solutions = self.raw_data_nodes[episode: episode + batch_size], self.raw_data_tours[
                                                                                                    episode: episode + batch_size]
            # cluster/expansion/grid/implosion/uniform
            if os.path.splitext(self.data_path)[1] == '.pkl':
                self.problems = self.raw_pkl_nodes[episode: episode + batch_size]
                self.solutions = None
                print("problems:", self.problems.shape)  

        self.x_edges, self.x_edges_values, self.x_node_indices, self.x_nodes_false = get_edge_node_problems(self.problems, self.num_neighbors, self.knn_node_edge)

        self.noaug_problems = self.problems  
        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            if aug_factor == 8:  # test 
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)
                if self.x_nodes_false is not None:
                    self.x_nodes_false = self.x_nodes_false.repeat(aug_factor, 1, 1)
            else:   # or 128
                self.problems = self.problems.repeat(aug_factor, 1, 1)
                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)
                if self.x_nodes_false is not None:
                    self.x_nodes_false = self.x_nodes_false.repeat(aug_factor, 1, 1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)


    def load_raw_data(self, episode, begin_index=0):  
        print('load raw dataset begin!')
        self.raw_data_nodes = []
        self.raw_data_tours = []
        for line in tqdm(open(self.data_path, "r").readlines()[0+begin_index: episode+begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            self.raw_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

            self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours, requires_grad=False)
        print(f'load raw dataset done!', )


    def load_pkl_distribution_problems(self, load_path=None, episode=None):
        if self.data_path is not None:   
            import os
            import pickle
            filename = self.data_path
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            problems = torch.FloatTensor(data).cuda()
            self.raw_pkl_nodes = problems


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done


    def step(self, selected):
        # selected.shape: (batch, pomo)
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done


    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


    def _get_best_distance(self, batch_size):
        if self.solutions != None:
            gathering_index = self.solutions.unsqueeze(2).expand(batch_size, self.problem_size, 2)
            # shape: (batch, problem, 2) 
            seq_expanded = self.noaug_problems   
            ordered_seq = seq_expanded.gather(dim=-2, index=gathering_index)
            # shape: (batch, problem, 2)  
            rolled_seq = ordered_seq.roll(dims=-2, shifts=-1)  
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
            # shape: (batch, problem) 
            travel_distances = segment_lengths.sum(-1).mean()

        else:  
            travel_distances = self.optimal_label

        return travel_distances

