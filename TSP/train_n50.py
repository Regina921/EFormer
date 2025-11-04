##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0  
##########################################################################################
# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils import create_logger, copy_all_src
from TSPTrainer import TSPTrainer as Trainer

#####[method]#####################################################################################
 
# parameters
TSP = 50  
NN = 20  
KK = 0   
 
env_params = {
    'mode': 'train',
    'problem_size': TSP,
    'pomo_size': TSP,
    'num_neighbors': NN,      
    'knn_node_edge': KK,      
    'data_path': None,        
    'optimal_label': None,

}

# method: "edge_node" /"KNN_edge" /"AB_edge" /"Gknn_Nab"/ "Gab_Nknn"
model_params = {
    'method': "edge_node",    # edge+node
    # 'method': "AB_edge",    # only edge

    'knn_node_edge': KK,     
    'embedding_dim': 256,     
    'sqrt_embedding_dim': 256**(1/2),  
    'head_num': 16,           
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),  
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',

    'hidden_dim': 256,   
    'GCN_dim': 6, 
    'mlp_layers': 3,
    'aggregation': "mean",
 
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,       
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,],    
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 510,
    'train_episodes': 1000 * 100,   
    'train_batch_size': 128,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_50.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,   
 
        'path': './result_tsp50',
        'epoch': 510,

    }
}

logger_params = {
    'log_file': {
        'desc': f'{CUDA_DEVICE_NUM}train_K0N{NN}_tsp50',   
  
        'filename': 'run_log10W'
    }
}
 
##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()

 