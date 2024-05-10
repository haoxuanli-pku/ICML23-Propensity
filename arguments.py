import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='yahooR3', help='Choose from {yahooR3, coat, product}')
    parser.add_argument('--base_model_args', type=dict, default={'emb_dim': 10, 'learning_rate': 0.01, 'imputaion_lambda': 0.01, 'weight_decay': 1}, 
                help='base model arguments.')
    parser.add_argument('--propensity_model_args', type=dict, default={'learning_rate': 0.1, 'weight_decay': 0.001}, 
                help='propensity model arguments.')
    parser.add_argument('--imputation_model_args', type=dict, default= {'learning_rate': 1e-1, 'weight_decay': 1e-4}, 
                help='imputation model arguments.')          
    parser.add_argument('--training_args', type=dict, default = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [20, 500]}, 
                help='training arguments.')
    parser.add_argument('--uniform_ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    return parser.parse_args()
