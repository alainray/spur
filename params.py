from easydict import EasyDict  as edict
train_args = edict()
play_args = edict()
args = edict()

args.seed = 123
args.cometKey = 'w4JbvdIWlas52xdwict9MwmyH' 
args.cometWs = 'alainray'
args.cometName = 'spur'
args.spurious_probability = 0.5
args.model = 'mlp'
args.hidden_dim = 100
args.max_cur_iter = 0
args.n_layers = 1                   # Number of layers to forget
train_args.n_interventions = 0
train_args.total_iterations = 10
train_args.dataset = 'cfmnist'
train_args.mode = ["std", 'play']
play_args.n_interventions = 0
play_args.total_iterations = 5
play_args.dataset = 'cmnist'