from easydict import EasyDict  as edict


def update_args(args):

    for k,v in task_args.items():
        if k != 'dataset':
            args['task_'+k] = v
        for k, v in task_args.dataset.items():
            args['task_dataset_' + k] = v
    for k,v in play_args.items():
        if k != 'dataset':
            args['play_'+k] = v
        for k, v in play_args.dataset.items():
            args['play_dataset_' + k] = v
        
    return args


task_args = edict()
play_args = edict()
eval_args = edict()
args = edict()

args.seed = 222
args.showData = False # show samples of data at start of training.
args.cometKey = 'w4JbvdIWlas52xdwict9MwmyH' 
args.cometWs = 'alainray'
args.cometName = 'spur'
args.use_comet = False
args.model = 'scnn'
args.load_pretrained = False
args.pretrained_model_type = "nobs"
args.pretrained_path = 'models/scnn_0.625_cmnist_baseline.pth'
args.frozen_features = False
args.save_model = False
args.save_model_folder = 'models'
args.save_grads = True
args.save_grads_folder = 'grads'
args.save_model_path = "cmnist_baseline.pth"
args.dataset_paths = {'synmnist': "SynMNIST"}
args.hidden_dim = 100
args.max_cur_iter = 0
args.forget_method = 'random' # random/absolute/mag_full/mag_partial
args.forget_criteria = 'avg_gradients'
args.forget_asc = True # if True forget weights with least magnitude, else with greatest magnitude
args.forget_threshold = 0.1
args.n_freeze_layers = 1    
args.n_layers = 1                       # Number of layers to forget
task_args.duplicate = False             # Do we add a different color copy of training image to dataset?

task_args.n_interventions = 10
task_args.total_iterations = 100
task_args.dataset = {'name': 'synmnist', 'p': 0.5 , 'bg': 'nobg', 'splits': ['train','val','test'], 'baseline': False, 'bs': 16000}
task_args.mode = ["task"
                   #, 'play'
                   ,'forget'
                   ]
play_args.n_interventions = 0
play_args.total_iterations = 200
play_args.dataset = {'name': 'synmnist', 'p': 0.95 , 'bg': 'nobg', 'splits': ['train','val'], 'baseline': False, 'bs': 10000}

play_args.duplicate = False
args.eval_datasets = dict()

   # {'name': 'synmnist', 'p': 0.95 , 'bg': 'nobg', 'splits': ['train','val'], 'bs': 10000}
    
    #{'synmnist': { 'p': 0.75 , 'bg': 'nobg', 'splits': ['val'], 'bs': 128}}
     

if 'task' in task_args.mode:
    args.eval_datasets['task'] = task_args.dataset
if 'play' in task_args.mode:
    args.eval_datasets['play'] = play_args.dataset

#args.eval_datasets['eval'] = {'name': 'synmnist', 'p': 0.95 , 'bg': 'images', 'splits': ['train','val','test'], 'bs': 10000}

args = update_args(args)
args.task_args = task_args
args.play_args = play_args
args.eval_args = eval_args