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

# ----------- COMET ML ---------------------
args.cometKey = 'w4JbvdIWlas52xdwict9MwmyH' 
args.cometWs = 'alainray'
args.cometName = 'spur'
args.use_comet = False


# --------------- MODEL ---------------------
args.model = 'scnn'
args.hidden_dim = 100
args.output_dims = 1  # Equals number of classes

# ---------- TRAINING PARAMS ----------------
args.load_pretrained = False                                   # Do we start with a pretrained model or not
args.pretrained_model_type = "nobs"                            # 'nobs' = trained on colored images - "bs" = trained on grayscale images. 
args.pretrained_path = 'models/scnn_0.625_cmnist_baseline.pth' # path to pretrained model
args.frozen_features = False                                   # If true, train only classifier.
args.n_freeze_layers = 1 
task_args.n_interventions = 50                                 # Amount of times we stop training before applying intervention (forgetting, playing)
task_args.total_iterations = 2000                              # 9452 = 1 epoch

# ---------- MODEL PERSISTENCE --------------
args.save_model = False                                        # Save last model
args.save_best = True                                          # Save best performing model
args.save_model_folder = 'models'                              # Folder where models are stored 
args.save_grads = False                                        # Whether to save the average gradient per epoch      
args.save_grads_folder = 'grads'                               # Folder where gradients are saved to
args.after_training = False                                    # Should we add additional training after final intervention
args.save_model_path = ".pth"                                  # suffix for saved model (name depends on model settings)

# ------------------- TRAINING METHOD ------------------------------
args.max_cur_iter = 0
args.mode = ["task"                                             # task = train on dataset defined in task_args 
               #, 'play'                                        #  play = train on dataset defined in play_args 
               #,'forget'                                       # forget = after training on task, forget using method defined in args.forget_method
                   ]
args.base_method = "gdro"                                       # gdro = group distributionally robust optimization
                                                                # rw = reweight losses
                                                                # irm = invariant risk minimization
                                                                # arm = Adaptive Risk Minimization
                                                                # erm = Empirical Risk Minimization 

args.forget_method = 'random'                                   # random = Forget random weights
                                                                # absolute = Forget 
                                                                # mag_full = Forget based on magnitude of criteria, considering the whole model.
                                                                # mag_partial = Forget based on magnitude of criteria, considering each layer of the model.
args.forget_criteria = 'avg_gradients'                          # avg_gradients = consider gradient magnitude
                                                                # magnitude = consider parameter magnitude
args.forget_asc = True                                          # if True forget weights with least magnitude, else with greatest magnitude
args.forget_threshold = 0.1                                     # Forget ratio cutoff (0.1 means forget 10%)
args.n_layers = 1                                               # Number of layers to forget

task_args.duplicate = False                                     # Do we add a different color copy of training image to dataset? IS THIS BEING USED??
#task_args.dataset = {'name': 'synmnist', 'p': 0.875 , 'bg': 'nobg', 'splits': ['train','val'], 'baseline': False, 'bs': 16000}
play_args.n_interventions = 1
play_args.total_iterations = 200
play_args.dataset = {'name': 'synmnist', 'p': 0.95 , 'bg': 'nobg', 'splits': ['train','val'], 'baseline': False, 'bs': 10000}

# --------- DATASET -----------------------------------------------------------------------------
args.eval_datasets = dict()                                    # Which datasets to evaluate
args.task_datasets = dict()     
args.dataset_paths = {'synmnist': "../datasets/SynMNIST",      # Path for each dataset
                      'mnistcifar': "../datasets/MNISTCIFAR"}
task_args.dataset = {'name': 'mnistcifar', 'corr': 0.75, 'splits': ['train','id', 'val'], 'bs': 10000, "binarize": True}
args.task_datasets['env1'] = {'name': 'mnistcifar', 'corr': 0.75, 'splits': ['train','id', 'val'], 'bs': 10000, "binarize": True}
args.task_datasets['env2'] = {'name': 'mnistcifar', 'corr': 0.9, 'splits': ['train','id', 'val'], 'bs': 10000, "binarize": True}


if 'play' in args.mode:
    args.eval_datasets['play'] = play_args.dataset
# All datasets listed on eval_datasets will be evaluated. One dataset per key, however, each dataset may evaluate multiple splits.
for ds_id, ds in args.task_datasets.items():
    args.eval_datasets[f'task_{ds_id}'] = ds 
args.eval_datasets['eval'] = {'name': 'mnistcifar', 'corr': 0.0, 'splits': ['val'], 'bs': 50000, "binarize": True}
play_args.duplicate = False                                     # PROBABLY NEVER USED

   # {'name': 'synmnist', 'p': 0.95 , 'bg': 'nobg', 'splits': ['train','val'], 'bs': 10000}
    
    #{'synmnist': { 'p': 0.75 , 'bg': 'nobg', 'splits': ['val'], 'bs': 128}}
#if 'task' in task_args.mode:
#    args.eval_datasets['task'] = task_args.dataset
#args.eval_datasets['eval'] = {'name': 'synmnist', 'p': 0.95 , 'bg': 'images', 'splits': ['train','val','test'], 'bs': 10000}

# --------------- Consolidate all settings on args --------------------
args = update_args(args)
args.task_args = task_args
args.play_args = play_args
args.eval_args = eval_args