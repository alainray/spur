from easydict import EasyDict  as edict
train_args = edict()
play_args = edict()
args = edict()

args.seed = 111
args.showData = False # show samples of data at start of training.
args.cometKey = '' 
args.cometWs = ''
args.cometName = ''
args.spurious_probability = 0.99
args.model = 'scnn'
args.hidden_dim = 100
args.max_cur_iter = 0
args.n_layers = 1                # Number of layers to forget
args.train_baseline = True # Non Colored version of dataset
args.play_baseline = False
train_args.duplicate = False            # Do we add a different color copy of training image to dataset?
args.aid_perc = 0.0              # What ratio should we add. 0: same as baseline, 1: every image has a copy
train_args.n_interventions = 0
train_args.total_iterations = 1000
train_args.dataset = 'cmnist'
train_args.mode = ["std"
                   #, 'play'
                  # ,'forget'
                   ]
play_args.n_interventions = 0
play_args.total_iterations = 200
play_args.dataset = 'cmnist'

play_args.duplicate = False

for k,v in train_args.items():
    args['train_'+k] = v
for k,v in play_args.items():
    args['play_'+k] = v