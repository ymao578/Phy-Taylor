import os
import time

import copy
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import helperfns
import RTaylorNN as tnn
import Rtraining

params = {}

params['data_name'] = 'climate'
params['seed'] = 1

input_dim = 12; 
params['uncheckable_dist_weights'] = ['tn','tn','tn']
params['uncheckable_output_size'] = [input_dim,50,5,2]
params['uncheckable_epd'] = np.array([0,0,2])
params['uncheckable_act'] = ['elu','elu','none']

params['uncheckable_com_type1'] = ['relu','none','none']
params['uncheckable_com_type2'] = ['none','none','none']

#params['uncheckable_com_type1'] = ['none','none','none']
#params['uncheckable_com_type2'] = ['none','none','none']

params['uncheckable_dist_biases'] = ['normal','normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0,0])) 

params['traj_len'] = 150   #must equate to batch size
params['Xwidth'] = 12
params['Ywidth'] = 2
params['lYwidth'] = 2

params['dynamics_lam'] = 1;

params['exp_name'] = 'exp_1'
params['folder_name'] = 'f2taylorcomon1'


params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])

params['number_of_data files_for_training'] = 10
params['num_passes_per_file'] = 100 * 8 * 100
params['batch_size'] = 150
params['num_steps_per_batch'] = 2
params['loops for val'] = 1

for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')