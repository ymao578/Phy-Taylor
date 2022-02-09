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

# settings related to dataset
params['data_name'] = 'Pendulum'
n = 1  # dimension of system (and input layer)

params['seed'] = 10000

# defaults related to initialization of parameters

input_dim = 2; 
params['uncheckable_dist_weights'] = ['tn']
params['uncheckable_output_size'] = [input_dim,2]
params['uncheckable_epd'] = np.array([3])
params['uncheckable_act'] = ['none']
#params['uncheckable_act'] = ['elu','elu','elu','elu','elu']

params['uncheckable_com_type1'] = ['none']
params['uncheckable_com_type2'] = ['none']
params['uncheckable_dist_biases'] = ['none']
params['uncheckable_num_of_layers'] = len(np.array([0])) 

#params['checkable_dist_weights'] = ['tn','tn']
#params['checkable_output_size'] = [3,10,3]
#params['checkable_epd'] = np.array([2,0])
#params['checkable_act'] = ['elu']
#params['checkable_dist_biases'] = ['none']
#params['checkable_num_of_layers'] = len(np.array([0])) 

params['traj_len'] = 500   #must equate to batch size
params['Xwidth'] = 2
params['Ywidth'] = 2
params['lYwidth'] = 2
#params['lZwidth'] = 2

params['dynamics_lam'] = 1
#params['future_lam'] = 0.5
#params['L1_lam'] = 0.001

params['exp_name'] = 'exp_1'
params['folder_name'] = 'taylor_test_com_none'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])



params['number_of_data files_for_training'] = 5
params['num_passes_per_file'] = 100 * 8 * 100
params['batch_size'] = 500
params['num_steps_per_batch'] = 2
params['loops for val'] = 1



for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')