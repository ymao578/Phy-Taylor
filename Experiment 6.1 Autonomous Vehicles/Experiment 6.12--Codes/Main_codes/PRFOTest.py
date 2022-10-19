import os
import time

import copy
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco

import helperfns
import PRTaylorNN as tnn
import PRtraining

params = {}

# settings related to dataset
params['data_name'] = 'Car_test'
#n = 1  # dimension of system (and input layer)

params['seed'] = 50

# defaults related to initialization of parameters

input_dim = 7; 

params['uncheckable_dist_weights'] = ['tn','tn','tn']
params['uncheckable_output_size'] = [input_dim,350,10,2]
params['uncheckable_epd'] = np.array([0,0,1])
params['uncheckable_act'] = ['elu','elu','none']
params['uncheckable_com_type1'] = ['none','none','none']
params['uncheckable_com_type2'] = ['none','none','none']
params['uncheckable_dist_biases'] = ['normal','normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0,0])) 


#params['uncheckable_phyweightsB'] = tf.convert_to_tensor(Phy_lay_B, dtype=tf.float32)


params['checkable_dist_weights'] = ['tn']
params['checkable_output_size'] = [2,2]
params['checkable_epd'] = np.array([1])

Phy_lay_B = np.ones((2,5), dtype=np.float32)
Phy_lay_B[0][0] = 0;
Phy_lay_B[0][1] = 0;
Phy_lay_B[1][0] = 0;
Phy_lay_B[1][1] = 0;
params['uncheckable_phyweightsB'] = Phy_lay_B

params['checkable_act'] = ['none']
params['checkable_com_type1'] = ['none']
params['checkable_com_type2'] = ['none']
params['checkable_dist_biases'] = ['none']
params['checkable_num_of_layers'] = len(np.array([0])) 

params['traj_len'] = 200   #must equate to batch size
params['Xwidth'] = 7
params['Ywidth'] = 2
params['Zwidth'] = 2
params['lYwidth'] = 2
params['lZwidth'] = 2

params['dynamics_lam'] = 1;
params['future_lam'] = 4;
#params['L1_lam'] = 0.001

params['exp_name'] = 'exp_1'
params['folder_name'] = 'Ptaylor_car_taylor50'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])


params['number_of_data files_for_training'] = 2
params['num_passes_per_file'] = 1000 * 8 * 1000
params['batch_size'] = 200
params['num_steps_per_batch'] = 2
params['loops for val'] = 1



for count in range(1):     
     PRtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')