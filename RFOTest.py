import os
import time
import copy
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import helperfns
import RTaylorNN as tnn
import Rtraining
import PhyAct as pha

params = {}
input_dim = 6; 
out1 = 50;
out2 = 10;

params['data_name'] = 'Car_test'
params['seed'] = 10000000
params['uncheckable_dist_weights'] = ['tn','tn','tn']
params['uncheckable_output_size'] = [input_dim,out1,out2,6]
params['uncheckable_epd'] = np.array([0,0,1])
params['uncheckable_act'] = ['elu','elu','elu']
params['uncheckable_com_type1'] = ['none','none','none']
params['uncheckable_com_type2'] = ['none','none','none']
params['uncheckable_dist_biases'] = ['normal','normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0,0])) 

##embed physical knowledge#################################################################################################
##layer 1##
Phy_lay1_A = np.zeros((out1,input_dim), dtype=np.float32)
Phy_lay1_B = np.ones((out1,input_dim), dtype=np.float32)
phyBias_lay1_A = np.zeros((out1,1), dtype=np.float32)
phyBias_lay1_B = np.ones((out1,1), dtype=np.float32)

Phy_lay1_A[0][0] = 1;
Phy_lay1_A[0][3] = 0.005;
Phy_lay1_A[1][1] = 1;
Phy_lay1_A[1][4] = 0.005;
Phy_lay1_A[2][2] = 1;
Phy_lay1_A[2][5] = 0.005;

Phy_lay1_B[0]    = 0;
Phy_lay1_B[1]    = 0;
Phy_lay1_B[2]    = 0;

phyBias_lay1_A[0:3] = 1
phyBias_lay1_B[0:3] = 0
###########

##Layer2##
Phy_lay2_A = np.zeros((out2,out1), dtype=np.float32)
Phy_lay2_B = np.ones((out2,out1), dtype=np.float32)
phyBias_lay2_A = np.zeros((out2,1), dtype=np.float32)
phyBias_lay2_B = np.ones((out2,1), dtype=np.float32)


Phy_lay2_A[0][0] = 1;
Phy_lay2_A[1][1] = 1;
Phy_lay2_A[2][2] = 1;

Phy_lay2_B[0] = 0;
Phy_lay2_B[1] = 0;
Phy_lay2_B[2] = 0;

phyBias_lay2_A[0:3] = 1
phyBias_lay2_B[0:3] = 0
###########

##Layer3##
Phy_lay3_A = np.zeros((6,65), dtype=np.float32)
Phy_lay3_B = np.ones((6,65), dtype=np.float32)

phyBias_lay3_A = np.zeros((6,1), dtype=np.float32)
phyBias_lay3_B = np.ones((6,1), dtype=np.float32)

Phy_lay3_A[0][0] = 1;
Phy_lay3_A[1][1] = 1;
Phy_lay3_A[2][2] = 1;

Phy_lay3_B[0] = 0;
Phy_lay3_B[1] = 0;
Phy_lay3_B[2] = 0;

phyBias_lay3_A[0:3] = 1
phyBias_lay3_B[0:3] = 0
###########

params['uncheckable_phyweightsA'] = [Phy_lay1_A, Phy_lay2_A, Phy_lay3_A]
params['uncheckable_phyweightsB'] = [Phy_lay1_B, Phy_lay2_B, Phy_lay3_B]
params['uncheckable_phybiasesA'] = [phyBias_lay1_A, phyBias_lay2_A, phyBias_lay3_A]
params['uncheckable_phybiasesB'] = [phyBias_lay1_B, phyBias_lay2_B, phyBias_lay3_B]
###########################################################################################################################

params['traj_len'] = 150   #must equate to batch size
params['Xwidth'] = 6
params['Ywidth'] = 6
params['lYwidth'] = 6
params['dynamics_lam'] = 1;

params['exp_name'] = 'exp'
params['folder_name'] = 'car'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])

params['number_of_data files_for_training'] = 5
params['num_passes_per_file'] = 1000 * 8 * 1000
params['batch_size'] = 150
params['num_steps_per_batch'] = 2
params['loops for val'] = 1

for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')