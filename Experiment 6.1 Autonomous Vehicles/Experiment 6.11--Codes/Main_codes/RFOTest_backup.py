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
out1 = 15;

params['data_name'] = 'Car_test'
params['seed'] = 9
params['uncheckable_dist_weights'] = ['tn','tn']
params['uncheckable_output_size'] = [input_dim,out1,6]
params['uncheckable_epd'] = np.array([3,0])
params['uncheckable_act'] = ['elu','none']
params['uncheckable_com_type1'] = ['none','none']
params['uncheckable_com_type2'] = ['none','none']
params['uncheckable_dist_biases'] = ['normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0])) 

##embed physical knowledge#################################################################################################

##sampling period
T = 0.005;
##

##layer 1##
a_input_dim = 209;

Phy_lay1_A = np.zeros((out1,a_input_dim), dtype=np.float32)
Phy_lay1_B = np.zeros((out1,a_input_dim), dtype=np.float32)
phyBias_lay1_A = np.zeros((out1,1), dtype=np.float32)
phyBias_lay1_B = np.ones((out1,1), dtype=np.float32)

Phy_lay1_A[0][0] = 1;
Phy_lay1_A[0][3] = T;
Phy_lay1_A[1][1] = 1;
Phy_lay1_A[1][4] = T;
Phy_lay1_A[2][2] = 1;
Phy_lay1_A[2][5] = T;
Phy_lay1_A[3][3] = 1;
Phy_lay1_A[4][4] = 1;
Phy_lay1_A[5][5] = 1;


Phy_lay1_B[3][0]      = 1;
Phy_lay1_B[3][3]      = 1;
Phy_lay1_B[3][6]      = 1;
Phy_lay1_B[3][9]      = 1;
Phy_lay1_B[3][21]     = 1;
Phy_lay1_B[3][27]     = 1;
Phy_lay1_B[3][30]     = 1;
Phy_lay1_B[3][42]     = 1;
Phy_lay1_B[3][73]     = 1;
Phy_lay1_B[3][83]     = 1;
Phy_lay1_B[3][86]     = 1;
Phy_lay1_B[3][98]     = 1;
Phy_lay1_B[3][129]     = 1;
Phy_lay1_B[3][194]     = 1;

Phy_lay1_B[4]         = 1;
Phy_lay1_B[5]         = 1;
Phy_lay1_B[6]         = 1;
Phy_lay1_B[7]         = 1;
Phy_lay1_B[8]         = 1;
Phy_lay1_B[9]         = 1;
Phy_lay1_B[10]         = 1;
Phy_lay1_B[11]         = 1;
Phy_lay1_B[12]         = 1;
Phy_lay1_B[13]         = 1;
Phy_lay1_B[14]         = 1;

phyBias_lay1_B[0:3] = 0
###########

##Layer2##
Phy_lay2_A = np.zeros((6,out1), dtype=np.float32)
Phy_lay2_B = np.ones((6,out1), dtype=np.float32)
phyBias_lay2_A = np.zeros((6,1), dtype=np.float32)
phyBias_lay2_B = np.ones((6,1), dtype=np.float32)

Phy_lay2_A[0][0] = 1;
Phy_lay2_A[1][1] = 1;
Phy_lay2_A[2][2] = 1;
Phy_lay2_A[3][3] = 1;

Phy_lay2_B[0] = 0;
Phy_lay2_B[1] = 0;
Phy_lay2_B[2] = 0;
Phy_lay2_B[3] = 0;

phyBias_lay2_B[0:3] = 0
###########

params['uncheckable_phyweightsA'] = [Phy_lay1_A, Phy_lay2_A]
params['uncheckable_phyweightsB'] = [Phy_lay1_B, Phy_lay2_B]
params['uncheckable_phybiasesA'] = [phyBias_lay1_A, phyBias_lay2_A]
params['uncheckable_phybiasesB'] = [phyBias_lay1_B, phyBias_lay2_B]
###########################################################################################################################

params['traj_len'] = 200   #must equate to batch size
params['Xwidth'] = 6
params['Ywidth'] = 6
params['lYwidth'] = 6
params['dynamics_lam'] = 1;


params['exp_name'] = 'exp'
params['folder_name'] = 'EmdPhySinone9'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])

params['number_of_data files_for_training'] = 4
params['num_passes_per_file'] = 1000 * 8 * 1000
params['batch_size'] = 200
params['num_steps_per_batch'] = 2
params['loops for val'] = 1

for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')