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
out1 = 6;

params['data_name'] = 'Oscillators'
params['seed'] = 10000000
params['uncheckable_dist_weights'] = ['tn','tn']
params['uncheckable_output_size'] = [input_dim,out1,6]
params['uncheckable_epd'] = np.array([3,0])
params['uncheckable_act'] = ['none','none']
params['uncheckable_com_type1'] = ['none','none']
params['uncheckable_com_type2'] = ['none','none']
params['uncheckable_dist_biases'] = ['normal','normal']
params['uncheckable_num_of_layers'] = len(np.array([0,0])) 

##embed physical knowledge#################################################################################################

##layer 1##========================================================================================
a_input_dim = 209;  #number of nodes after expansion

T = 0.005

Phy_lay1_A = np.zeros((out1,a_input_dim), dtype=np.float32)  #knowledge matrix
Phy_lay1_B = np.zeros((out1,a_input_dim), dtype=np.float32)  #learning parameter indication matrix

phyBias_lay1_A = np.zeros((out1,1), dtype=np.float32)  #knowledge bias
phyBias_lay1_B = np.ones((out1,1), dtype=np.float32)   #learning parameter indication bias

Phy_lay1_A[0][0] = 1;
Phy_lay1_A[1][1] = 1;
Phy_lay1_A[2][2] = 1;
Phy_lay1_A[3][3] = 1;
Phy_lay1_A[4][4] = 1;
Phy_lay1_A[5][5] = 1;


Phy_lay1_B[0][3]    = 1;
Phy_lay1_B[1][4]    = 1;
Phy_lay1_B[2][5]    = 1;
Phy_lay1_B[3][0]    = 1;
Phy_lay1_B[3][1]    = 1;
Phy_lay1_B[3][6]    = 1;
Phy_lay1_B[3][7]    = 1;
Phy_lay1_B[3][12]    = 1;
Phy_lay1_B[3][27]    = 1;
Phy_lay1_B[3][28]    = 1;
Phy_lay1_B[3][33]    = 1;
Phy_lay1_B[3][48]    = 1;
Phy_lay1_B[3][83]    = 1;
Phy_lay1_B[3][84]    = 1;
Phy_lay1_B[3][89]    = 1;
Phy_lay1_B[3][104]    = 1;
Phy_lay1_B[3][139]    = 1;
Phy_lay1_B[4][0]    = 1;
Phy_lay1_B[4][1]    = 1;
Phy_lay1_B[4][2]    = 1;
Phy_lay1_B[4][6]    = 1;
Phy_lay1_B[4][7]    = 1;
Phy_lay1_B[4][8]    = 1;
Phy_lay1_B[4][12]    = 1;
Phy_lay1_B[4][13]    = 1;
Phy_lay1_B[4][17]    = 1;
Phy_lay1_B[4][27]    = 1;
Phy_lay1_B[4][28]    = 1;
Phy_lay1_B[4][29]    = 1;
Phy_lay1_B[4][33]    = 1;
Phy_lay1_B[4][34]    = 1;
Phy_lay1_B[4][38]    = 1;
Phy_lay1_B[4][48]    = 1;
Phy_lay1_B[4][49]    = 1;
Phy_lay1_B[4][53]    = 1;
Phy_lay1_B[4][63]    = 1;
Phy_lay1_B[4][83]    = 1;
Phy_lay1_B[4][84]    = 1;
Phy_lay1_B[4][85]    = 1;
Phy_lay1_B[4][89]    = 1;
Phy_lay1_B[4][90]    = 1;
Phy_lay1_B[4][94]    = 1;
Phy_lay1_B[4][104]    = 1;
Phy_lay1_B[4][105]    = 1;
Phy_lay1_B[4][109]    = 1;
Phy_lay1_B[4][119]    = 1;
Phy_lay1_B[4][139]    = 1;
Phy_lay1_B[4][140]    = 1;
Phy_lay1_B[4][144]    = 1;
Phy_lay1_B[4][154]    = 1;
Phy_lay1_B[4][174]    = 1;
Phy_lay1_B[5][1]    = 1;
Phy_lay1_B[5][2]    = 1;
Phy_lay1_B[5][12]    = 1;
Phy_lay1_B[5][13]    = 1;
Phy_lay1_B[5][17]    = 1;
Phy_lay1_B[5][48]    = 1;
Phy_lay1_B[5][49]    = 1;
Phy_lay1_B[5][53]    = 1;
Phy_lay1_B[5][63]    = 1;
Phy_lay1_B[5][139]    = 1;
Phy_lay1_B[5][140]    = 1;
Phy_lay1_B[5][144]    = 1;
Phy_lay1_B[5][154]    = 1;
Phy_lay1_B[5][174]    = 1;


phyBias_lay1_B[0:3] = 0
###########===========================================================================================================

##Layer2##============================================================================================================
Phy_lay2_A = np.zeros((6,out1), dtype=np.float32)
Phy_lay2_B = np.zeros((6,out1), dtype=np.float32)
phyBias_lay2_A = np.zeros((6,1), dtype=np.float32)
phyBias_lay2_B = np.ones((6,1), dtype=np.float32)

Phy_lay2_A[0][0] = 1;
Phy_lay2_A[1][1] = 1;
Phy_lay2_A[2][2] = 1;
Phy_lay2_A[3][3] = 1;
Phy_lay2_A[4][4] = 1;
Phy_lay2_A[5][5] = 1;

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
params['folder_name'] = 'oscillaor_none_partial_degree'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])

params['number_of_data files_for_training'] = 40
params['num_passes_per_file'] = 1000 * 8 * 1000
params['batch_size'] = 200
params['num_steps_per_batch'] = 2
params['loops for val'] = 1

for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')