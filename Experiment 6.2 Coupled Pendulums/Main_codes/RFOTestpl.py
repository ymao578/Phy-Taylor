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

Phy_lay1_B[3][3]    = 1;
Phy_lay1_B[3][4]    = 1;
Phy_lay1_B[3][9]    = 1;
Phy_lay1_B[3][10]    = 1;
Phy_lay1_B[3][14]    = 1;
Phy_lay1_B[3][15]    = 1;
Phy_lay1_B[3][21]    = 1;
Phy_lay1_B[3][22]    = 1;
Phy_lay1_B[3][24]    = 1;
Phy_lay1_B[3][30]    = 1;
Phy_lay1_B[3][31]    = 1;
Phy_lay1_B[3][35]    = 1;
Phy_lay1_B[3][36]    = 1;
Phy_lay1_B[3][42]    = 1;
Phy_lay1_B[3][43]    = 1;
Phy_lay1_B[3][45]    = 1;
Phy_lay1_B[3][50]    = 1;
Phy_lay1_B[3][51]    = 1;
Phy_lay1_B[3][57]    = 1;
Phy_lay1_B[3][58]    = 1;
Phy_lay1_B[3][60]    = 1;
Phy_lay1_B[3][73]    = 1;
Phy_lay1_B[3][74]    = 1;
Phy_lay1_B[3][76]    = 1;
Phy_lay1_B[3][79]    = 1;
Phy_lay1_B[3][86]    = 1;
Phy_lay1_B[3][87]    = 1;
Phy_lay1_B[3][91]    = 1;
Phy_lay1_B[3][92]    = 1;
Phy_lay1_B[3][98]    = 1;
Phy_lay1_B[3][99]    = 1;
Phy_lay1_B[3][104]    = 1;
Phy_lay1_B[3][106]    = 1;
Phy_lay1_B[3][107]    = 1;
Phy_lay1_B[3][113]    = 1;
Phy_lay1_B[3][114]    = 1;
Phy_lay1_B[3][116]    = 1;
Phy_lay1_B[3][129]    = 1;
Phy_lay1_B[3][130]    = 1;
Phy_lay1_B[3][132]    = 1;
Phy_lay1_B[3][135]    = 1;
Phy_lay1_B[3][141]    = 1;
Phy_lay1_B[3][142]    = 1;
Phy_lay1_B[3][148]    = 1;
Phy_lay1_B[3][149]    = 1;
Phy_lay1_B[3][151]    = 1;
Phy_lay1_B[3][164]    = 1;
Phy_lay1_B[3][165]    = 1;
Phy_lay1_B[3][167]    = 1;
Phy_lay1_B[3][170]    = 1;
Phy_lay1_B[3][194]    = 1;
Phy_lay1_B[3][195]    = 1;
Phy_lay1_B[3][197]    = 1;
Phy_lay1_B[3][200]    = 1;
Phy_lay1_B[3][204]    = 1;
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

Phy_lay1_B[4][0:209]    = 1;


Phy_lay1_B[5][4]    = 1;
Phy_lay1_B[5][5]    = 1;
Phy_lay1_B[5][15]    = 1;
Phy_lay1_B[5][16]    = 1;
Phy_lay1_B[5][19]    = 1;
Phy_lay1_B[5][20]    = 1;
Phy_lay1_B[5][24]    = 1;
Phy_lay1_B[5][25]    = 1;
Phy_lay1_B[5][26]    = 1;
Phy_lay1_B[5][51]    = 1;
Phy_lay1_B[5][52]    = 1;
Phy_lay1_B[5][55]    = 1;
Phy_lay1_B[5][56]    = 1;
Phy_lay1_B[5][60]    = 1;
Phy_lay1_B[5][61]    = 1;
Phy_lay1_B[5][62]    = 1;
Phy_lay1_B[5][65]    = 1;
Phy_lay1_B[5][66]    = 1;
Phy_lay1_B[5][70]    = 1;
Phy_lay1_B[5][71]    = 1;
Phy_lay1_B[5][72]    = 1;
Phy_lay1_B[5][79]    = 1;
Phy_lay1_B[5][80]    = 1;
Phy_lay1_B[5][81]    = 1;
Phy_lay1_B[5][82]    = 1;
Phy_lay1_B[5][143]    = 1;
Phy_lay1_B[5][146]    = 1;
Phy_lay1_B[5][147]    = 1;
Phy_lay1_B[5][151]    = 1;
Phy_lay1_B[5][152]    = 1;
Phy_lay1_B[5][156]    = 1;
Phy_lay1_B[5][157]    = 1;
Phy_lay1_B[5][161]    = 1;
Phy_lay1_B[5][162]    = 1;
Phy_lay1_B[5][163]    = 1;
Phy_lay1_B[5][170]    = 1;
Phy_lay1_B[5][171]    = 1;
Phy_lay1_B[5][172]    = 1;
Phy_lay1_B[5][173]    = 1;
Phy_lay1_B[5][176]    = 1;
Phy_lay1_B[5][177]    = 1;
Phy_lay1_B[5][181]    = 1;
Phy_lay1_B[5][182]    = 1;
Phy_lay1_B[5][183]    = 1;
Phy_lay1_B[5][190]    = 1;
Phy_lay1_B[5][191]    = 1;
Phy_lay1_B[5][192]    = 1;
Phy_lay1_B[5][193]    = 1;
Phy_lay1_B[5][204]    = 1;
Phy_lay1_B[5][205]    = 1;
Phy_lay1_B[5][206]    = 1;
Phy_lay1_B[5][207]    = 1;
Phy_lay1_B[5][208]    = 1;
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
params['folder_name'] = 'oscillaor_none_un_degree'
params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])

params['number_of_data files_for_training'] = 40
params['num_passes_per_file'] = 1000 * 8 * 1000
params['batch_size'] = 200
params['num_steps_per_batch'] = 2
params['loops for val'] = 1

for count in range(1):     
     Rtraining.main_exp(copy.deepcopy(params))
print('Done Done Done')