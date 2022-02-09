import os
import time
import pickle

import numpy as np
import tensorflow as tf

import tensorflow_constrained_optimization as tfco

#import TaylorNN as tnn

import RTaylorNN as tnn


#Fromulate Consraint Optimazation
class ExampleProblem(tfco.ConstrainedMinimizationProblem):
    
    def __init__(self,y, ly, params, upper_bound):
#         self._x = x
        self._y = y
        self._ly = ly
        self._params = params
        self._upper_bound = upper_bound
        
    #Note: the number of constaint must be element-wise, cannot be vector- or matrix-wise!!!!!!!!!!!!!    
    @property 
    def num_constraints(self):
        return 1    

    def objective(self):
    # In eager mode, the predictions must be a nullary function returning a
    # Tensor. In graph mode, they could be either such a function, or a Tensor
    # itself.
    
        return define_loss(self._y, self._ly, self._params, self._trainable_var)


    def constraints(self):   
       # In eager mode, the predictions must be a nullary function returning a
       # Tensor. In graph mode, they could be either such a function, or a Tensor
       # itself.
       # The constraint is (recall >= self._recall_lower_bound), which we convert
       # to (self._recall_lower_bound - recall <= 0) because
       # ConstrainedMinimizationProblems must always provide their constraints in
       # the form (tensor <= 0).
       #
       # The result of this function should be a tensor, with each element being
       # a quantity that is constrained to be non-positive. We only have one
       # constraint, so we return a one-element tensor.
        
#         print('=============')
#         print(self._z[0][0])
#         print(type(self._upper_bound))
#         print('===============')
        
        return self._z[0][0] - self._upper_bound




#Define loss funcions for training
def define_loss(y, ly,params,trainable_var):
    """Define the (unregularized) loss functions for the training.

    Arguments:
       import os
import time

import numpy as np
import tensorflow as tf

import tensorflow_constrained_optimization as tfco

import TaylorNN as tnn
print(tf.__version__) x  -- placeholder for input: x(k)
        y  -- output of uncheckable model, e.g., y = [x(k+1); u(k+1)]: 
        z  -- output of checkable model for prediction: e.g., z = f(y(k+m))
        
        ly -- label of y
        lz -- lable of z
        
    Returns:
        loss1 -- supervised dynamics loss
        loss2 -- supervised future loss
        loss --  sum of above two losses

    Side effects:
        None
    """
    

    # Embedding dynamics into uncheck model, learning via prediction
    loss1 = params['dynamics_lam']*tf.reduce_mean(tf.square(y-ly))
    #loss2 = params['future_lam']*tf.reduce_mean(tf.square(z-lz))
    
    #if params['dynamics_lam']:
     #   l1_regularizer = tf.contrib.layers.l1_regularizer(scale=params['L1_lam'], scope=None)
        # TODO: don't include biases? use weights dict instead?
      #  loss_L1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=trainable_var)
    #else:
    #    loss_L1 = tf.zeros([1, ], dtype=tf.float64)
    
    loss = loss1
    
    
    # Embedding future performace into check model, learning via prediction at long future
    #loss2 = params['future_lam']*tf.reduce_mean(tf.square(z-lz))
    
    #loss = loss1
    
    return loss

#====================================================================================
def save_files(sess, csv_path, params, weights, biases):
    """Save error files, weights, biases, and parameters.

    Arguments:
        sess -- TensorFlow session
        csv_path -- string for path to save error file as csv
        train_val_error -- table of training and validation errors
        params -- dictionary of parameters for experiment
        weights -- dictionary of weights for all networks
        biases -- dictionary of biases for all networks

    Returns:
        None (but side effect of saving files and updating params dict.)

    Side effects:
        Save train_val_error, each weight W, each bias b, and params dict to file.
        Update params dict: minTrain, minTest, minRegTrain, minRegTest
    """

    for key, value in weights.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    for key, value in biases.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
        
    save_params(params)
    
###########################################################################################################################################
    
def save_params(params):
    """Save parameter dictionary to file.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Saves params dict to pkl file
    """
    with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
#=======================================================================================================       
def try_exp(params):
    '''Initilize Taylor NN'''
    x, y, ly, weights, biases = tnn.create_DeepTaylor_net(params)

    '''return a list of all the trainable variables'''
    
    trainable_var = tf.trainable_variables()

    #upper_bound = 2.0
    
    loss = define_loss(y, ly, params, trainable_var)
    
    #my_problem = ExampleProblem(y=y, z=z, ly=ly, lz=lz, params = params, upper_bound = upper_bound)

    
    #optimizer = tf.train.AdagradDAOptimizer(learning_rate=0.000001)
    
    #optimizer = tfco.ProxyLagrangianOptimizerV1(
    #      optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate= 0.0001))

    # double check why variable list?
    #train_op = optimizer.minimize(my_problem.objective(), var_list = trainable_var)

   
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #train_op = optimizer.minimize(my_problem.objective(), var_list = trainable_var)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)

    #optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.001)
    
    #optimizer = tfco.ProxyLagrangianOptimizerV1(
    #      optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate= 0.0001))

    # double check why variable list?
    
    train_op = optimizer.minimize(loss, var_list = trainable_var)




    
    # Launch graph and initialize
    sess = tf.Session()
    saver = tf.train.Saver()

    # Initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Path to csv file that saves training data
    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')
    #print(csv_path)


    count = 0
    best_error = 10000
   
    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])
    
    #valx = np.loadtxt(('./data/%s_valX.csv' % (params['data_name'])), delimiter=',',
    #                                    dtype=np.float32)
    #valy = np.loadtxt(('./data/%s_valY.csv' % (params['data_name'])), delimiter=',',
    #                                    dtype=np.float32)
    ## = np.loadtxt(('./data/%s_valz.csv' % (params['data_name'])), delimiter=',',
    #                                    dtype=np.float32)
    
    #valx = valx[0:params['batch_size'],:]
    #valy = valy[0:params['batch_size'],:]
    #valz = valz[0:params['batch_size'],:]
    
    #valx = valx.reshape([params['val_size'],params['Xwidth']])
    #valy = valy.reshape([params['val_size'],params['lYwidth']])
    #valz = valz.reshape([params['val_size'],params['lZwidth']])

 
    for f in range(params['number_of_data files_for_training'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['number_of_data files_for_training']) + 1  # 1...data_train_len


        if (params['number_of_data files_for_training'] > 1) or (f == 0):  # don't keep reloading data if always same; 
            
            # load the raw data of total data_traint_len files################################################################# 
            data_train_x = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], file_num)), delimiter=',',
                                        dtype=np.float32)
            data_train_ly = np.loadtxt(('./data/%s_train%d_y.csv' % (params['data_name'], file_num)), delimiter=',',
                                        dtype=np.float32)
            #data_train_lz = np.loadtxt(('./data/%s_lz%d.csv' % (params['data_name'], file_num)), delimiter=',',
            #                            dtype=np.float32)

            #Total len in 1 training file
            total_length = data_train_x.shape[0]
            num_batches = int(np.floor(total_length / params['batch_size']))
            
         
        ind = np.arange(total_length)
        np.random.shuffle(ind)

        data_train_x  = data_train_x[ind, :]
        data_train_ly = data_train_ly[ind, :]
        #data_train_lz = data_train_lz[ind, :]
        
        #LEN = data_train_x.shape[0]
        #data_train_x = data_train_x.reshape([LEN,params['Xwidth']])
        #data_train_ly = data_train_lz.reshape([LEN,params['lYwidth']])
        #data_train_lz = data_train_lz.reshape([LEN,params['lZwidth']])

        for step in range(params['num_steps_per_batch'] * num_batches):
            if params['batch_size'] < data_train_x.shape[0]:
                offset = (step * params['batch_size']) % (total_length - params['batch_size'])
            else:
                offset = 0 
            batch_data_train_x = data_train_x[offset:(offset + params['batch_size']), :]
            batch_data_train_ly = data_train_ly[offset:(offset + params['batch_size']), :]
            #batch_data_train_lz = data_train_lz[offset:(offset + params['batch_size']), :]
            
            feed_dict_train = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
            feed_dict_train_loss = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
            #feed_dict_val = {x: np.transpose(valx), ly: np.transpose(valy)}
        
        
            ##traning######################################################################################################################
            sess.run(train_op, feed_dict=feed_dict_train)
            ###############################################################################################################################

            if step % params['loops for val'] == 0:
                train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                #val_error = sess.run(loss, feed_dict=feed_dict_val)
                
                #Error = [train_error, val_error]
                
                print(train_error)
                
                count = count + 1
                
                save_files(sess, csv_path, params, weights, biases)
 
            
    saver.restore(sess, params['model_path'])
    save_files(sess, csv_path,params, weights, biases)
    tf.reset_default_graph()
    
def main_exp(params):
    tf.compat.v1.set_random_seed(params['seed'])
    np.random.seed(params['seed'])
    try_exp(params)