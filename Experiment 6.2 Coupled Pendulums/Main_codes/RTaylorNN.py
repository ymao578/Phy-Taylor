import tensorflow as tf
import numpy as np 
import PhyAct as pha

#=================================================================================================================
#=================================================================================================================
# ========================= All of the function definitions are below ============================================
#=================================================================================================================
#=================================================================================================================

##Construct TaylorNN ######################################################################################################################
def taylor_nn(prev_layer, weights, phyweights, unkweights, biases, inx_phy, inx_unk, com_type1, com_type2, act_type, num_of_layers,                       expansion_order, name ='U'):
    
    """Apply a NN to input from previous later

    Arguments:
        prev_layer      -- input from previous NN
        weights         -- dictionary of weights
        biases          -- dictionary of biases (uniform(-1,1) distribution, normal(0,1) distrubution, none--zeros)
        act_type        -- dictionary of activation functions (sigmoid, relu, elu, or none): user option 
        num_of_layers   -- number of weight matrices or layers: user option 
        expansion_order -- dictionary of Taylor expansion order: user option 

    Returns:
        output of network for input from previous layer
    """
    
    for i in np.arange(num_of_layers):
        
        ##linear mapping parameters
        aa = 0.001;
        bb = 100;
        ##
        
        #Compressor One###
        if com_type1['com1%s%d' % (name,i + 1)] == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif com_type1['com1%s%d' % (name,i + 1)] == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif com_type1['com1%s%d' % (name,i + 1)] == 'lin':
            prev_layer = tf.multiply(prev_layer, aa) + bb
        elif com_type1['com1%s%d' % (name,i + 1)] == 'none':
            prev_layer = prev_layer
        ###################################################################################################################################
        
        #Compressor Two###
        if com_type2['com2%s%d' % (name,i + 1)] == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif com_type2['com2%s%d' % (name,i + 1)] == 'none':
            prev_layer = prev_layer
        ###################################################################################################################################

        #save raw input###
        input_raw = prev_layer
        raw_input_shape = input_raw.shape
        ###################################################################################################################################
        
        #The expaned input via Taylor expansion is denoted by input_epd###
        input_epd = input_raw
        ###################################################################################################################################
        
        #Anxiliary index###
        Id = np.arange(raw_input_shape[0].value)
        ###################################################################################################################################
        
        #Nolinear mapping through Taylor expansion###
        for _ in range(expansion_order['E%s%d' % (name, i +  1)]):
            for j in range(raw_input_shape[0]):
                for q in range(raw_input_shape[1]):
                    
                    x_temp = tf.multiply(input_raw[j,q], input_epd[Id[j]:(Id[raw_input_shape[0]-1]+1),q])
                    x_temp = tf.expand_dims(x_temp,1)
                    if q == 0:
                        tem_temp = x_temp
                    else:
                        tem_temp = tf.concat((tem_temp,x_temp),1)
                Id[j] = input_epd.shape[0] 
                input_epd = tf.concat((input_epd,tem_temp),0)
        ###################################################################################################################################
          
        #Compute T.NN output###
        
        #pre_weight = tf.multiply(weights['W%s%d' % (name,i + 1)], unkweights['W%s%d' % (name,i + 1)]) + phyweights['W%s%d' % (name,i + 1)]
          
        pre_bias   = tf.multiply(biases['b%s%d' % (name,i + 1)], inx_unk['b%s%d' % (name,i + 1)])  
            
        pre_weight_A = phyweights['W%s%d' % (name,i + 1)]    
        pre_weight_B = tf.multiply(weights['W%s%d' % (name,i + 1)], unkweights['W%s%d' % (name,i + 1)])
        prev_layer_A = tf.matmul(pre_weight_A, input_epd) + inx_phy['b%s%d' % (name,i + 1)]
        prev_layer_B = tf.matmul(pre_weight_B, input_epd) + pre_bias
        
        if act_type['act%s%d' % (name,i + 1)]    == 'sigmoid':
            prev_layer_B = pha.pysigmoid(prev_layer_B, inx_unk['b%s%d' % (name,i + 1)])
        elif act_type['act%s%d' % (name,i + 1)]  == 'relu':
            prev_layer_B = pha.pyrelu(prev_layer_B, inx_unk['bk%s%d' % (name,i + 1)])
        elif act_type ['act%s%d' % (name,i + 1)] == 'elu':
            prev_layer_B = pha.pyelu(prev_layer_B, inx_unk['b%s%d' % (name,i + 1)])
        elif act_type['act%s%d' % (name,i + 1)] == 'none':
            prev_layer_B = prev_layer_B
            
        prev_layer = prev_layer_A + prev_layer_B
        #################################################################################################################################     
    #Returen final output of created DeepTaylor###
    return prev_layer
###########################################################################################################################################

##Initilize the Matrix and Biases##########################################################################################################
def initilization(widths, comT1, comT2, act, epd, dist_weights, dist_biases, phyweightsA, phyweightsB, phybiasesA, phybiasesB, 
                  scale, name='U'):
    """Create a decoder network: a dictionaries of weights, biases, activation function and expansion_order

    Arguments:
        widths       -- array or list of widths for layers of network
        comT1        -- list of compressor function 1
        comT2        -- list of compressor function 2
        act          -- list of string for activation functions
        epd          -- array of expansion order
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases  -- array or list of strings for distributions of bias vectors
        scale        -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        name         -- string for prefix on weight matrices (default 'D' for decoder)

    Returns:
        weights         -- dictionary of weights
        com_type1       -- dictionary of compressor function 1
        com_type2       -- dictionary of compressor function 2
        biases          -- dictionary of biases
        act_type        -- dictionary of activation functions
        expansion_order -- dictionary of expansion order
    """
    
    weights         = dict()
    biases          = dict()
    act_type        = dict()
    expansion_order = dict()
    com_type1       = dict()
    com_type2       = dict()
    phyweights      = dict() 
    unkweights      = dict()
    inx_phy         = dict()
    inx_unk         = dict()

    for i in np.arange(len(widths)):
        ind = i + 1
        weights['W%s%d' % (name, ind)]        = weight_variable(widths[i], var_name='W%s%d' % (name, ind),
                                                         distribution=dist_weights[ind - 1], scale=scale)
        biases['b%s%d' % (name, ind)]         = bias_variable([widths[i][0], 1], var_name='b%s%d' % (name, ind),
                                                       distribution=dist_biases[ind - 1])
        act_type['act%s%d' % (name,ind)]      = act[i]
        expansion_order['E%s%d' % (name,ind)] = epd[i]
        
        com_type1['com1%s%d' % (name,ind)]    = comT1[i]
        com_type2['com2%s%d' % (name,ind)]    = comT2[i]
        
        phyweights['W%s%d' % (name,ind)]   = phyweightsA[i]
        unkweights['W%s%d' % (name,ind)]   = phyweightsB[i]
        inx_phy['b%s%d' % (name,ind)]    = phybiasesA[i]
        inx_unk['b%s%d' % (name,ind)]    = phybiasesB[i]
        
    return weights, com_type1, com_type2, biases, act_type, expansion_order, phyweights, unkweights, inx_phy, inx_unk
###########################################################################################################################################

##Create the variable of weight matrix#####################################################################################################
def weight_variable(shape, var_name, distribution, scale=0.1):
    """Create a variable for a weight matrix.

    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)

    Returns:
        a TensorFlow variable for a weight matrix

    Raises ValueError if distribution is filename but shape of data in file does not match input shape
    """
    
    if distribution == 'tn':
        initial = tf.random.truncated_normal(shape, stddev=scale, dtype=tf.float32)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float32)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))         
    return tf.Variable(initial, name=var_name)
###########################################################################################################################################



##Create bias variable#####################################################################################################################
def bias_variable(shape, var_name, distribution):
    """Create a variable for a bias vector.

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')

    Returns:
        a TensorFlow variable for a bias vector
    """
    
    if distribution == 'uniform':
        initial = tf.random.uniform(shape, minval=-0.2, maxval=0.2, dtype=tf.float32)
    elif distribution == 'normal':
        initial = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    elif distribution == 'none':
        initial = tf.constant(1, shape=shape, dtype=tf.float32) 
    return tf.Variable(initial, name=var_name)
###########################################################################################################################################



##Layer Shape List Generation due to Taylor Mapping########################################################################################
def exp_length(output_size, epd):
    """Generate shape list of expanded layer.

    Arguments:
        output_size -- [input dimention, layer output size list]
        epd         -- layer expansion order list
    Returns:
        shape list of expanded layer  
    """
    
    layer_shape = np.zeros((len(epd),2)) #layer shape width
    for layer_index in range(len(output_size)-1):
        expansion_index = np.ones([output_size[layer_index],1])  #expansion index
        EP_length = np.sum(expansion_index) #expansion length
        if epd[layer_index] >= 1:
            for ed in range(epd[layer_index]):
                for g in range(output_size[layer_index]):
                    expansion_index[g] = np.sum(expansion_index[g:(output_size[layer_index])])
                EP_length = np.sum(expansion_index) + EP_length
            
        layer_shape[layer_index,0] = output_size[layer_index+1]
        layer_shape[layer_index,1] = EP_length
    return layer_shape
###########################################################################################################################################



##Create DeepTaylor for a detailed problem################################################################################################
def create_DeepTaylor_net(params):
    
    """Create a DeepTaylor that consists of uncheckable and check models in order

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        x       --  placeholder for input
        y       --  output of uncheckable model, e.g., y = [x(k+1); u(k+1)]
        z       --  output of checkable model, e.g., z = f(y(k+m))
        ly      --  labels of uncheckable model
        lz      --  labels of checkable model
        weights --  dictionary of weights
        biases  --  dictionary of biases
    """
    
    ##Placeholder for the input, output, lables. Must be in tf.float32, due to constraint optimizator!!!!!!!!##############################
    x = tf.compat.v1.placeholder(tf.float32, [params['Xwidth'],params['traj_len']])
    ly = tf.compat.v1.placeholder(tf.float32, [params['lYwidth'],params['traj_len']])
    #######################################################################################################################################
    
    UN_widths = exp_length(output_size=params['uncheckable_output_size'], epd=params['uncheckable_epd'])
    UN_widths = UN_widths.astype(np.int64)
    
    weights, com_type1, com_type2, biases, act_type, expansion_order, phyweights, unkweights, inx_phy, inx_unk =                                     initilization(widths=UN_widths, comT1=params['uncheckable_com_type1'], comT2=params['uncheckable_com_type2'],                                             act=params['uncheckable_act'], epd=params['uncheckable_epd'], dist_weights=params['uncheckable_dist_weights'],                             dist_biases=params['uncheckable_dist_biases'], phyweightsA=params['uncheckable_phyweightsA'],
                        phyweightsB=params['uncheckable_phyweightsB'], phybiasesA=params['uncheckable_phybiasesA'],                                                 phybiasesB=params['uncheckable_phybiasesB'], scale = 0.1, name='U')
    
    y = taylor_nn(prev_layer=x, weights=weights, phyweights=phyweights, unkweights=unkweights, biases=biases, inx_phy=inx_phy,                               inx_unk=inx_unk,  com_type1=com_type1, com_type2=com_type2, act_type=act_type,                                                           num_of_layers=params['uncheckable_num_of_layers'], expansion_order=expansion_order,name='U')
    
    return x, y, ly, weights, biases
###########################################################################################################################################   