import tensorflow as tf
import numpy as np 

##elu#######################################################################################################################################
def pyelu(x, inx_unk):
    
    ex_unk = tf.nn.elu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################

##relu######################################################################################################################################
def pyrelu(x, inx_unk):
    
    ex_unk = tf.nn.relu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################

##sigmoid###################################################################################################################################
def pysigmoid(x, inx_unk):
    
    ex_unk = tf.nn.sigmoid(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################

##tanh######################################################################################################################################
def pytanh(x, inx_unk):
    
    ex_unk = tf.nn.tanh(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################

##softsign##################################################################################################################################
def pysoftsign(x, inx_unk):
    
    ex_unk = tf.nn.softsign(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################

##selu######################################################################################################################################
def pyselu(x, inx_unk):
    
    ex_unk = tf.nn.selu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    
    return ex
############################################################################################################################################












