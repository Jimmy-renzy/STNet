'''
Implementation of communication function:
    -Impulsive Channel
    -Modulation BPSK
    -Add Noise(Gaussian & Impulsive)

Author: Lucyyang
'''

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.stats import levy_stable
import random


def CalScale(GSNR,alpha,R):
    '''
    To calculate the parameter - scale from given alpha and GSNR
    according to Symmertic Alpha-Stable Distribution
    :param GSNR: Geometry SNR
    :param alpha: Characteristic exponent
    :param R: Code Rate
    :return: scale parameter - gamma
    '''
    GSNR = 10 ** (GSNR / 10)  # Eb/No conversion from dB to decimal
    S0 = 1 / (np.sqrt(7.12 * GSNR * R))
    gamma = ((1.78 * S0) ** alpha) / 1.78
    scale = gamma ** (1 / alpha)
    return scale

# Channel (Impluse Noise)
def Inoise(alpha_train,loc,scale,size):
    # rvs(alpha, beta, loc=0, scale=1, size=1, random_state=None)
    y = np.float32(levy_stable.rvs(alpha_train,0,loc,scale,size))
    return y

# Channel (AWGN)
# y_test = s_test + sigmas[i]*np.random.standard_normal(s_test.shape)

## BPSK layer
def modulateBPSK(x):
    return -2*x + 1

def addNoise(x,sigma,alpha_train):
    '''
    Add noise (Gaussian and Impulsive)
    :param x:
    :param sigma:
    :return:
    '''
    # tf.py_func接收的是tensor，然后将其转化为numpy array送入func函数，最后再将func函数输出的numpy array转化为tensor返回。
    w=tf.py_func(Inoise, [alpha_train, 0, sigma, K.shape(x)], tf.float32)
    w.set_shape(x.get_shape())
    return x + w


# Channel (Rayleigh Noise)
def Rnoise(SNR, size):
    '''
    Generate Rayleigh Noise
    :param SNR:
    :param size: [256, 16]
    '''
    sigma = np.sqrt(0.5 / np.power(10, SNR / 10))
    n = np.zeros(size)
    row = size[0]
    col = size[1]
    for i in range(row):
        for j in range(col):
            p = random.random()
            n[i][j] = sigma * np.sqrt(2 * np.log(1 / (1 - p)))
    return np.float32(n)


def addRayleighNoise(x, SNR):
    '''
    Add Rayleigh Noise
    :param x:
    :param SNR:
    :return:
    '''
    # tf.py_func接收的是tensor，然后将其转化为numpy array送入func函数，最后再将func函数输出的numpy array转化为tensor返回。
    w = tf.py_func(Rnoise, [SNR, K.shape(x)], tf.float32)
    w.set_shape(x.get_shape())
    return x + w


def return_output_shape(input_shape):
    return input_shape


if __name__ == '__main__':
    ## Check the calculation and result
    alpha_train=2
    R=1/2
    GSNR=1
    scale=CalScale(GSNR,alpha_train,R)
    print(scale)
