import os
import sys
import pyldpc
import numpy as np
from scipy.stats import levy_stable
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN
from keras.layers.core import Dense
from time import time
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

################## 参数定义
NNType = 'SimpleRNN'  # LSTM SimpleRNN
filepath = "/Weights/LSTM_weights/LSTM_weights_alpha2.0.h5"
NNTrain = True
SNR_train = 2       # training on GSNR=2dB
alpha_train = 2.0   # impulsive noise 修改强度的同时记得更改mode.save(filepath)
batch_size = 256
epochs = 2**6
optimizer = 'adam'
loss = 'mse'        # MSE（均方方差) 预测数据和原始数据对应点误差的平方和的均值

################## 训练数据和标签生成

### LDPC编码
k = 8
N = 16
R = k/N  # 码率
nums = 2**18  # 参与训练的码组 2^20 = 1048576

# LDPC校验矩阵
H = [[0, 0 ,0 ,0 ,0, 1 ,0, 1, 0,1 ,1, 0, 0, 0, 0, 0],
     [0 ,1 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,1 ,1 ,0],
     [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,1 ,0 ,1],
     [0 ,0 ,1 ,0 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0],
     [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,1 ,0 ,1 ,0],
     [1 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,1 ,1],
     [0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,1],
     [0 ,0, 0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,0 ,0]]

# 转置的生成矩阵
tG = pyldpc.CodingMatrix(H)

# 编码
Y_train = np.zeros((nums, k), dtype='int64')
for i in range(nums):
    Y_train[i] = np.random.randint(0, 2, k)

X_train = np.zeros((nums, N), dtype='int64')
for i in range(nums):
    # pyldpc.Coding()中 x = pow(-1,d)将tG*Y_train[i]结果中 0,1置换为1,-1
    X_train[i] = (pyldpc.Coding(tG, Y_train[i], 0) + 1) / 2 #又将pyldpc.Coding()返回的1,-1转换为1,0

print("调制前")
print(X_train[1])

### BPSK调制
for i in range(nums):
    X_train[i] = -2 * X_train[i] + 1

print("调制后")
print(X_train[1])

### 加噪(脉冲噪声)

# 脉冲噪声的scale参数
def calscale(GSNR, alpha, R):
    GSNR = 10**(GSNR/10)  # Eb/No conversion from dB to decimal
    S0 = 1/np.sqrt(7.12*GSNR*R)
    gamma = ((1.78*S0)**alpha)/1.78
    scale = gamma**(1/alpha)
    return scale


scale_train = calscale(SNR_train, alpha_train, R)
impulsive_noise = levy_stable.rvs(alpha_train, 0, 0, scale_train, (nums, N))

X_train = X_train + impulsive_noise
print("加噪后")
print(X_train[1])
print("输入网络前调整为")
print(X_train[1])

######################## Define LSTM model


# metric errors
def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))


def errors(y_true, y_pred):
    print(y_true,y_pred)
    return K.sum(tf.cast(K.not_equal(y_true, K.round(y_pred)),tf.int32))


model = Sequential()
if NNType == 'SimpleRNN':
    X_train = X_train.reshape(-1, 16, 1)
    model.add(SimpleRNN(128, return_sequences=True, recurrent_dropout=0.5, input_shape=(None, 1)))
    model.add(SimpleRNN(64, recurrent_dropout=0.5, input_shape=(None, 1)))
    model.add(Dense(32, activation='sigmoid'))
else:
    print("Wrong RNNtype! Only for SimpleRNN!")
    sys.exit()
model.add(Dense(8))
model.compile(loss=loss, optimizer=optimizer, metrics=[ber, errors])
model.summary()


if NNTrain:
    t = time()
    model.fit(X_train, Y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose=1, shuffle=True)
    t = time() - t
    print("Training Time Used:{}s = {}min".format(t, t / 60))
    model.save_weights(filepath=filepath)
else:
    model.load_weights(filepath=filepath)

############# Test NNDecoder on Different GSNRs
test_batch = 1000
num_words = 100000  # multiple of test_batch

SNR_dB_start_Eb = 0
SNR_dB_stop_Eb = 7
SNR_points = 8
SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

nb_errors = np.zeros(SNR_points, dtype=int)
nb_bits = np.zeros(SNR_points, dtype=int)
seedrand = np.zeros(100, dtype=int)
for sr in range(1, 100):
    seedrand[sr] = np.random.randint(0, 2 ** 14, size=(1))  # seedrand[sr-1]+1

t = time()
for i in range(0, SNR_points):  # different  SNR
    scale = calscale(SNRs[i], alpha_train, R)
    print("GSNR={},scale={}".format(SNRs[i], scale))

    for ii in range(0, np.round(num_words / test_batch).astype(int)):

        # Source
        np.random.seed(seedrand[ii])
        d_test = np.random.randint(0, 2, size=(test_batch, k))

        x_test = np.zeros((test_batch, N))   # (1000, 16)
        for iii in range(0, test_batch):
            x_test[iii] = (pyldpc.Coding(tG, d_test[iii], 2) + 1) / 2

        # Modulator (BPSK)
        s_test = -2 * x_test + 1

        # Channel (AWGN)
        # y_test = s_test + sigmas[i]*np.random.standard_normal(s_test.shape)

        # Channel (Impluse Noise)
        y_test = s_test + levy_stable.rvs(alpha_train, 0, 0, scale, (test_batch, N))

        # Decoder
        if NNType == 'LSTM' or NNType == 'SimpleRNN':
            y_test = y_test.reshape(-1, 16, 1)
        nb_errors[i] += model.evaluate(y_test, d_test, batch_size=test_batch, verbose=2)[2]
        nb_bits[i] += d_test.size
t = time() - t
print("Decoding Time used {}s = {}min".format(t, t/60))

# Plot Bit-Error-Rate
legend = []
print(nb_errors/nb_bits)
plt.plot(SNRs, nb_errors/nb_bits)
legend.append(NNType)

plt.legend(legend, loc=3)
plt.yscale('log')
plt.xlabel('$E_b/N_0$')
plt.ylabel('BER')
plt.grid(True)
plt.show()

