'''
完成LDPC编码 训练集和测试集的数据生成
'''
import pyldpc
import numpy as np
from keras import backend as K

def encoding(k=8,N=16,H=None):
    if(k==8 and N == 16):
        H = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
    tG=pyldpc.CodingMatrix(H)
    return tG


# Create all possible information words
def dec2bin(num):
    l = np.zeros((8), dtype='int64')
    i = 7
    while True:
        num, remainder = divmod(num, 2)
        l[i] = int(remainder)
        i = i - 1
        if num == 0:
            return l

# Generate the training data
def genData(k,N,num):
    tG = encoding(k, N, [])
    # Generate label
    label = np.zeros((num, k), dtype='int64')
    for s in range(0, 256):
        label[s] = dec2bin(s)

    # Create sets of all possible codewords (codebook)
    data = np.zeros((num, N), dtype=int)
    for i in range(0, num):
        data[i] = (pyldpc.Coding(tG, label[i], 0) + 1) / 2   # 修改了pyLDPC源文件，pyLDPC生产码字不加噪声，统一用noise_layers层加噪
    data = data.reshape(-1, 16)  # 在reshape前，data维度为(256, 16)，reshape后，维度没有发生变化，暂且理解为让下级模块更清楚数据的维度
    return data, label

def genRanData(k,N,num,seedrand):
    np.random.seed(seedrand)
    tG = encoding(k, N, [])
    d_test = np.random.randint(0, 2, size=(num, k))
    x_test = np.zeros((num, N))
    for j in range(0, num):
        x_test[j] = (pyldpc.Coding(tG, d_test[j], 2) + 1) / 2
    return x_test, d_test

def genRnnData(k, N, num):
    tG = encoding(k, N, [])
    # Generate label
    label = np.zeros((num, k), dtype='int64')
    for i in range(num/256):
        for j in range(0, 256):
            label[i*256+j] = dec2bin(j)

    # Create sets of all possible codewords (codebook)
    data = np.zeros((num, N), dtype=int)
    for i in range(0, num):
        data[i] = (pyldpc.Coding(tG, label[i], 0) + 1) / 2  # no Noise! HY：修改了pyLDPC源文件，pyLDPC生产码字不加噪声，统一用noise_layers层加噪
    data = data.reshape(-1, 16)  # 在reshape前，data维度为(256, 16)，reshape后，维度没有发生变化，暂且理解为让下级模块更清楚数据的维度
    return data, label


if __name__ == '__main__':
    k = 8
    N = 16
    num = 256
    tG = encoding(8, 16, [])
    print("tG:", tG)
    test = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    print("test:", test.shape)
    code = tG.dot(test) % 2

    print('code:', code)
    # test 为一维向量，shape(8,); tG 为二维矩阵，shape(16, 8)
    # tG.dot(test) 相当于 tG 的每一维数组与 test 做点积，最终code为(16,)

    data, label = genData(k, N, num)
    print('Data:', data.shape, data[0:5])
    print('Label:', label.shape, label[0:5])

    H = [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
    H = np.array(H)
    dec_data = pyldpc.Decoding_BP(H, data[0], 0)
    error = K.not_equal(data, dec_data)
    print("error:", error)
