import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from Layers.STUnit import STUnit, GatedDense, RSUnit
from keras.optimizers import Adam
from Model import CommFunc
from Datasets import Data
from time import time
from scipy.stats import levy_stable
from Metrics import metrics
from Prune.APMS import APMS


class STNNDecoder():
    def __init__(self,alpha=1.8, GSNR=1, NNtype='STUnit', use_bias=False):
        self.k = 8
        self.N = 16
        self.R = self.k/self.N
        self.optimizer = Adam(0.0002, 0.5)
        self.loss = 'binary_crossentropy'
        self.GSNR_train = GSNR
        self.alpha_train = alpha
        self.use_bias = use_bias
        self.scale_train = CommFunc.CalScale(self.GSNR_train,self.alpha_train,self.R)
        self.model, self.modulator_layers, self.noise, self.decoder_layers, self.decoder= self.build_Decoder(NNtype)

    def __del__(self):
        print('Delete Object')

    def return_output_shape(self, input_shape):
        return input_shape

    def compose_model(self,layers):
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def build_Decoder(self, NNtype):
        # Define modulator
        modulator_layers = [Lambda(CommFunc.modulateBPSK,
                                   input_shape=(self.N,), output_shape=self.return_output_shape, name="modulator")]
        modulator = self.compose_model(modulator_layers)
        modulator.compile(optimizer=self.optimizer, loss=self.loss)

        # Define noise
        noise_layers = [Lambda(CommFunc.addNoise, arguments={'sigma': self.scale_train,'alpha_train':self.alpha_train},
                               input_shape=(self.N,), output_shape=self.return_output_shape, name="noise")]
        # noise_layers = [Lambda(CommFunc.addRayleighNoise, arguments={'SNR': self.GSNR_train},
        #                        input_shape=(self.N,), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        # Define decoder
        decoder_layers = []
        if NNtype == 'GatedDense':
            decoder_layers.append(GatedDense(256, input_shape=(self.N,), use_bias=self.use_bias))
            decoder_layers.append((GatedDense(128, use_bias=self.use_bias)))
            decoder_layers.append((GatedDense(32, use_bias=self.use_bias)))
        elif NNtype == 'STUnit':
            decoder_layers.append(STUnit(256, input_shape=(self.N,), use_bias=self.use_bias))
            decoder_layers.append((STUnit(128, use_bias=self.use_bias)))
            decoder_layers.append((STUnit(32, use_bias=self.use_bias)))
        elif NNtype == 'RSUnit':
            decoder_layers.append(RSUnit(256, input_shape=(self.N,), use_bias=self.use_bias))
            decoder_layers.append((RSUnit(128, use_bias=self.use_bias)))
            # decoder_layers.append((Dense(units=32, use_bias=self.use_bias)))
            decoder_layers.append((RSUnit(32, use_bias=self.use_bias)))
        else:
            print("Wrong NNtype! Only for STUnit/MinGatedDense/GatedDense !")

        decoder_layers.append((Dense(units=8, activation='sigmoid')))
        decoder = self.compose_model(decoder_layers)
        decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER, metrics.errors])

        # Define model
        model_layers = modulator_layers + noise_layers + decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()
        return model, modulator_layers, noise, decoder_layers, decoder

    def train(self,epochs=2**16,batch_size=256,GSNR=1,verbose=1):
        # calculate the training scale
        scale_train = CommFunc.CalScale(GSNR, self.alpha_train, self.R)
        noise_layers = [Lambda(CommFunc.addNoise, arguments={'sigma': scale_train, 'alpha_train': self.alpha_train},
                               input_shape=(None, 1), output_shape=self.return_output_shape, name="noise")]
        # noise_layers = [Lambda(CommFunc.addRayleighNoise, arguments={'SNR': self.GSNR_train},
        #                        input_shape=(self.N,), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)
        # build the whole model
        model_layers = self.modulator_layers + noise_layers + self.decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])

        # generate the training data
        X_train, Y_train = Data.genData(self.k, self.N, batch_size)
        X_val, Y_val = Data.genData(self.k, self.N, batch_size)
        t = time()
        history = self.model.fit(X_train, Y_train,
                                 validation_data=[X_val,Y_val],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 shuffle=True)
        t=time()-t
        print("Time Used:{}s = {}min".format(t, t / 60))
        return self.model, history

    def test(self, alpha, GSNR_low, GSNR_up, interval, test_batch, num_words):
        '''
        在(GSNR_low, GSNR_up)范围内，产生测试码字，对NND的译码性能进行测试，返回误比特率。
        :param alpha: 脉冲噪声的强度
        :param GSNR_low: GSNR测试范围的下限
        :param GSNR_up: GSNR测试范围的上限
        :param interval: GSNR测试范围内选取的测试点总数
        :param test_batch: 每个测试集的size为(test_batch, k)
        :param num_words: 共生成多少个测试集
        :return: 误比特率ber
        '''
        np.seterr(divide='ignore', invalid='ignore')
        # set testing arrange
        SNR_dB_start_Eb = GSNR_low
        SNR_dB_stop_Eb = GSNR_up
        SNR_points = interval
        SNR_dB_start_Es = SNR_dB_start_Eb + 10 * np.log10(self.k / self.N)
        SNR_dB_stop_Es = SNR_dB_stop_Eb + 10 * np.log10(self.k / self.N)
        SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

        sigma_start = np.sqrt(1 / (2 * 10 ** (SNR_dB_start_Es / 10)))
        sigma_stop = np.sqrt(1 / (2 * 10 ** (SNR_dB_stop_Es / 10)))
        sigmas = np.linspace(sigma_start, sigma_stop, SNR_points)

        nb_errors = np.zeros(len(sigmas), dtype=int)
        nb_bits = np.zeros(len(sigmas), dtype=int)
        ber = np.zeros(len(sigmas), dtype=float)
        seedrand = np.zeros(np.round(num_words / test_batch).astype(int), dtype=int)

        t = time()
        for sr in range(0, np.round(num_words / test_batch).astype(int)):
            seedrand[sr] = np.random.randint(0, 2 ** 14, size=(1))  # seedrand[sr-1]+1
        for i in range(0, len(sigmas)):  # different  SNR
            scale = CommFunc.CalScale(SNRs[i], alpha, self.R)
            for ii in range(0, np.round(num_words / test_batch).astype(int)):
                # Source
                x_test, d_test=Data.genRanData(self.k, self.N, test_batch, seedrand[ii])
                # Modulator (BPSK)
                s_test = -2 * x_test + 1

                # Channel (alpha-stable Impulsive)
                y_test = s_test + levy_stable.rvs(alpha, 0, 0, scale, (test_batch, self.N))

                # Channel (AWGN)
                # y_test = s_test + sigmas[i] * np.random.standard_normal(s_test.shape)

                # Channel (Rayleigh)
                # y_test = s_test + CommFunc.Rnoise(SNRs[i], (test_batch, self.N))

                # NN Decoder Predict
                nb_errors[i] += self.decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=2)[2]
                nb_bits[i] += d_test.size
                ber = np.float32(nb_errors/nb_bits)
        t = time() - t
        print("Time Used:{}s = {}min".format(t, t / 60))
        return ber


if __name__ == '__main__':
    STNND = STNNDecoder(alpha=2.0, GSNR=2, NNtype='STUnit', use_bias=True)

    # STNND.train(2 ** 18, 256, 2, verbose=1)
    # STNND.model.save_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/STNN_weights_218_alpha2.0.h5")

    STNND.model.load_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/STNN_weights_218_alpha2.0.h5")
    ber = STNND.test(2.0, 0, 7, 8, 100, 100000)
    print(ber)

    # 基于多层敏感度的自适应剪枝算法
    # Clipper = APMS(STNND)
    # Clipper.apmsUnderAWGN()





