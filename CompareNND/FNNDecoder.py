import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.optimizers import Adam
from Model import CommFunc
from Datasets import Data
from time import time
from scipy.stats import levy_stable
from Metrics import metrics


class DNNDecoder():
    def __init__(self, alpha=1.8, GSNR = 2, use_bias=True):
        self.k = 8
        self.N = 16
        self.R = self.k/self.N
        self.optimizer = Adam(0.0002, 0.5)  # beta1：一阶矩估计的指数衰减率
        self.loss = 'binary_crossentropy'
        self.alpha_train = alpha
        self.GSNR_train = GSNR
        self.scale_train=CommFunc.CalScale(self.GSNR_train,self.alpha_train,self.R)
        self.design = [256, 128, 32]  # [512, 256, 64] [256, 128, 32] [128, 64, 32]
        self.model, self.modulator_layers, self.noise, self.decoder_layers, self.decoder = self.build_Decoder(use_bias)
        np.seterr(divide='ignore', invalid='ignore')

    def __del__(self):
        print('Delete Object')

    def return_output_shape(self, input_shape):
        return input_shape

    def compose_model(self, layers):
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def build_Decoder(self, use_bias):
        # Define modulator
        modulator_layers = [Lambda(CommFunc.modulateBPSK,
                                   input_shape=(self.N,), output_shape=self.return_output_shape, name="modulator")]
        modulator = self.compose_model(modulator_layers)
        modulator.compile(optimizer=self.optimizer, loss=self.loss)

        # Define noise
        noise_layers = [Lambda(CommFunc.addNoise, arguments={'sigma': self.scale_train,'alpha_train':self.alpha_train},
                               input_shape=(self.N,), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        # Define decoder
        decoder_layers = [Dense(self.design[0], activation='sigmoid', input_shape=(self.N,), use_bias=use_bias)]
        decoder_layers.append(Dense(self.design[1], activation='sigmoid', use_bias=use_bias))
        decoder_layers.append(Dense(self.design[2], activation='sigmoid', use_bias=use_bias))
        decoder_layers.append(Dense(self.k, activation='sigmoid'))
        decoder = self.compose_model(decoder_layers)
        decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER,metrics.errors])

        # Define model
        model_layers = modulator_layers + noise_layers + decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        return model, modulator_layers, noise, decoder_layers, decoder

    def train(self, epochs=2**16, batch_size=256, GSNR=1, verbose=1):
        scale_train = CommFunc.CalScale(GSNR, self.alpha_train, self.R)
        noise_layers = [
            Lambda(CommFunc.addNoise, arguments={'sigma': scale_train, 'alpha_train': self.alpha_train},
                   input_shape=(self.N,), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        model_layers = self.modulator_layers + noise_layers + self.decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()
        # generate training data
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
        print("Training Time Used:{}s = {}min".format(t, t / 60))
        print("---------------------------------")
        model.summary()
        return self.model, history

    def test(self,alpha,GSNR_low,GSNR_up,interval,test_batch,num_words):
        t = time()
        SNR_dB_start_Eb = GSNR_low
        SNR_dB_stop_Eb = GSNR_up
        SNR_points = interval
        SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

        nb_errors = np.zeros(SNR_points, dtype=int)
        nb_bits = np.zeros(SNR_points, dtype=int)
        ber = np.zeros(SNR_points, dtype=float)
        seedrand = np.zeros(np.round(num_words / test_batch).astype(int), dtype=int)

        for sr in range(1, 100):
            seedrand[sr] = np.random.randint(0, 2 ** 14, size=(1))  # seedrand[sr-1]+1
        for i in range(0, SNR_points):  # different SNRs
            scale = CommFunc.CalScale(SNRs[i], alpha, self.R)
            print("GSNR={},scale={}".format(SNRs[i], scale))
            for ii in range(0, np.round(num_words / test_batch).astype(int)):
                # Source
                x_test, d_test=Data.genRanData(self.k, self.N, test_batch, seedrand[ii])
                # Modulator (BPSK)
                s_test = -2 * x_test + 1
                # Channel (alpha-stable)
                y_test = s_test + levy_stable.rvs(alpha, 0, 0, scale, (test_batch, self.N))
                # Decoder
                nb_errors[i] += self.decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=2)[2]
                nb_bits[i] += d_test.size
                ber = np.float32(nb_errors/nb_bits)
        t = time() - t
        print("Testing Time used {}s = {}min".format(t, t/60))
        return ber


if __name__ == '__main__':
    DnnDecoder = DNNDecoder(2.0, 2, use_bias=True)
    # DnnDecoder.train(2 ** 18, 256, 2, 1)
    # DnnDecoder.model.save_weights("D:/研二上/STNet_Decoder/Weights/DNN_weights/DNN_weights_AWGN.h5")
    DnnDecoder.model.load_weights("D:/研二上/STNet_Decoder/Weights/DNN_weights/DNN_weights_AWGN.h5")
    ber = DnnDecoder.test(2.0, 0, 6, 7, 100, 100000)
    print("ber:", ber)

