import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers import SimpleRNN,LSTM, GRU
from keras.layers.wrappers import Bidirectional
from scipy.stats import levy_stable
from keras.optimizers import Adam
from Model import CommFunc
from Datasets import Data
from time import time
from Metrics import metrics


class RNNDecoder():
    def __init__(self, alpha=1.8, GSNR=1, RNNtype='LSTM'):
        self.k = 8
        self.N = 16
        self.R = self.k/self.N
        self.optimizer = Adam(0.0002, 0.5)
        self.loss = 'binary_crossentropy'
        self.GSNR_train = GSNR
        self.alpha_train = alpha
        self.scale_train=CommFunc.CalScale(self.GSNR_train, self.alpha_train, self.R)
        self.model, self.modulator_layers, self.noise, self.decoder_layers, self.decoder = self.build_Decoder(RNNtype)

    def return_output_shape(self, input_shape):
        return input_shape

    def compose_model(self, layers):
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def build_Decoder(self,RNNtype):
        # Define modulator
        modulator_layers = [Lambda(CommFunc.modulateBPSK,
                                   input_shape=(None,1), output_shape=self.return_output_shape, name="modulator")]
        modulator = self.compose_model(modulator_layers)
        modulator.compile(optimizer=self.optimizer, loss=self.loss)

        # Define noise
        noise_layers = [Lambda(CommFunc.addNoise, arguments={'sigma': self.scale_train,'alpha_train':self.alpha_train},
                               input_shape=(None, 1), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        # Define decoder
        decoder_layers = []
        if RNNtype == 'LSTM':
            decoder_layers = [LSTM(256, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5)]
            decoder_layers.append(LSTM(128, recurrent_dropout=0.5))

        elif RNNtype == 'GRU':
            decoder_layers = [GRU(256, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5)]
            decoder_layers.append(GRU(128, recurrent_dropout=0.5))

        elif RNNtype == 'SimpleRNN':
            decoder_layers = [Bidirectional(SimpleRNN(200, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5))]
            decoder_layers.append(Bidirectional(SimpleRNN(200, recurrent_dropout=0.5)))
        else:
            print("Wrong RNNtype! Only for LSTM/GRU/SimpleRNN !")

        decoder_layers.append(Dense(32))
        decoder_layers.append(Dense(8))
        decoder = self.compose_model(decoder_layers)
        decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER, metrics.errors])

        # Define model
        model_layers = modulator_layers + noise_layers + decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()
        return model, modulator_layers, noise, decoder_layers, decoder

    def train(self, epochs, batch_size=256, GSNR=1,verbose=1):
        print("epochs:",epochs)
        scale_train = CommFunc.CalScale(GSNR, self.alpha_train, self.R)

        noise_layers = [
            Lambda(CommFunc.addNoise, arguments={'sigma': scale_train, 'alpha_train': self.alpha_train},
                   input_shape=(None, 1), output_shape=self.return_output_shape, name="noise")]
        noise = self.compose_model(noise_layers)
        noise.compile(optimizer=self.optimizer, loss=self.loss)

        model_layers = self.modulator_layers + noise_layers + self.decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])


        X_train,Y_train = Data.genData(self.k,self.N,batch_size)
        print("X_train before:", X_train.shape)
        print(X_train[0])
        X_train = X_train.reshape(-1,16,1)
        print("X_train after:", X_train.shape)
        print(X_train[0])
        X_val, Y_val = Data.genData(self.k, self.N, batch_size)
        X_val=X_val.reshape(-1,16,1)
        t = time()
        history = model.fit(X_train, Y_train,
                                 validation_data=[X_val,Y_val],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 shuffle=True)
        t=time()-t
        print("Training Time Used:{}s = {}min".format(t, t / 60))
        return self.model,history

    def test(self,alpha,GSNR_low,GSNR_up,interval,test_batch,num_words):
        np.seterr(divide='ignore', invalid='ignore')
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
        batch_num = np.round(num_words / test_batch).astype(int)
        seedrand = np.zeros(batch_num, dtype=int)

        t = time()
        for sr in range(1, batch_num):
            seedrand[sr] = np.random.randint(0, 2 ** 14, size=(1))  # seedrand[sr-1]+1
        for i in range(0, len(sigmas)):  # different  SNR
            scale = CommFunc.CalScale(SNRs[i], alpha, self.R)
            # print("GSNR={},scale={}".format(SNRs[i], scale))
            for ii in range(0, batch_num):
                # Source
                x_test, d_test=Data.genRanData(self.k, self.N, test_batch, seedrand[ii])
                # Modulator (BPSK)
                s_test = -2 * x_test + 1
                # Channel (alpha-stable)
                y_test = s_test + levy_stable.rvs(alpha, 0, 0, scale, (test_batch, self.N))
                y_test = y_test.reshape(-1,16,1)
                # Decoder
                nb_errors[i] += self.decoder.evaluate(y_test, d_test, batch_size=test_batch, verbose=2)[2]

                nb_bits[i] += d_test.size
                ber = np.float32(nb_errors/nb_bits)
            # print(y_test[0:5],d_test[0:5])
        t = time() - t
        print("Decoding Time Used:{}s = {}min".format(t, t / 60))
        return ber,nb_bits,nb_errors


if __name__ == '__main__':
    LSTMDecoder=RNNDecoder(2.0, 1, 'LSTM')
    LSTMDecoder.model.load_weights("Weights/RNN_weights/LSTM_weights_216_alpha2.0.h5")
    LSTMDecoder.test(2.0, 0, 0, 1, 100, 100000)
    # LSTMDecoder.train(2**16, 256, 1, 0)
    # LSTMDecoder.model.save_weights("Weights/RNN_weights/LSTM_weights_216_alpha2.0.h5")
    # ber1,nb_bits,nb_errors = LSTMDecoder.test(2.0, 0, 9, 10, 100, 100000)
    # print(ber1,nb_bits,nb_errors)

    # arr = np.load("D:/D研二上/STNet_Decoder/20200428_13_41_ber_result.npy")
    # print(arr[2])

    # def get_flops(model_h5_path):
    #     session = tf.compat.v1.Session()
    #     graph = tf.compat.v1.get_default_graph()
    #
    #     with graph.as_default():
    #         with session.as_default():
    #             model = tf.keras.models.load_model(model_h5_path)
    #
    #             run_meta = tf.compat.v1.RunMetadata()
    #             opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    #
    #             # We use the Keras session graph in the call to the profiler.
    #             flops = tf.compat.v1.profiler.profile(graph=graph,
    #                                                   run_meta=run_meta, cmd='op', options=opts)
    #
    #             return flops.total_float_ops

### LSTM


### GRU



