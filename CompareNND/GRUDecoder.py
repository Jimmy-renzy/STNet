import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import GRU, Bidirectional
from scipy.stats import levy_stable
from keras.optimizers import Adam
from Model import CommFunc
from Datasets import Data
from time import time
from Metrics import metrics

# Don't add noise in Lamda Layer, add noise dependently instead of
# 因为在Lamda层加噪声，就意味着1bit的输入只对应noise_layer每次单独生成1bit的噪声 而DNN是整组16bit的噪声生成

class GRUDecoder():
    def __init__(self, alpha=1.8, GSNR=1):
        self.k = 8
        self.N = 16
        self.R = self.k/self.N
        # self.optimizer = Adam(0.0002, 0.5)
        self.optimizer = Adam(0.0002, 0.5)
        self.loss = 'binary_crossentropy'
        self.GSNR_train = GSNR
        self.alpha_train = alpha
        self.scale_train=CommFunc.CalScale(self.GSNR_train, self.alpha_train, self.R)
        self.model, self.decoder_layers, self.decoder = self.build_Decoder()

    def return_output_shape(self, input_shape):
        return input_shape

    def compose_model(self, layers):
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model

    def build_Decoder(self):
        # Define decoder
        decoder_layers = [Bidirectional(GRU(256, return_sequences=True, input_shape=(None, 1), recurrent_dropout=0.5))]
        decoder_layers.append(Bidirectional(GRU(128, recurrent_dropout=0.5)))
        decoder_layers.append(Dense(32))
        decoder_layers.append(Dense(8))
        decoder = self.compose_model(decoder_layers)
        decoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER, metrics.errors])

        # Define model
        model_layers = decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()
        return model, decoder_layers, decoder

    def train(self, epochs=2**3, batch_size=2**17, GSNR=1, verbose=1):
        model_layers = self.decoder_layers
        model = self.compose_model(model_layers)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[metrics.BER])
        model.summary()

        X_train, Y_train = Data.genRanData(self.k, self.N, batch_size, np.random.randint(0, 100, size=1))
        X_val, Y_val = X_train, Y_train
        X_val = X_val.reshape(-1, 16, 1)

        # modulate
        X_train = -2 * X_train + 1
        # add noise
        # X_train = X_train + levy_stable.rvs(self.alpha_train, 0, 0, self.scale_train, (batch_size, self.N))
        X_train = X_train.reshape(-1, 16, 1)
        print("X_train shape:", X_train.shape)
        print("X_train:", X_train[0])
        t = time()
        history = model.fit(X_train, Y_train,
                                 validation_data=[X_val, Y_val],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=verbose,
                                 shuffle=True)
        t=time()-t
        print("Training Time Used:{}s = {}min".format(t, t / 60))
        return self.model, history

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
        t = time() - t
        print("Decoding Time Used:{}s = {}min".format(t, t / 60))
        return ber,nb_bits,nb_errors


if __name__ == '__main__':
    RnnDecoder = GRUDecoder(1.8, 1)
    RnnDecoder.train(2**6, 2**20, 1, 1)
    # RnnDecoder.model.save_weights("Weights/LSTM_weights/RNN_weights_216_alpha2.0.h5")
    # RnnDecoder.model.load_weights("Weights/LSTM_weights/RNN_weights_216_alpha2.0.h5")
    ber1,nb_bits,nb_errors = RnnDecoder.test(1.8, 0, 9, 10, 100, 10000)
    print(ber1, nb_bits, nb_errors)



