# APMS: Automatic Pruning based on Multi-layer Sensitivity

import numpy as np
from Prune import Prune


class APMS():
    def __init__(self, STNND):
        self.STNND = STNND

    def __del__(self):
        print('Delete Object')

    def simplePrune(self):
        ber_both = []
        ber = [0.10131125, 0.07345625, 0.050625, 0.033465, 0.02140375, 0.01315625, 0.0080025, 0.004865, 0.003]
        ber_both.append(ber)
        pruning_rate = np.array([95, 68, 44, 74])
        self.STNND.model.load_weights("Weights/STNet_weights/STNN_weights_218_alpha1.5.h5")
        weight_matrix = self.STNND.model.get_weights()
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        self.STNND.model.set_weights(weight_pruned)

        self.STNND.train(2 ** 10, 256, 1, verbose=0)
        weight_matrix = self.STNND.model.get_weights()
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        self.STNND.model.set_weights(weight_pruned)
        ber_pruned = self.STNND.test(1.5, 0, 8, 9, 100, 100000)
        ber_both.append(ber_pruned)
        print(pruning_rate)
        print(np.array(ber_both))
        Prune.prune_plot(ber_both, pruning_rate)

    def spFineTuning(self):
        # AWGN
        ber_both = []
        ber = [8.479000e-02, 5.104250e-02, 2.633125e-02, 1.122750e-02, 3.751250e-03, 9.262500e-04, 1.612500e-04,
               2.500000e-05, 2.500000e-06]
        ber_both.append(ber)
        pruning_rate = np.array([18, 18, 18, 18])  # [58, 58, 58, 58]
        for i in range(0, 2):
            pruning_rate += 20
            self.STNND.model.load_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")
            weight_matrix = self.STNND.model.get_weights()
            weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
            self.STNND.model.set_weights(weight_pruned)
            self.STNND.train(2 ** 10, 256, 1, verbose=0)
            self.STNND.model.save_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")

        self.STNND.model.load_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")
        weight_matrix = self.STNND.model.get_weights()
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        self.STNND.model.set_weights(weight_pruned)
        ber_pruned = self.STNND.test(2.0, 0, 8, 9, 100, 100000)
        ber_both.append(ber_pruned)
        print(pruning_rate)
        print(np.array(ber_both))
        Prune.prune_plot(ber_both, pruning_rate)

    def apmsUnderAWGN(self):
        ber_both = []
        self.STNND.model.load_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/STNN_weights_218_alpha2.0.h5")
        ber = self.STNND.test(2.0, 0, 7, 8, 100, 100000)
        ber_both.append(ber)
        pruning_rate = [0, 0, 0, 0]
        threshold = [100, 100, 100, 100]
        step = [6, 5, 5, 5, 4, 4, 3, 2, 1]  # 自适应剪枝率
        for i in range(len(pruning_rate)):  # 逐层剪枝
            self.STNND.model.load_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")
            weight_matrix = self.STNND.model.get_weights()
            weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
            self.STNND.model.set_weights(weight_pruned)
            ber = self.STNND.test(2.0, 6, 7, 2, 100, 100000)

            while True:
                weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
                self.STNND.model.set_weights(weight_pruned)
                ber_pruned = self.STNND.test(2.0, 6, 7, 2, 100, 100000)

                # 比较 ber 和 ber_pruned：依据是误码率  比较结果决定了step的大小和是否break
                if pruning_rate[i] >= threshold[i]:
                    break
                if ber_pruned[0] / ber[0] <= 1.3 and ber_pruned[1] / ber[1] <= 1.3:
                    pruning_rate[i] += step[0]
                elif ber_pruned[0] / ber[0] <= 1.5 and ber_pruned[1] / ber[1] <= 1.5:
                    pruning_rate[i] += step[6]  # 调整剪枝率：pruning_rate
                else:
                    break

            print("第" + str(i + 1) + "层剪枝率：", pruning_rate[i])
            self.STNND.train(2 ** 10, 256, 1, verbose=0)
            self.STNND.model.save_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")
            print("------------------------------")
        print("pruning_rate", pruning_rate)

        self.STNND.model.load_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/Prune/STNN_weights_218_alpha2.0.h5")
        weight_matrix = self.STNND.model.get_weights()
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        self.STNND.model.set_weights(weight_pruned)
        self.STNND.model.save_weights("D:/研二上/STNet_Decoder/Weights/STNet_weights/Prune/Pruned_STNN_weights_218_alpha2.0.h5")
        ber_pruned = self.STNND.test(2.0, 0, 7, 8, 100, 100000)
        ber_both.append(ber_pruned)
        print(np.array(ber_both))
        Prune.prune_plot(ber_both, pruning_rate)

    def apmsUnderImpulsNoise(self):
        ber_both = []
        self.STNND.model.load_weights("Weights/STNet_weights/STNN_weights_218_alpha1.5.h5")
        ber = self.STNND.test(1.5, 0, 8, 9, 100, 100000)
        ber_both.append(ber)
        pruning_rate = [0, 0, 0, 0]
        threshold = [100, 100, 100, 100]
        # threshold = [20, 70, 70, 30]
        step = [5, 2]  # 自适应剪枝率
        for i in range(len(pruning_rate)):  # 逐层剪枝
            self.STNND.model.load_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha1.5.h5")
            weight_matrix = self.STNND.model.get_weights()
            if i == 0:
                weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
                self.STNND.model.set_weights(weight_pruned)
                ber = self.STNND.test(1.5, 7, 8, 2, 100, 100000)

            while True:
                weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
                self.STNND.model.set_weights(weight_pruned)
                ber_pruned = self.STNND.test(1.5, 7, 8, 2, 100, 100000)
                print("ber:", ber[7], ber[8])
                print("ber_pruned:", ber_pruned)
                print(ber_pruned[0] / ber[7])
                print(ber_pruned[1] / ber[8])
                # 比较 ber 和 ber_pruned：依据是误码率  比较结果决定了step的大小和是否break
                if pruning_rate[i] >= threshold[i]:
                    break
                if ber_pruned[0] / ber[7] <= 1.12 and ber_pruned[1] / ber[8] <= 1.12:
                    pruning_rate[i] += step[0]
                elif ber_pruned[0] / ber[7] <= 1.25 and ber_pruned[1] / ber[8] <= 1.25:
                    pruning_rate[i] += step[1]  # 调整剪枝率：pruning_rate
                else:
                    break

            print("第" + str(i + 1) + "层剪枝率：", pruning_rate[i])
            self.STNND.train(2 ** 11, 256, 1, verbose=0)
            self.STNND.model.save_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha1.5.h5")
            print("------------------------------")
        print("pruning_rate", pruning_rate)

        self.STNND.model.load_weights("Weights/STNet_weights/Prune/STNN_weights_218_alpha1.5.h5")
        weight_matrix = self.STNND.model.get_weights()
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        self.STNND.model.set_weights(weight_pruned)
        self.STNND.model.save_weights("Weights/STNet_weights/Prune/Pruned_STNN_weights_218_alpha1.5.h5")
        ber_pruned = self.STNND.test(1.5, 0, 8, 9, 100, 100000)
        ber_both.append(ber_pruned)
        print(np.array(ber_both))
        Prune.prune_plot(ber_both, pruning_rate)
