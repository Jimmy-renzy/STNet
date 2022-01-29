import numpy as np
import matplotlib.pyplot as plt


def findKthLargest(nums, k):
    '''
    Add noise (Gaussian and Impulsive)
    :param nums:
    :param k:
    :return: kth largest num in nums
    '''
    import heapq
    return heapq.nlargest(k, nums)[-1]


# function to prune weights connections
def pruneWeight(sublist, pruning_percent): #2D numpy array of weight matrix, percentage of pruning
    # 二维变一维
    sublist = sublist.flatten()
    num = len(sublist)
    remain_num = num * (100 - pruning_percent) / 100
    N = int(round(remain_num)) # no.N of elements that should remain after pruning

    list_tmp = [0 for i in range(len(sublist))]
    for i in range(len(sublist)):
        if sublist[i] < 0:
            list_tmp[i] = -sublist[i]
        else:
            list_tmp[i] = sublist[i]
    threshold = findKthLargest(list_tmp, N)

    final_list = sublist
    for k in range(len(final_list)):
        if abs(final_list[k]) < threshold:
            final_list[k] = 0

    return final_list


def pruneModel(weight_matrix, pruning_rate):

    weight_matrix_pruned = []    # 剪织后的权重矩阵

    for i in range(len(weight_matrix)):
        if i % 2 == 1:
            weight_matrix_pruned.append(weight_matrix[i])
            continue
        else:
            list_pruned = pruneWeight(weight_matrix[i], pruning_rate[int(i/2)])
            weight_matrix_pruned.append(np.reshape(list_pruned, weight_matrix[i].shape))

    return weight_matrix_pruned


def prune_plot(ber, pruning_rate):
    SNR_dB_start_Eb = 0
    SNR_dB_stop_Eb = 8
    SNR_points = 9
    SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

    legend = []

    plt.plot(SNRs, ber[0])
    legend.append('STNN')
    plt.plot(SNRs, ber[1])
    legend.append('STNN_prune' + str(pruning_rate))

    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.grid(True)
    plt.show()


def multi_layer_prune_plot(layer, ber, snr_start, snr_stop, interval):
    SNR_dB_start_Eb = snr_start
    SNR_dB_stop_Eb = snr_stop
    SNR_points = interval
    SNRs = np.linspace(SNR_dB_start_Eb, SNR_dB_stop_Eb, SNR_points)

    legend = []
    for i in range(len(ber)):
        plt.plot(SNRs, ber[i])
        legend.append("layer" + str(layer) + " prune" + str(i*10) + '%')

    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.grid(True)
    plt.show()


"""
    # 层剪枝敏感度测试
    GnnDecoder.model.load_weights("Weights/MGN_weights/min_gated_net_weights_218.h5")

    weight_matrix = GnnDecoder.model.get_weights()
    # nums = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    nums = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    ber_pruned = []
    for i in range(len(nums)):
        pruning_rate = [nums[i], 0, 0, 0]
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        GnnDecoder.model.set_weights(weight_pruned)
        # ber = GnnDecoder.test(2.0, 0, 7, 8, 100, 200000)
        ber = GnnDecoder.test(2.0, 6, 7, 2, 100, 500000)
        print(ber)
        ber_pruned.append(ber)

    print("ber_pruned:")
    print(ber_pruned)
    # Prune.prune_plot(2, ber_pruned, 0, 7, 8)
    pr = [nums[i]/100.0 for i in range(len(nums))]
    ber = [ber_pruned[i][0] for i in range(len(ber_pruned))]
    plt.plot(pr, ber)
    plt.grid(True)
    plt.show()


    #层敏感度组合剪枝
    GnnDecoder.model.load_weights("Weights/min_gated_net_weights_218.h5")
    ber = GnnDecoder.test(2.0, 0, 7, 8, 100, 100000)
    weight_matrix = GnnDecoder.model.get_weights()
    pruning_rate = [10, 30, 35, 10]
    weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
    GnnDecoder.model.set_weights(weight_pruned)
    ber_pruned = GnnDecoder.test(2.0, 0, 7, 8, 100, 100000)
    print("ber_pruned:", ber_pruned)

    SNRs = np.linspace(0, 7, 8)
    legend = []
    plt.plot(SNRs, ber)
    legend.append("MGN")
    plt.plot(SNRs, ber_pruned)
    legend.append("MGN_prune[10, 30, 35, 10]")

    plt.legend(legend, loc=3)
    plt.yscale('log')
    plt.xlabel('$E_b/N_0$')
    plt.ylabel('BER')
    plt.grid(True)
    plt.show()


    # 层剪枝敏感度测试
    GnnDecoder.model.load_weights("Weights/min_gated_net_weights_218.h5")
    ber = GnnDecoder.test(2.0, 0, 7, 8, 100, 100000)

    weight_matrix = GnnDecoder.model.get_weights()
    nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # nums = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    ber_pruned = []
    ber_pruned.append(ber)
    for i in range(len(nums)):
        pruning_rate = [0, 0, nums[i], 0]
        weight_pruned = Prune.pruneModel(weight_matrix, pruning_rate)
        GnnDecoder.model.set_weights(weight_pruned)
        ber = GnnDecoder.test(2.0, 0, 7, 8, 100, 100000)
        print(ber)
        ber_pruned.append(ber)

    print("ber_pruned:")
    print(ber_pruned)
    Prune.prune_plot(3, ber_pruned, 0, 7, 8)
"""