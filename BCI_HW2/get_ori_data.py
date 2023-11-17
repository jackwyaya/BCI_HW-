import pandas as pd
import scipy.io as io
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import h5py
from h5py import Dataset
import os
import bottleneck as bn
import math
from tqdm import tqdm


def load_mat(data_path):
    data = io.loadmat(data_path)
    return data

def compute_MI(x, y):
    MI = 0
    # print(x.shape,y.shape)
    # for i in range(x.shape[0]):
    #     MI+=metrics.adjusted_mutual_info_score(x[i],y[i])
    MI += metrics.adjusted_mutual_info_score(x, y)
    return MI


def select_top(x, y, num):
    MI = np.zeros(y.shape[1])

    for i in range(y.shape[1]):
        for j in range(x.shape[1]):
            MI[i] += compute_MI(y[:, i], x[:, j])

    print("MI=", MI)
    topk = np.argpartition(MI, -num)[-num:]
    topk = np.sort(topk)
    print(topk)
    print(y.shape)
    y = y[:, topk]
    print(y)
    return y

def z_score(x):
    print("preprocessing_x=", x.shape)
    x = preprocessing.scale(x, axis=0)
    return x

def moving_average(y,k):
    smooth_y = np.copy(y)

    for i in range(k):
        smooth_y[k:, :] = smooth_y[k:, :] + y[i:y.shape[0]-k+i, :]

    smooth_y=smooth_y/k
    smooth_y=smooth_y[k:,:]

    return smooth_y

def Bined_data(data_path):
    data = h5py.File(data_path)

    print("load..." + data_path)

    print("data.keys()=", data.keys())

    # print("data['#refs#']=",data['#refs#'].shape)

    print(data['chan_names'][:].shape)

    print(data['cursor_pos'])
    print(data['spikes'])
    print(data['t'])

    ori_x = np.array(data['cursor_pos'][:])
    t = np.array(data['t'][:])
    ori_x = ori_x.astype(float)
    t = t.astype(float)

    print("ori_x.shape=", ori_x.shape)
    print("t.shape=", t.shape)
    print("t=", t)
    chan_name = 0

    chan_num = data['spikes'][:].shape[1]
    print("chan_num=", chan_num)

    x = []
    y = []

    bin_size = 25

    for i in tqdm(range(chan_num)):

        # print("i=",i)
        now_chan_y = []
        now_chan_x=[]
        now_spike = np.array(data[(data['spikes'][chan_name][i])][:])
        now_spike=now_spike.astype(float)
        # print("now_spike.shape=",now_spike.shape)
        # print("now_spike=",now_spike)
        # print("t=",t)

        # plt.plot(np.,now_spike)

        k = 0
        j = 0

        #把时间对齐到t开始的时候
        while (now_spike.shape[0] <= 1 and now_spike[0][k] < t[0][0]):
            k = k + 1
            if (k == now_spike.shape[1] - 1):
                break

        while j < t.shape[1]:
            if (now_spike.shape[0] > 1 or now_spike.shape[1] == 0):
                break

            min = j
            max = j + bin_size

            if (max >= t.shape[1]):
                max = t.shape[1] - 1

            # print("min={},max={}".format(min,max))

            #运动轨迹只需要计算一次就行
            if (i == 0):
                now_x = ori_x[:, max - 1] - ori_x[:, min]
                now_x=np.array(now_x)
            # print("now_x=",now_x)

            now_y = 0

            while (k < now_spike.shape[1] and now_spike[0][k] <= t[0][max] and now_spike[0][k] >= t[0][min]):
                now_spike_=now_spike[0][k]
                t_max=t[0][max]
                t_min=t[0][min]
                now_y = now_y + 1
                k = k + 1


            j = j + bin_size

            now_chan_y.append(now_y)

            if (i == 0):
                x.append(now_x)

        if now_chan_y == []:
            now_chan_y = np.zeros(math.ceil(t.shape[1] / bin_size))
        # print("math.ceil(t.shape[1]/bin_size=",math.ceil(t.shape[1]/bin_size))
        now_chan_y = np.array(now_chan_y)

        if now_chan_y.shape[0] != math.ceil(t.shape[1] / bin_size):
            print("now_chan_y.shape[0]!=int(t.shape[1]/bin_size)")
            print(now_chan_y)
            print(now_chan_y.shape)

        y.append(now_chan_y)


    x = np.array(x)
    y = np.array(y).T

    # select M1 channal
    if ("loco" in data_path):
        y=y[:,0:96]

    print("x=",x)
    print("y=",y)

    print("x.shape=", x.shape)
    print("y.shape=", y.shape)

    return x,y

def get_Zenodo_data(data_path):
    x,y=Bined_data(data_path)

    print("x.shape=", x.shape)
    print("y.shape=", y.shape)

    y = y.astype(float)

    # 对神经信号和运动信号进行平滑
    # x = smooth_same(x, 3,mode="same")
    # y = smooth_same(y, 3,mode="same")

    # x = smooth_vaild(x, 3,mode="vaild")
    # y = smooth_vaild(y, 3,mode="vaild")

    # x = moving_average(x, 3)
    # y = moving_average(y, 3)
    # x=x[3:,:]

    # 对运动信号归一化
    # x = preprocessing.scale(x,axis=0)

    print("x.shape=", x.shape)
    print("y.shape=", y.shape)

    print("x=",x)
    print("y=",y)

    return x,y

def get_neuron_data(data_path):
    data = h5py.File(data_path)

    print("load..." + data_path)

    print("data.keys()=", data.keys())

    # print("data['#refs#']=",data['#refs#'].shape)

    print(data['chan_names'][:].shape)

    print(data['cursor_pos'])
    print(data['spikes'])
    print(data['t'])

    ori_pos = np.array(data['cursor_pos'][:])
    t = np.array(data['t'][:])

    ori_pos = ori_pos.astype(float)
    t = t.astype(float)

    print("ori_pos.shape=", ori_pos.shape)
    print("t.shape=", t.shape)
    print("t=", t)
    # chan_name = 0

    chan_num = data['spikes'][:].shape[1]
    print("chan_num=", chan_num)

    x = []
    y = []

    spike=[]
    kin_pos=[]

    for i in tqdm(range(chan_num)):
        for j in range(4):
            now_spike = np.array(data[(data['spikes'][j+1][i])][:])
            now_spike = now_spike.astype(float)
            # now_spike=now_spike[now_spike>t[0][0]]
            if now_spike.size>500:
                # print(now_spike.size)
                spike.append(now_spike)

    print("len(spike)=",len(spike))

    return t,spike,ori_pos

def mk_df(t,spike):
    data_df = pd.DataFrame()
    # data_df['start'] = np.squeeze(t[0][0])
    # data_df['stop']=np.squeeze(t[0][-1])

    data_df['start'] = [t[0][0]]
    data_df['stop']=[t[0][-1]]

    print(data_df.head())

    return data_df

def Bin_Neuron_data(t,spike,cursor,bin_size):
    print("cursor.shape=", cursor.shape)
    print("t.shape=", t.shape)

    neuron_num = len(spike)
    print("chan_num=", neuron_num)

    pos_ = []
    vel_ = []
    acc_ = []
    spikes_ = []

    bin_size = 25

    for i in tqdm(range(neuron_num)):

        now_chan_y = []
        now_chan_x = []
        now_spike = spike[i]

        k = 0
        j = 0

        # 把时间对齐到t开始的时候
        while (now_spike.shape[0] <= 1 and now_spike[0][k] < t[0][0]):
            k = k + 1
            if (k == now_spike.shape[1] - 1):
                break

        while j < t.shape[1]:
            if (now_spike.shape[0] > 1 or now_spike.shape[1] == 0):
                break

            min = j
            max = j + bin_size

            if (max >= t.shape[1]):
                max = t.shape[1] - 1

            # 运动轨迹只需要计算一次就行
            if (i == 0):
                now_pos= (cursor[:, max - 1] + cursor[:, min])/2
                now_vel = cursor[:, max - 1] - cursor[:, min]
                now_acc = cursor[:, max - 1]-cursor[:, max - 2]-(cursor[:, min+1]-cursor[:, min])

            # print("now_x=",now_x)

            now_y = 0

            while (k < now_spike.shape[1] and now_spike[0][k] <= t[0][max] and now_spike[0][k] >= t[0][min]):
                now_spike_ = now_spike[0][k]
                t_max = t[0][max]
                t_min = t[0][min]
                now_y = now_y + 1
                k = k + 1

            j = j + bin_size

            now_chan_y.append(now_y)

            if (i == 0):
                vel_.append(now_vel)
                pos_.append(now_pos)
                acc_.append(now_acc)

        if now_chan_y == []:
            now_chan_y = np.zeros(math.ceil(t.shape[1] / bin_size))
        # print("math.ceil(t.shape[1]/bin_size=",math.ceil(t.shape[1]/bin_size))
        now_chan_y = np.array(now_chan_y)

        if now_chan_y.shape[0] != math.ceil(t.shape[1] / bin_size):
            print("now_chan_y.shape[0]!=int(t.shape[1]/bin_size)")
            print(now_chan_y)
            print(now_chan_y.shape)

        spikes_.append(now_chan_y)

    spikes_ = np.array(spikes_)
    vel_ = np.array(vel_)
    pos_ = np.array(pos_)
    acc_ = np.array(acc_)
    spikes_=spikes_.T

    print("spikes_.shape=", spikes_.shape)
    print("vel_.shape=", vel_.shape)
    print("pos_.shape=", pos_.shape)
    print("acc_.shape=", acc_.shape)

    # print("spikes_=", spikes_)
    # print("vel_=", vel_)
    # print("pos_=", pos_)
    # print("acc_=", acc_)


    return spikes_,pos_,vel_,acc_

if __name__ == '__main__':

#---------------------------------------------------------------------
    # 读取Zenodo数据集的数据

    i='indy_20161024.mat'
    t,spike,cusor=get_neuron_data('Zenodo/{}'.format(i))
    mk_df(t,spike)
    # x, y = get_Zenodo_data('Zenodo/{}'.format(i))
    #
    # print("x.shape=",x.shape)
    # print("y.shape=",y.shape)
