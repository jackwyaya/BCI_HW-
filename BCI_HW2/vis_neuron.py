from math import pi

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from spykes_pk.spykes.plot.neurovis import NeuroVis
from spykes_pk.spykes.plot.popvis import PopVis
from get_ori_data import get_Zenodo_data, Bined_data, moving_average, get_neuron_data, mk_df, Bin_Neuron_data
from spykes_pk.spykes.ml.neuropop import NeuroPop



import h5py

def initiate_neurons(spikes):

    neuron_list = list()

    for i in range(len(spikes)):
        # instantiate neuron
        neuron = NeuroVis(spikes[i], name=(i + 1))
        neuron_list.append(neuron)
    # print(len(neuron_list))
    return neuron_list

def show_single(data_path):
    t,spike,pos=get_neuron_data(data_path)
    df=mk_df(t,spike)

    V=NeuroVis(spike[1],"indy")

    windows=[-5000,20000]

    raster=V.get_raster(event='start',df=df,window=windows,binsize=25)
    V.plot_raster(raster)

    psth=V.get_psth(event='start',df=df,window=windows)
    V.plot_psth(psth)

def show_pop(data_path):
    t,spike,pos=get_neuron_data(data_path)
    df=mk_df(t,spike)
    spike=initiate_neurons(spike)
    V=PopVis(spike[0:1],"indy")

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=.3)
    windows=[-5000,20000]

    psth=V.get_all_psth(event='start',df=df,window=windows,binsize=25,plot=True,conditions_names=['pop psth'])
    plt.show()

    # plt.figure(figsize=(10, 5))
    # V.plot_population_psth(all_psth=psth)
    # plt.show()

def compute_D(v):
    D=np.zeros((v.shape[0],1))

    D=np.arctan2(v[:,1],v[:,0])
    # D=np.degrees(D)
    # D[D<0]+=2*pi
    D=np.array(D)
    return D

def show_tuning(data_path,n_neurons=107):
    t, spike, pos = get_neuron_data(data_path)
    spikes_,pos_,vel_,acc_=Bin_Neuron_data(t, spike, pos,bin_size=25)

    D=compute_D(vel_)

    pop = NeuroPop(n_neurons=n_neurons, tunemodel='glm')

    X_train, X_test, Y_train, Y_test=train_test_split(D,spikes_,train_size=0.75)

    print("X_train.shape=",X_train.shape)
    print("Y_train.shape=",Y_train.shape)

    pop.fit(X_train, Y_train)

    Yhat_test = pop.predict(X_test)

    Ynull = np.mean(Y_train, axis=0)
    pseudo_R2 = pop.score(Y_test, Yhat_test,Ynull, method='pseudo_R2')
    print(pseudo_R2)

    plt.figure(figsize=[15, 15])

    for neuron in range(12):
        plt.subplot(4, 3, neuron + 1)
        pop.display(X_test, Y_test[:, neuron], neuron=neuron,
                    ylim=[0.8 * np.min(Y_test[:, neuron]), 1.2 *
                          np.max(Y_test[:, neuron])])

    plt.show()


if __name__ == '__main__':
    data_path='Zenodo/indy_20161024.mat'
    # show_single(data_path)enen1
    # show_pop(data_path)
    show_tuning(data_path=data_path,n_neurons=107)

