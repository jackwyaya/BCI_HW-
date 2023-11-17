#%%
#Import standard packages
import os

import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle

# from Examples_hippocampus.format_data import process_neuraldata
# If you would prefer to load the '.h5' example file rather than the '.pickle' example file. You need the deepdish package
# import deepdish as dd

#Import metrics
from Neural_Decoding.preprocessing_funcs import *
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho
import scipy.io as scio

#Import decoder functions
from Neural_Decoding.decoders import KalmanFilterDecoder
from data.data_preprocessing import load_data
from default import DT
from scipy.ndimage import gaussian_filter1d



if __name__ =='__main__':
    # %%original data
    dataset_name = 'indy_20160921_01.mat'
    DT = 0.05
    spike_binned, (pos_binned, vels_binned, acc_binned) = load_data(dataset_name= dataset_name,redo=False,dt = DT)
    #gaussian smooth 
    sigma = 3
    spike_binned = gaussian_filter1d(spike_binned,sigma=sigma,axis=0)
    #%%
    flg_plot = False
    lag=int(0.1/DT)# 100ms=0.1s/DT #What time bin of spikes should be used relative to the output
    lag = 0
    #(lag=-1 means use the spikes 1 bin before the output)

    #%%
    X_kf=spike_binned
    y_kf = np.concatenate((pos_binned, vels_binned, acc_binned), axis=1)
    time_length = X_kf.shape[0]

    # Re-align data to take lag into account
    if lag < 0:
        y_kf = y_kf[-lag:, :]
        X_kf = X_kf[0:time_length + lag, :]
    if lag > 0:
        y_kf = y_kf[0:time_length - lag, :]
        X_kf = X_kf[lag:time_length, :]
    # %%

    training_range = [0, 0.7]
    testing_range = [0.7, 0.85]
    #TODO fix
    # valid_range = [0,0.7]
    valid_range = [0.7,1.0]
    (X_kf_train, y_kf_train), (X_kf_valid, y_kf_valid), (X_kf_test, y_kf_test),(X_kf_train_mean,y_kf_train_mean) = split_data((X_kf,y_kf),training_range,valid_range,testing_range)
    #%%
    # Declare model
    #TODO C=1?
    model_kf = KalmanFilterDecoder(
        C=1)  # There is one optional parameter that is set to the default in this example (see ReadMe)
    model_kf.fit(X_kf_train, y_kf_train)

    # Get predictions
    y_valid_predicted_kf = model_kf.predict(X_kf_valid, y_kf_valid)
    R2_kf = get_R2(y_kf_valid, y_valid_predicted_kf)
    print('R2:', R2_kf) 
    rho_kf = get_rho(y_kf_valid, y_valid_predicted_kf)
    print('rho2:',rho_kf)  # I'm just printing the rho^2's of the 3rd and 4th entries that correspond to the velocities
    if flg_plot:
        fig_x_kf = plt.figure()
        plt.plot(y_kf_valid[:, 0] + y_kf_train_mean[0], 'b')
        plt.plot(y_valid_predicted_kf[:, 0] + y_kf_train_mean[0], 'r')
        plt.show()

        fig_y_kf = plt.figure()
        plt.plot(y_kf_valid[:, 1] + y_kf_train_mean[1], 'b')
        plt.plot(y_valid_predicted_kf[:, 1] + y_kf_train_mean[1], 'r')
        plt.show()


        plt.plot(y_kf_valid[:, 0]+y_kf_train_mean[0],y_kf_valid[:, 1]+y_kf_train_mean[1], 'b')
        plt.plot(y_valid_predicted_kf[:, 0]+y_kf_train_mean[0], y_valid_predicted_kf[:, 1]+y_kf_train_mean[1], 'r')
        plt.show()
    # result = np.vstack((R2_kf, rho_kf)).transpose().flatten()
    # print(result)



