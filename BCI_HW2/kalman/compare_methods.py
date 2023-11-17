#%%
#Import standard packages
import os

import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
#import  gaussian_filter1d
from scipy.ndimage import gaussian_filter1d
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
from models import LSTM,LinearRegression


import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from tqdm import tqdm
import numpy as np
import json
import scipy.io as scio
# from format_data import *
# from metrics.metric import *
# from model.net import build_model



def process_neuraldata(path,dt=.01,redo=False,delunsort=True):
    # %%
    ###Load Data###
    # path = r'/Users/macbookpro/Desktop/[Dataset]Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology/indy_20160921_01.mat'  # ENTER THE FOLDER THAT YOUR DATA IS IN
    # folder='/home/jglaser/Data/DecData/'
    data_folder = ''  # FOLDER YOU WANT TO SAVE THE DATA TO
    new_path = data_folder+'processed_'+path.split("/")[-1]
    if os.path.exists(new_path) and not redo:
        return new_path

    try:
        data = io.loadmat(path)
    except:
        data = mat73.loadmat(path)
    spike_times = data['spikes']  # Load spike times of all neurons
    pos = data['cursor_pos']  # Load x and y positions
    pos_times = data['t']  # Load times at which positions were recorded
    # %%
    # dt = .01  # Size of time bins (in seconds)
    t_start = pos_times[0]  # Time to start extracting data - here the first time position was recorded
    t_end = pos_times[-1] #Time to finish extracting data - when looking through the dataset, the final
    # position was recorded around t=5609, but the final spikes were recorded around t=5608
    downsample_factor = 1  # Downsampling of output (to make binning go faster). 1 means no downsampling.
    # %%
    # When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
    # First, we will put spike_times in a cleaner format: an array of arrays
    if delunsort:
        spike_times = np.squeeze(spike_times)[:,1:].flatten()
    else:
        spike_times = np.squeeze(spike_times).flatten()
    for i in range(spike_times.shape[0]):
        spike_times[i] = np.squeeze(spike_times[i])
    # %%
    ###Preprocessing to put spikes and output in bins###

    # Bin neural data using "bin_spikes" function
    neural_data = bin_spikes(spike_times, dt, t_start, t_end)
    # Erase zeros
    neural_data = neural_data[:, neural_data.sum(axis=0) >10]
    # Bin output (position) data using "bin_output" function
    pos_binned = bin_output(pos, pos_times, dt, t_start, t_end, downsample_factor)
    # %%

    import pickle


    io.savemat(new_path, {'neural_data': neural_data, 'pos_binned': pos_binned})
    return new_path

if __name__ =='__main__':
    # %%original data
    path =  r'/Users/macbookpro/Desktop/Datasets/[Dataset]Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology/indy_20160407_02.mat'  # ENTER THE FOLDER THAT YOUR DATA IS IN
    dt=0.04
    #%%  monkey
    # dataFile = r'Examples_hippocampus/processed_indy_20160921_01.mat'
    dataFile = process_neuraldata(path,dt=dt,redo=True,delunsort=False)
    data = scio.loadmat(dataFile)



    neural_data =data['neural_data']
    #gaussian1d
    # sigma = 1.5
    # neural_data = gaussian_filter1d(neural_data,sigma=sigma,axis=0)
    pos_binned = data['pos_binned']

    #%%
    flg_plot = False
    lag=0 #What time bin of spikes should be used relative to the output
    #(lag=-1 means use the spikes 1 bin before the output)

    #%%
    #The covariate is simply the matrix of firing rates for all neurons over time
    X_kf=neural_data
    temp = np.diff(pos_binned, axis=0)/dt
    vels_binned = np.concatenate((temp, temp[-1:, :]),
                                axis=0)

    # We will now determine acceleration
    temp = np.diff(vels_binned, axis=0)/dt  # The acceleration is the difference in velocities across time bins
    acc_binned = np.concatenate((temp, temp[-1:, :]),
                                axis=0)  # Assume acceleration at last time point is same as 2nd to last

    # The final output covariates include position, velocity, and acceleration
    y_kf = np.concatenate((pos_binned, vels_binned, acc_binned), axis=1)
    # y_kf = acc_binned
    # %%
    num_examples = X_kf.shape[0]

    # Re-align data to take lag into account
    if lag < 0:
        y_kf = y_kf[-lag:, :]
        X_kf = X_kf[0:num_examples + lag, :]
    if lag > 0:
        y_kf = y_kf[0:num_examples - lag, :]
        X_kf = X_kf[lag:num_examples, :]
    # %%
    # Set what part of data should be part of the training/testing/validation sets
    training_range = [0, 0.3]
    testing_range = [0.7, 0.85]
    valid_range = [0.4,0.5]
    # %%
    # Number of examples after taking into account bins removed for lag alignment
    num_examples_kf = X_kf.shape[0]

    # Note that each range has a buffer of 1 bin at the beginning and end
    # This makes it so that the different sets don't include overlapping data
    training_set = np.arange(np.int(np.round(training_range[0] * num_examples_kf)) + 1,
                             np.int(np.round(training_range[1] * num_examples_kf)) - 1)
    testing_set = np.arange(np.int(np.round(testing_range[0] * num_examples_kf)) + 1,
                            np.int(np.round(testing_range[1] * num_examples_kf)) - 1)
    valid_set = np.arange(np.int(np.round(valid_range[0] * num_examples_kf)) + 1,
                          np.int(np.round(valid_range[1] * num_examples_kf)) - 1)

    # Get training data
    X_kf_train = X_kf[training_set, :]
    y_kf_train = y_kf[training_set, :]

    # Get testing data
    X_kf_test = X_kf[testing_set, :]
    y_kf_test = y_kf[testing_set, :]

    # Get validation data
    X_kf_valid = X_kf[valid_set, :]
    y_kf_valid = y_kf[valid_set, :]
    # %%
    # Z-score inputs
    X_kf_train_mean = np.nanmean(X_kf_train, axis=0)
    X_kf_train_std = np.nanstd(X_kf_train, axis=0)
    X_kf_train = (X_kf_train - X_kf_train_mean) / X_kf_train_std
    X_kf_test = (X_kf_test - X_kf_train_mean) / X_kf_train_std
    X_kf_valid = (X_kf_valid - X_kf_train_mean) / X_kf_train_std

    # Zero-center outputs
    y_kf_train_mean = np.mean(y_kf_train, axis=0)
    y_kf_train = y_kf_train - y_kf_train_mean
    y_kf_test = y_kf_test - y_kf_train_mean
    y_kf_valid = y_kf_valid - y_kf_train_mean
    # %%
    # Declare model
    model = KalmanFilterDecoder(
        C=1)  # There is one optional parameter that is set to the default in this example (see ReadMe)
    model = LSTM()
    # model = LinearRegression()
    # Fit model

    # train(X_kf_train,y_kf_train,X_kf_valid,y_kf_valid)
    model.fit(X_kf_train, y_kf_train)

    # Get predictions
    y_valid_predicted_kf = model.predict(X_kf_valid, y_kf_valid)

    # Get metrics of fit (see read me for more details on the differences between metrics)
    # First I'll get the R^2
    R2_kf = get_R2(y_kf_valid, y_valid_predicted_kf)
    print('R2:', R2_kf)  # I'm just printing the R^2's of the 3rd and 4th entries that correspond to the velocities
    # Next I'll get the rho^2 (the pearson correlation squared)
    rho_kf = get_rho(y_kf_valid, y_valid_predicted_kf)
    print('rho2:',rho_kf ** 2)  # I'm just printing the rho^2's of the 3rd and 4th entries that correspond to the velocities
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
    result = np.vstack((R2_kf, rho_kf)).transpose().flatten()
    print(result)



