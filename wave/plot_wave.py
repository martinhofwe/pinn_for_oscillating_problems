import sys

#import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle

#######################################################################################################################
"""
Created on Tue Nov 13 17:04:43 2018

@author: bmoseley
"""


# This module defines various plotting helper variables and functions.
# taken from: https://github.com/benmoseley/seismic-simulation-complex-media/blob/701d3af7ac84120202bc9a741bfdf2320ad06ddc/shared_modules/plot_utils.py

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

rkb = {'red':     ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

rgb = {'red':     ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

rkb = LinearSegmentedColormap('RedBlackBlue', rkb)
rgb = LinearSegmentedColormap('RedGreenBlue', rgb)


def fig2rgb_array(fig, expand=False):
    fig.canvas.draw()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(shape)
VLIM = 0.6
CLIM = (1500,3600)
############################################################################################################
def plot_solution(t, velocity, wavefields, f_path_name):
    selected_timesteps = [0, 25, 50, 75, 100]
    #for s in selected_timesteps:
    #    print("step: ", t[s])
    wavefields = wavefields.reshape(wavefields.shape[0], 300, 300) # todo not hardcoded
    fig_res = plt.figure(figsize=(10,3))
    ##
    s = 5 * np.mean(np.abs(wavefields))

    for count, step in enumerate(selected_timesteps):
        plt.subplot(1, len(selected_timesteps), (count + 1))
        plt.imshow(-wavefields[step, :, :].T, vmin=-s, vmax=s)
        plt.gca().set_anchor('C')  # centre plot
        plt.axis('off')

    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)

def plot_comparison(considered_times, wavefields, wavefields_pred,  f_path_name):

    #for s in selected_timesteps:
    #    print("step: ", t[s])
    wavefields = wavefields.reshape(len(considered_times), 300, 300) # todo not hardcoded
    wavefields_pred = wavefields_pred.numpy().reshape(len(considered_times), 300, 300)
    assert wavefields.shape == wavefields_pred.shape

    fig_res = plt.figure(figsize=(20,9))
    s = 5 * np.mean(np.abs(wavefields))
    subfigs = fig_res.subfigures(nrows=3, ncols=1)
    subfigs[0].suptitle("Simulated")

    ax1 = subfigs[0].subplots(nrows=1, ncols=len(considered_times))
    for count, ax in enumerate(ax1):
        ax.set_title("t= {:.2f}".format(considered_times[count]) + " s")
        ax.imshow(-wavefields[count, :, :].T, vmin=-s, vmax=s)
        #ax.gca().set_anchor('C')  # centre plot
        ax.axis('off')
    subfigs[1].suptitle("Prediction")
    ax2 = subfigs[1].subplots(nrows=1, ncols=len(considered_times))
    for count, ax in enumerate(ax2):
        #ax.set_title("t= {:.2f}".format(t[step[count]]) + " s")
        ax.imshow(-wavefields_pred[count, :, :].T, vmin=-s, vmax=s)
        #ax.gca().set_anchor('C')  # centre plot
        ax.axis('off')
    subfigs[2].suptitle("Difference")
    ax3 = subfigs[2].subplots(nrows=1, ncols=len(considered_times))
    for count, ax in enumerate(ax3):
        #ax.set_title("t= {:.2f}".format(t[step[count]]) + " s")
        ax.imshow(-wavefields[count, :, :].T+wavefields_pred[count, :, :].T, vmin=-s, vmax=s)
        ax.axis('off')
    fig_res.savefig(f_path_name + '_comp.svg', format='svg', dpi=1200)


def plot_loss(loss_over_epoch, physics_scale, f_path_name, scaled):
    fig_s = plt.figure()
    plt.title("Loss over meta epochs, scaled=" +str(scaled))

    plt.plot([l[0] for l in loss_over_epoch], label='data_error')

    if scaled:
        plt.plot([l[1] * physics_scale for l in loss_over_epoch], label='physics_error m')
    else:
        plt.plot([l[1] for l in loss_over_epoch], label='physics_error')
    plt.yscale("log")
    plt.legend()
    fig_s.savefig(f_path_name + '.svg', format='svg', dpi=1200)




