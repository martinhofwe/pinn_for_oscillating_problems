import sys

import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle


def plot_solution(t, y_lbl, x_data, y_data, y_pred, p_end,f_path_name):
    fig_res = plt.figure(figsize=(16, 9))
    if y_pred is not None:
        plt.plot(t, y_pred, label="pred pos")
    if p_end is not None:
        plt.axvline(x=t[p_end], color='r', label='physics points end')
    plt.plot(t, y_lbl, label="exact solution")
    plt.scatter(x_data, y_data, label="data points", c="r")
    plt.legend()
    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)


def plot_loss(loss_over_epoch, physics_scale, f_path_name, scaled, exp=False):
    if exp:
        data = 2
        physics = 3
    else:
        data = 0
        physics = 1
    fig_s = plt.figure()
    plt.title("Loss over meta epochs, scaled=" +str(scaled))
    plt.plot([l[data] for l in loss_over_epoch], label='data_error')

    if scaled:
        plt.plot([l[physics] * physics_scale for l in loss_over_epoch], label='physics_error')
    else:
        plt.plot([l[physics] for l in loss_over_epoch], label='physics_error')
    plt.yscale("log")
    plt.legend()
    fig_s.savefig(f_path_name + '.svg', format='svg', dpi=1200)


def plot_terms_diff(t, f_pred, y_pred, y_lbl_all, f_path_name, p_plot_start=50):
    fig_res = plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(t[p_plot_start:], f_pred, label="p error")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t, y_pred - y_lbl_all, label="d error")
    plt.legend()

    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)



def plot_terms_detail(t, y_m1, y_lbl_m1, y_m1_dx, y_orig_m1_dx, y_m1_dx2, y_m1_dx2_calc,  f_path_name):
    fig_res = plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.plot(t, y_m1, label="m2")
    plt.plot(t, y_lbl_m1, label="m2 matlab")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, y_m1_dx, label="m1")
    plt.plot(t, y_orig_m1_dx, label="m1 matlab")
    plt.legend()
    # plt.show()
    plt.subplot(3, 1, 3)
    plt.plot(t, y_m1_dx2, label="m2 dx")
    plt.plot(t, y_m1_dx2_calc, label="m2 dx matlab")
    plt.legend()
    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)




