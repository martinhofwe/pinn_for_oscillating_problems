import sys

import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle



def plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, y_pred, f_path_name, epoch=None):
    fig_res = plt.figure(figsize=(16, 9))
    if epoch is not None:
        fig_res.suptitle(str(epoch))
    plt.subplot(2, 1, 1)
    if y_pred is not None:
        plt.plot(t, y_pred[:, 1], label="pred pos m2 (wagon)")
    plt.plot(t, y_lbl_m2, label="exact solution pos m2 (wagon)")
    plt.scatter(x_data, y_data_m2, label="data points", c="r")
    plt.legend()
    # plt.show()
    plt.subplot(2, 1, 2)
    if y_pred is not None:
        plt.plot(t, y_pred[:, 0], label="pred pos m1")
    plt.plot(t, y_lbl_m1, label="exact solution m1")
    plt.scatter(x_data, y_data_m1, label="data points", c="r")
    plt.legend()
    if epoch is not None:
        fig_res.savefig(f_path_name + str(epoch) + '.svg', format='svg', dpi=1200)
    else:
        fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)


def plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_orig_m2_dx, y_m1_dx, y_orig_m1_dx, y_m2_dx2, y_m2_dx2_calc, y_m1_dx2, y_m1_dx2_calc,  f_path_name):
    fig_res = plt.figure(figsize=(12, 12))
    plt.subplot(6, 1, 1)
    plt.plot(t, y_m2, label="m2")
    plt.plot(t, y_lbl_m2, label="m2 matlab")
    plt.legend()
    plt.subplot(6, 1, 2)
    plt.plot(t, y_m1, label="m1")
    plt.plot(t, y_lbl_m1, label="m1 matlab")
    plt.legend()
    # plt.show()
    plt.subplot(6, 1, 3)
    plt.plot(t, y_m2_dx, label="m2 dx")
    plt.plot(t, y_orig_m2_dx, label="m2 dx matlab")
    plt.legend()
    plt.subplot(6, 1, 4)
    plt.plot(t, y_m1_dx, label="m1 dx")
    plt.plot(t, y_orig_m1_dx, label="m1 dx matlab")
    plt.legend()
    plt.subplot(6, 1, 5)
    plt.plot(t, y_m2_dx2, label="m2 dx2")
    plt.plot(t, y_m2_dx2_calc, label="m2 dx2 matlab")
    plt.legend()
    plt.subplot(6, 1, 6)
    plt.plot(t, y_m1_dx2, label="m1 dx2")
    plt.plot(t, y_m1_dx2_calc, label="m1 dx2 matlab")
    plt.legend()
    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)


def plot_loss(loss_over_epoch, physics_scale, f_path_name, scaled):
    fig_s = plt.figure()
    plt.title("Loss over meta epochs, scaled=" +str(scaled))

    plt.plot([l[0] for l in loss_over_epoch], label='data_error m1')
    plt.plot([l[1] for l in loss_over_epoch], label='data_error m2')

    if scaled:
        plt.plot([l[2] * physics_scale for l in loss_over_epoch], label='physics_error m1')
        plt.plot([l[3] * physics_scale for l in loss_over_epoch], label='physics_error m2')
    else:
        plt.plot([l[2] for l in loss_over_epoch], label='physics_error m1')
        plt.plot([l[3] for l in loss_over_epoch], label='physics_error m2')
    plt.yscale("log")
    plt.legend()
    fig_s.savefig(f_path_name + '.svg', format='svg', dpi=1200)

def plot_loss_scale(loss_over_epoch, physics_scale, f_path_name, scaled, p1_scale, d1_scale):
    fig_s = plt.figure()
    plt.title("Loss over meta epochs, scaled=" +str(scaled))

    plt.plot([l[0] * d1_scale for l in loss_over_epoch], label='data_error m1')
    plt.plot([l[1] for l in loss_over_epoch], label='data_error m2')

    if scaled:
        plt.plot([l[2] * physics_scale * p1_scale for l in loss_over_epoch], label='physics_error m1')
        plt.plot([l[3] * physics_scale for l in loss_over_epoch], label='physics_error m2')
    else:
        plt.plot([l[2] for l in loss_over_epoch], label='physics_error m1')
        plt.plot([l[3] for l in loss_over_epoch], label='physics_error m2')
    plt.yscale("log")
    plt.legend()
    fig_s.savefig(f_path_name + '.svg', format='svg', dpi=1200)


def plot_terms_diff(t, f_pred_m1, f_pred_m2, y_pred, y_lbl_all, f_path_name, p_plot_start=50):
    fig_res = plt.figure(figsize=(12, 12))
    plt.subplot(4, 1, 1)
    plt.plot(t[p_plot_start:], f_pred_m2[p_plot_start:], label="p error  m2")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, y_pred[:, 1] - y_lbl_all[:, 1], label="d error  m2")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t[p_plot_start:], f_pred_m1[p_plot_start:], label="p error  m1")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t, y_pred[:, 0] - y_lbl_all[:, 0], label="d error  m1")
    plt.legend()
    fig_res.savefig(f_path_name + '.svg', format='svg', dpi=1200)




