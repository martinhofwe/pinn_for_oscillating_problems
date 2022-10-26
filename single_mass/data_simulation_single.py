import sys

import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle

def simulate_single_mass(start_vector,  end_time, steps, m1_in, css_in, dss_in):
    np.random.seed(1234)
    tf.random.set_seed(1234)

    start_vec = start_vector

    cSS = css_in #0.5e6 * 2  # Stiffness Secondary Suspension
    dSS = dss_in #1.5e4 * 2  # Stiffness Primary Suspension
    mWK = m1_in

    cSF = cSS
    dSF = dSS

    m1 = mWK
    c = cSF # todo überprüfuen und vgl mit 2 masse implementierung
    d = dSF
    #c = (cSF / mWK) # todo überprüfuen und vgl mit 2 masse implementierung
    #d = (dSF / mWK)

    # todo remove
    '''
    Av = 5.9233e-7;
    Omc = 0.8246;
    Omr = 0.0206;
    Om = np.logspace(-2, 1, 100)
    Fs = 200
    tExci = np.linspace(0, 10, 4000)  # time
    Fs_Exci = 5 * Fs;
    tExci_raw = np.linspace(0, 25, 25001)
    frequ = np.hstack((np.linspace(0.05, 1.95, 40), np.linspace(2, 15, 100)))
    # for f1 in frequ:
    f1 = 15.
    Sz = (Av * Omc ** 2.) / ((f1 ** 2. + Omr ** 2) * (f1 ** 2 + Omc ** 2))
    z_raw = Sz * np.sin(tExci_raw * f1 * 2 * np.pi)
    z_raw = z_raw * scale_factor

    u_raw = z_raw
    up_raw = np.diff(u_raw, prepend=0)
    # up_raw = np.gradient(u_raw)# gradient vs diff?
    u_orig = np.expand_dims(u_raw, -1)
    up_orig = np.expand_dims(np.diff(np.squeeze(u_raw), prepend=0), -1)

    u_orig = np.expand_dims(np.interp(tExci, tExci_raw, u_raw), 1)  # todo why interpolation?
    up_orig = np.expand_dims(np.interp(tExci, tExci_raw, up_raw), 1)

    tExci = np.expand_dims(tExci, 1)
    u_orig = np.zeros_like(tExci)
    up_orig = np.zeros_like(u_orig)
    '''
    ################################################################################
    tExci = np.linspace(0, end_time, steps)  # time

    u_orig = np.zeros_like(tExci)
    up_orig = np.zeros_like(u_orig)

    A = np.array([
        [0, 1],
        [-cSF / mWK, -dSF / mWK]])

    B = np.array(
        [[0, 0],
         [cSF / mWK, dSF / mWK]])

    C = np.eye(2)# np.array([[1, 1]])  # extract x1 = position
    D = np.zeros((2, 2))#np.zeros((1, 2))

    #C = np.array([[1, 0]])  # extract x1 = position
    #D = np.zeros((1, 2))

    OneMassOsci = sig.StateSpace(A, B, C, D)

    sys_input = np.hstack((u_orig, up_orig))

    tsim_nom_orig, y_orig, xsim_nom_orig = sig.lsim(OneMassOsci, sys_input, np.squeeze(tExci), X0=start_vec)

    return tsim_nom_orig, y_orig, xsim_nom_orig, [m1, c, d]

def get_data(y_orig_all, r_debug, exp_len, time_step):

      y_m1 = np.expand_dims(y_orig_all[:exp_len,0], 1)
      y_m1_dx = np.expand_dims(y_orig_all[:exp_len, 1], 1)
      #y_m1 = np.expand_dims(y_orig_all[:exp_len], 1)

      u = np.zeros_like(y_m1)[:exp_len]
      up = np.zeros_like(u)[:exp_len] # todo get as input
      if r_debug:
        y_m1_dx2 = np.expand_dims(np.gradient(y_orig_all[:exp_len, 1]) / time_step,1)
        print(y_m1.shape)
        print(y_m1_dx.shape)
        print(y_m1_dx2.shape)
        return y_m1, y_m1_dx, y_m1_dx2, u, up

      else:
          return y_m1, None, None, u, up

def get_simulated_data_single_mass(start_vector, end_time=20, steps=4001, exp_len=400,m1=15000, css=0.5e6 * 2,dss=1.5e4 * 2, debug_data=True):
    tsim_nom_orig, y_orig_all, xsim_nom_orig, simul_const = simulate_single_mass(start_vector, end_time, steps,m1, css, dss)
    y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out = get_data(y_orig_all, debug_data, exp_len=exp_len, time_step=end_time/steps)
    return y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out, np.expand_dims(tsim_nom_orig[:exp_len], 1), simul_const# + tsim_nom_orig #todo remove exp len








