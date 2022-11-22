import sys

import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle


def simulate_two_mass(start_vector,  end_time, steps, m1_in, m2_in, css_in, dss_in):
    np.random.seed(1234)
    tf.random.set_seed(1234)

    start_vec = start_vector
    cPS = 1e6 * 4  # Stiffness Primary Suspension
    dPS = 1e4 * 4  # Damping Primary Suspension
    mDG = m1_in
    cSS = css_in#0.5e6 * 2  # Stiffness Secondary Suspension
    dSS = dss_in #1.5e4 * 2  # Stiffness Primary Suspension
    mWK = m2_in#30000 / 2

    m2 = mWK
    m1 = mDG

    c1 = cPS
    d1 = dPS
    c2 = cSS
    d2 = dSS

    tExci = np.linspace(0, end_time, steps)  # time

    u_orig = np.zeros_like(tExci)
    up_orig = np.zeros_like(u_orig)

    ################################################################################
    # 2-Mass
    # x1 = position, x2 = velocity of m1 and x3 = position, x4 = velocity of m2
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-cSS / mWK, -dSS / mWK, cSS / mWK, dSS / mWK],
        [0.0, 0.0, 0.0, 1.0],
        [cSS / mDG, dSS / mDG, -(cSS + cPS) / mDG, -(dSS + dPS) / mDG]])

    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [cPS / mDG, dPS / mDG]])

    C = np.eye(4)
    D = np.zeros((4, 2))

    TwoMassOsci = sig.StateSpace(A, B, C, D)
    sys_input2 = np.hstack((u_orig, up_orig))

    # t2,y2,x_2 = sig.lsim(TwoMassOsci,test,np.squeeze(tExci))
    tsim_nom_orig, y_orig_all, xsim_nom_orig = sig.lsim(TwoMassOsci, sys_input2, np.squeeze(tExci), X0=start_vec)
    return tsim_nom_orig, y_orig_all, xsim_nom_orig, [m2, m1, c1, d1, c2, d2]

def get_data(y_orig_all, r_debug, exp_len, time_step):

      y_m2 = np.expand_dims(y_orig_all[:exp_len,0], 1) # careful m2 is the upper mass!
      y_m2_dx = np.expand_dims(y_orig_all[:exp_len, 1], 1)
      y_m1 = np.expand_dims(y_orig_all[:exp_len, 2], 1)
      y_m1_dx = np.expand_dims(y_orig_all[:exp_len, 3], 1)

      u = np.zeros_like(y_m2)[:exp_len]
      up = np.zeros_like(u)[:exp_len]
      if r_debug:
          y_m2_dx2 = np.expand_dims(np.gradient(y_orig_all[:exp_len, 1]) / time_step,1)
          y_m1_dx2 = np.expand_dims(np.gradient(y_orig_all[:exp_len, 3]) / time_step,1)

          return y_m2, y_m2_dx, y_m2_dx2, y_m1, y_m1_dx, y_m1_dx2, u, up
      else:
          return y_m2, None, None, y_m1, None, None, u, up

def get_simulated_data_two_mass(start_vector, end_time=20, steps=4001, exp_len=400,m1=15000, m2=15000, css=0.5e6 * 2,dss=1.5e4 * 2, debug_data=True):
    tsim_nom_orig, y_orig_all, xsim_nom_orig, simul_const = simulate_two_mass(start_vector, end_time, steps,m1, m2, css, dss)
    y_m2_out, y_m2_dx_out, y_m2_dx2_out, y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out = get_data(y_orig_all, debug_data, exp_len=exp_len, time_step=end_time/steps)
    return [y_m2_out, y_m2_dx_out, y_m2_dx2_out, y_m1_out, y_m1_dx_out, y_m1_dx2_out, u_out, up_out, np.expand_dims(tsim_nom_orig[:exp_len], 1)], simul_const# + tsim_nom_orig #todo remove exp len








