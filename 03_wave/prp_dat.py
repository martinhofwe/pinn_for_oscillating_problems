"""
********************************************************************************
prep dataset for pinn
********************************************************************************
"""

import numpy as np
import tensorflow as tf

def prp_dat(
        t, x, y, 
        N_ini, N_bnd, N_pde
):

        # dataset for PINN
        t_star, x_star, y_star = np.meshgrid(t, x, y)
        t_star, x_star, y_star = t_star.reshape(-1, 1), x_star.reshape(-1, 1), y_star.reshape(-1, 1)
        TX  = np.c_[t_star, x_star, y_star]
        lb = tf.cast(tf.constant(tf.reduce_min(TX, axis = 0)), dtype = tf.float32)
        ub = tf.cast(tf.constant(tf.reduce_max(TX, axis = 0)), dtype = tf.float32)

        # uniform sampling for initial condition
        t_ini = tf.ones((N_ini, 1), dtype = tf.float32) * lb[0]
        x_ini = tf.random.uniform((N_ini, 1), lb[1], ub[1], dtype = tf.float32)
        y_ini = tf.random.uniform((N_ini, 1), lb[2], ub[2], dtype = tf.float32)
        u_ini = tf.exp(-(x_ini - 3) ** 2) * tf.exp(-(y_ini - 3) ** 2)

        # uniform sampling for boundary condition
        t_bndy = tf.random.uniform((int(N_bnd/2), 1), lb[0], ub[0], dtype = tf.float32)
        x_bndy = tf.random.uniform((int(N_bnd/2), 1), lb[1], ub[1], dtype = tf.float32)
        y_bndy = lb[2] + (ub[2] - lb[2]) * tf.keras.backend.random_bernoulli((int(N_bnd/2), 1), .5, dtype = tf.float32) # either lb[2] or lb2 + ub[2] - lb[2] normally ub2
        t_bndx = tf.random.uniform((int(N_bnd/2), 1), lb[0], ub[0], dtype = tf.float32)
        x_bndx = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((int(N_bnd/2), 1), .5, dtype = tf.float32)
        y_bndx = tf.random.uniform((int(N_bnd/2), 1), lb[2], ub[2], dtype = tf.float32)

        # uniform sampling for PDE residual
        t_pde = tf.random.uniform((N_pde, 1), lb[0], ub[0], dtype = tf.float32)
        x_pde = tf.random.uniform((N_pde, 1), lb[1], ub[1], dtype = tf.float32)
        y_pde = tf.random.uniform((N_pde, 1), lb[2], ub[2], dtype = tf.float32)

        return TX, lb, ub, \
                t_ini, x_ini, y_ini, u_ini, \
                t_bndx, x_bndx, y_bndx, \
                t_bndy, x_bndy, y_bndy, \
                t_pde, x_pde, y_pde

