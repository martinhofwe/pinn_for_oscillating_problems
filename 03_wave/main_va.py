"""
********************************************************************************
main file to execute
********************************************************************************
"""
import sys
import pickle
import os
#
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pinn_va import PINN
from config_gpu import config_gpu
from params import params
from prp_dat import prp_dat
from plot_sol import *
from fdm import FDM

def main():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #task_id = int(sys.argv[1])
    
    # gpu confiuration
    config_gpu(gpu_flg = 1)

    # params
    f_in, f_out, width, depth, \
    w_init, b_init, act, \
    lr, opt, \
    f_scl, laaf, c, \
    w_ini, w_bnd, w_pde, BC, \
    f_mntr, r_seed, \
    n_epch, n_btch, c_tol = params()

    if task_id < 5:
        act = "tanh"
    elif 5 <= task_id < 10 :
        act = "sine"
    elif 10 <= task_id < 15:
        act = "sine_all"

    test_time_idx = [0, 100, 200, 300, 400, 500]

    # domain
    tmin = 0.; tmax = 10.; nt = int(5e2) + 1
    xmin = 0.; xmax =  5.; nx = int(1e2) + 1
    ymin = 0.; ymax =  5.; ny = int(1e2) + 1
    t_ = np.linspace(tmin, tmax, nt)
    x_ = np.linspace(xmin, xmax, nx)
    y_ = np.linspace(ymin, ymax, ny)
    dt = t_[1] - t_[0]
    dx = x_[1] - x_[0]
    dy = y_[1] - y_[0]
    cfl = c * dt / dx
    print("CFL number:", cfl)

    x, y = np.meshgrid(x_, y_)
    u    = np.empty((nt, nx, ny))
    print("tmin: %.3f, tmax: %.3f, nt: %d, dt: %.3e" % (tmin, tmax, nt, dt))
    print("xmin: %.3f, xmax: %.3f, nx: %d, dx: %.3e" % (xmin, xmax, nx, dx))
    print("ymin: %.3f, ymax: %.3f, ny: %d, dy: %.3e" % (ymin, ymax, ny, dy))

    # FDM simulation
    u_FDM = FDM(xmin, xmax, nx, dx, 
                ymin, ymax, ny, dy, 
                nt, dt, 
                x, y, u, c, BC)
    

    # prep data
    TX, lb, ub, \
    t_ini, x_ini, y_ini, u_ini, \
    t_bndx, x_bndx, y_bndx, \
    t_bndy, x_bndy, y_bndy, \
    t_pde, x_pde, y_pde = prp_dat(t_, x_, y_, 
                                    N_ini = int(1e4), 
                                    N_bnd = int(1e4), 
                                    N_pde = int(5e4))

    pinn = PINN(t_ini, x_ini, y_ini, u_ini, 
                t_bndx, x_bndx, y_bndx, 
                t_bndy, x_bndy, y_bndy, 
                t_pde, x_pde, y_pde, 
                f_in, f_out, width, depth,
                TX, u_FDM, test_time_idx,
                w_init, b_init, act, 
                lr, opt, 
                f_scl, laaf, c, 
                w_ini, w_bnd, w_pde, BC, 
                f_mntr, r_seed, )
    
        ################################
    result_folder_name = 'res_wave_jap'
    os.makedirs(result_folder_name, exist_ok=True)
    
    experiment_name = "wave_va_" + str(depth) + "_w_" + str(
        width) + "_af_" + act + "_lr_" + str(lr) + "_ps_" + str(w_pde) + "_id_" + str(task_id)
    print("Config name: ", experiment_name)
    os.makedirs(result_folder_name + "/" + experiment_name, exist_ok=True)
    os.makedirs(result_folder_name + "/" + experiment_name + "/plots_2d", exist_ok=True)
    os.makedirs(result_folder_name + "/" + experiment_name + "/plots_3d", exist_ok=True)
    plots_path = result_folder_name + "/" + experiment_name + "/"
    
    ##########################################
    '''
    fig = plt.figure(figsize = (16, 16))
    ax  = fig.add_subplot(3, 1, 1, projection = "3d")
    plt.axis('off')
   
    surf = ax.plot_surface(x, y, u_FDM[0,:,:], cmap = "coolwarm", 
                                linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (t, x, y)", fontstyle = "italic")
    ax  = fig.add_subplot(3, 1, 2, projection = "3d")
    plt.axis('off')
    surf = ax.plot_surface(x, y, u_FDM[100,:,:], cmap = "coolwarm", 
                                linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (t, x, y)", fontstyle = "italic")
    
    ax  = fig.add_subplot(3, 1, 3, projection = "3d")
    plt.axis('off')
    surf = ax.plot_surface(x, y, u_FDM[200,:,:], cmap = "coolwarm", 
                                linewidth = 0, vmin = -.5, vmax = .5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x", fontstyle = "italic")
    ax.set_ylabel("y", fontstyle = "italic")
    ax.set_zlabel("u (t, x, y)", fontstyle = "italic")
    
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(plots_path +'sol.png', format='png', dpi=1200, bbox_inches='tight')
    assert False
    '''
    #################################

    t0 = time.time()
    with tf.device("/device:GPU:0"):
        pinn.train(epoch = n_epch, batch = n_btch, tol = c_tol)
    t1 = time.time()
    elps = t1 - t0
    print(">>>>> elapse time for training (sec):", elps)
    print(">>>>> elapse time for training (min):", elps / 60.)

    '''
    # inference
    x_inf, y_inf = np.meshgrid(x_, y_)
    x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
    elps = 0
    for t in t_:
        t_inf = np.ones_like(x_inf) * t
        t0 = time.time()
        u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)
        t1 = time.time()
        temp = t1 - t0
        elps += temp
    print(">>>>> elapse time for inference (sec):", elps)
    print(">>>>> elapse time for inference (min):", elps / 60.)
    '''
    
    # x_inf = np.unique(TX[:,1:2])
    # y_inf = np.unique(TX[:,2:3])
    # x_inf, y_inf = np.meshgrid(x_inf, y_inf)
    # x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
    # elps = 0.
    # for n in range(nt):
    #     if n % 100 == 0:
    #         print("currently", n)
    #     t = n * dt   # convert to real time
    #     u_fdm = u_FDM[n,:,:]
    #     n = np.array([n])
    #     t_inf = np.unique(TX[:,0:1])
    #     t_inf = np.tile(t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:,n]
    #     t0 = time.time()
    #     u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)
    #     t1 = time.time()
    #     temp = t1 - t0
    #     elps += temp
    # print(">>>>> elapse time for inference (sec):", elps)
    # print(">>>>> elapse time for inference (min):", elps / 60.)

    fig_l = plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_log,     alpha = .7, linestyle = "-",  label = "loss", c = "k")
    # plt.plot(pinn.ep_log, pinn.loss_ini_log, alpha = .5, linestyle = "--", label = "loss_ini")
    # plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .5, linestyle = "--", label = "loss_bnd")
    # plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .5, linestyle = "--", label = "loss_pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    #plt.show()
    fig_l.savefig(plots_path +'loss_log.svg', format='svg', dpi=1200)

    fig_l = plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_d_test,     alpha = .7, linestyle = "-",  label = "loss", c = "k")
    # plt.plot(pinn.ep_log, pinn.loss_ini_log, alpha = .5, linestyle = "--", label = "loss_ini")
    # plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .5, linestyle = "--", label = "loss_bnd")
    # plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .5, linestyle = "--", label = "loss_pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    #plt.show()
    fig_l.savefig(plots_path +'loss_d_test.svg', format='svg', dpi=1200)

    fig_l = plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_p_test,     alpha = .7, linestyle = "-",  label = "loss", c = "k")
    # plt.plot(pinn.ep_log, pinn.loss_ini_log, alpha = .5, linestyle = "--", label = "loss_ini")
    # plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .5, linestyle = "--", label = "loss_bnd")
    # plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .5, linestyle = "--", label = "loss_pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    #plt.show()
    fig_l.savefig(plots_path +'loss_p_test.svg', format='svg', dpi=1200)
    
    fig_l = plt.figure(figsize = (8, 4))
    plt.plot(pinn.ep_log, pinn.loss_log,     alpha = .7, linestyle = "-",  label = "loss", c = "k")
    plt.plot(pinn.ep_log, pinn.loss_ini_log, alpha = .5, linestyle = "--", label = "loss_ini")
    plt.plot(pinn.ep_log, pinn.loss_bnd_log, alpha = .5, linestyle = "--", label = "loss_bnd")
    plt.plot(pinn.ep_log, pinn.loss_pde_log, alpha = .5, linestyle = "--", label = "loss_pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    #plt.show()
    fig_l.savefig(plots_path +'loss_log_all.svg', format='svg', dpi=1200)
    
    ###################### save logged losses #######################################
    with open(result_folder_name + "/" + experiment_name + "/loss_log.pkl", "wb") as fp:
        pickle.dump(pinn.loss_log, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_ini_log.pkl", "wb") as fp:
        pickle.dump(pinn.loss_ini_log, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_bnd_log.pkl", "wb") as fp:
        pickle.dump(pinn.loss_bnd_log, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_pde_log.pkl", "wb") as fp:
        pickle.dump(pinn.loss_pde_log, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_d_test.pkl", "wb") as fp:
        pickle.dump(pinn.loss_d_test, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_p_test.pkl", "wb") as fp:
        pickle.dump(pinn.loss_p_test, fp)
    #################################################################################

    for tm in range(nt):
        if tm % 100 == 0:
            tm = np.array([tm])

            t_inf = np.unique(TX[:,0:1])
            x_inf = np.unique(TX[:,1:2])
            y_inf = np.unique(TX[:,2:3])

            x_inf, y_inf = np.meshgrid(x_inf, y_inf)
            x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
            t_inf = np.tile(t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:,tm]

            u_hat, gv_hat = pinn.infer(t_inf, x_inf, y_inf)

            fig  = plt.figure(figsize = (6, 6))
            ax   = fig.add_subplot(1, 1, 1, projection = "3d")
            surf = ax.plot_surface(x, y, tf.reshape(u_hat, shape = [nx, ny]), cmap = "coolwarm", 
                                linewidth = 0, vmin = -.5, vmax = .5)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(-1, 1)
            ax.set_xlabel("x", fontstyle = "italic")
            ax.set_ylabel("y", fontstyle = "italic")
            ax.set_zlabel("u (t, x, y)", fontstyle = "italic")
            #plt.show()
            
            fig.savefig(plots_path +'plots_3d/res_3d_' + str(tm) +'.svg', format='svg', dpi=1200)
            
            nr_ticks = 5
            tick_spacing = x.shape[0] // nr_ticks
            fig_res = plt.figure(figsize=(10,10))
            plt.imshow(tf.reshape(u_hat, shape = [nx, ny]), cmap = "coolwarm", vmin = -.5, vmax = .5)
            plt.gca().set_anchor('C')  # centre plot
            plt.xticks(ticks=np.arange(0, x.shape[0], 1)[::tick_spacing], labels=[str(tick) for tick in x[0, :]][::tick_spacing])
            plt.yticks(ticks=np.arange(0, x.shape[1], 1)[::tick_spacing], labels=[str(tick) for tick in y[:, 0]][::tick_spacing])
            fig_res.savefig(plots_path +'plots_2d/res_2d_' + str(tm) +'.svg', format='svg', dpi=1200)
            plt.close('all')

    # for n in range(nt):
    #     if n % (int(nt / 5)) == 0:
    #         t = n * dt   # convert to real time
            # u_fdm = u_FDM[n,:,:]
    #         n = np.array([n])
    #         t_inf = np.unique(TX[:,0:1])
    #         x_inf = np.unique(TX[:,1:2])
    #         y_inf = np.unique(TX[:,2:3])
    #         x_inf, y_inf = np.meshgrid(x_inf, y_inf)
    #         x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
    #         t_inf = np.tile(t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:,n]
    #         u_, gv_ = pinn.infer(t_inf, x_inf, y_inf)

    #         fig = plt.figure(figsize=(16, 4))

    #         ax = fig.add_subplot(1, 1, 1, projection = "3d")
    #         ax.plot_surface(x, y, u_fdm, cmap="coolwarm", vmin = -1., vmax = 1.)
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)
    #         ax.set_zlim(-1., 1.)

    #         ax = fig.add_subplot(1, 2, 2, projection = "3d")
    #         ax.plot_surface(x, y, u_.numpy().reshape(nx, ny), cmap="coolwarm", vmin = -1., vmax = 1.)
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)
    #         ax.set_zlim(-1., 1.)

    #         plt.show()

            # u_fdm = u_FDM[tm,:,:]
            # u_diff = u_fdm - u_.numpy().reshape(nx, ny)
            # u_l2  = np.linalg.norm(u_diff, ord=2) / np.linalg.norm(u_fdm, ord=2)
            # u_mse = np.mean(np.square(u_diff)) / np.sqrt(nx * ny)
            # u_sem = np.std (np.square(u_diff), ddof = 1) / np.sqrt(nx * ny)
            # print("t: %.3f, l2: %.3e, mse: %.3e, sem: %.3e" % (t, u_l2, u_mse, u_sem))

if __name__ == "__main__":
    main()
