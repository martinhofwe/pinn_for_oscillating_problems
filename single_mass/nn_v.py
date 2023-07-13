import sys
import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
import scipy.io
from numpy.random import random

from data_simulation_single import get_simulated_data_single_mass
from plot_single_mass import plot_solution, plot_terms_detail, plot_loss, plot_terms_diff

#np.random.seed(12345)
#tf.random.set_seed(12345)


class Logger(object):
    def __init__(self, save_loss_freq, print_freq=1_000):
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.start_time = time.time()
        self.save_loss_freq = save_loss_freq
        self.print_freq = print_freq
        self.loss_over_epoch = []

    def __get_elapsed(self):
        return datetime.fromtimestamp((time.time() - self.start_time)).strftime("%m/%d/%Y, %H:%M:%S")

    def __get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, pinn):
        print("\nTraining started")
        print("================")
        print(pinn.model.summary())

    def log_train_epoch(self, epoch, loss, log_data):
        if epoch % self.save_loss_freq == 0:
            data_error, physics_error = [m.numpy() for m in log_data]
            self.loss_over_epoch.append([data_error, physics_error])
        if epoch % self.print_freq == 0:
            if not epoch % self.save_loss_freq == 0:
                data_error, physics_error = [m.numpy() for m in log_data]
            #print(f"{'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e} ")
            print(f"{'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error = {data_error:.4e}  physics error = {physics_error:.4e} ")

    def log_train_end(self, epoch, log_data):
        print("==================")
        data_error, physics_error = [m.numpy() for m in log_data]
        print(
            f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error = {data_error:.4e} physics error = {physics_error:.4e}  ")


class PhysicsInformedNN(object):
    def __init__(self, layers, h_activation_function, logger, simul_constants, domain, physics_scale, lr, data,
                 simul_results, storage_path):

        inputs, outputs = self.setup_layers(layers, h_activation_function)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="concat_model")


        self.storage_path = storage_path
        self.dtype = tf.float32
        m, c, d = simul_constants
        scaling_factor = 1.0
        self.c = tf.constant(c / scaling_factor, dtype=self.dtype)
        self.d = tf.constant(d / scaling_factor, dtype=self.dtype)
        self.m = tf.constant(m / scaling_factor, dtype=self.dtype)

        self.x_ic, self.y_lbl_ic, self.x_physics, self.input_all, self.y_lbl_all = data
        self.y_m1_simul, self.y_m1_dx_simul, self.y_m1_dx2_simul, self.u_simul, self.up_simul, self.tExci = simul_results
        self.scaling_factor = tf.constant(1.0, dtype=self.dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.8, decay=0.)
        self.logger = logger
        self.physics_scale = physics_scale
        self.domain = domain
        self.storage_path = ""
        self.collocation_cuts = []

    def setup_layers(self, layers, h_activation_function):
        inputs = tf.keras.Input(shape=(layers[0],))
        x = inputs
        for count, width in enumerate(layers[1:-1]):
            if h_activation_function == "sine_all":
                print(width, ": sine af")
                x = tf.keras.layers.Dense(width, activation=tf.math.sin)(x)
            elif h_activation_function == "sine" and count == 0:
                print(width, ": sine af")
                x = tf.keras.layers.Dense(width, activation=tf.math.sin)(x)
            else:
                print(width, ": tanh af")
                x = tf.keras.layers.Dense(width, activation=tf.keras.activations.tanh)(x)
            #x = tf.keras.layers.Concatenate(axis=1)([x, inputs])
        print("Output layer:")
        print(layers[-1], ": no af")
        outputs = tf.keras.layers.Dense(layers[-1], activation=None)(x)

        return inputs, outputs

    def store_intermediate_result(self, epoch, pred_y_all, physics_loss):
        with open(self.storage_path + "loss_epoch.pkl", "wb") as fp:
            pickle.dump(self.logger.loss_over_epoch, fp)

        plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss", scaled=False)
        plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss_scaled", scaled=True)
        plot_solution(self.input_all, self.y_lbl_all, self.x_ic, self.y_lbl_ic, pred_y_all, None, self.storage_path + "/plots/" + "res_"+str(epoch))
        plot_terms_diff(self.input_all, physics_loss, pred_y_all, self.y_lbl_all,
                        self.storage_path + "res_error_all", p_plot_start=0)

        plt.close('all')

    # The actual PINN
    def f_model(self, x_):
        with tf.GradientTape() as tape:
            tape.watch(x_)
            y_pred = self.pred_with_grad(x_)
            y_dx = y_pred[:, 1:]
            y_dx2 = tape.batch_jacobian(y_dx, x_)[:, :, 0]
        del tape

        y = tf.expand_dims(y_pred[:, 0], -1)
        loss = (-y_dx2/self.scaling_factor) -(self.d/self.m)*y_dx - (self.c/self.m)*y 

        return loss

    # The actual PINN
    def f_model_detail(self, x_):
        with tf.GradientTape() as tape:
            tape.watch(x_)
            y_pred = self.pred_with_grad(x_)
            y_dx = y_pred[:, 1:]
            y_dx2 = tape.batch_jacobian(y_dx, x_)[:, :, 0]
        del tape

        y = tf.expand_dims(y_pred[:, 0], -1)
        loss = (-y_dx2/self.scaling_factor) -(self.d/self.m)*y_dx - (self.c/self.m)*y 

        return loss, y
    def sample_collocation_points(self, nr_p_points):
        t_points = np.expand_dims((self.domain[1] - self.domain[0]) * random(nr_p_points) + self.domain[0], 1)
        return tf.cast(t_points, tf.float32)

    def pred_with_grad(self, x_points):
        with tf.GradientTape() as t:
            t.watch(x_points)
            pred = self.model(x_points)
        dx = t.batch_jacobian(pred, x_points)[:, :, 0]
        y_pred_full = tf.concat((pred, dx), axis=1)
        return y_pred_full

    def calc_loss_ic(self):
        diff = self.y_lbl_ic - self.pred_with_grad(self.x_ic)[:, 0:1]
        diff = tf.square(diff)
        ic_loss = tf.reduce_mean(diff)

        return ic_loss

    def calc_physics_loss(self, x_col): # changed to sqauring weights and loss before multiplying
        p_loss = self.f_model(x_col)
        p_loss_mean = tf.reduce_mean(tf.square(p_loss))
        return p_loss_mean

    @tf.function
    def train_step(self, x_col):
        with tf.GradientTape(persistent=True) as tape:
            # data loss / initial condition
            data_loss = self.calc_loss_ic()

            # physics loss
            p_loss = self.calc_physics_loss(x_col)
            combined_weighted_loss = data_loss + (self.physics_scale * p_loss)

            # retrieve gradients
        grads = tape.gradient(combined_weighted_loss, self.model.weights)
        del tape
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        physics_loss, pred_y_all = self.f_model_detail(self.input_all)
        data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_y_all) - tf.squeeze(self.y_lbl_all)))
        physics_loss_mean = tf.reduce_mean(tf.square(physics_loss))
        log_data = [data_loss, physics_loss_mean]

        return combined_weighted_loss, log_data, pred_y_all, physics_loss

    def fit(self):
        self.logger.log_train_start(self)
        for epoch in range(self.tf_epochs):
            #loss_value, log_data, pred_parameters, physics_losses = self.train_step(self.sample_collocation_points(1_000))
            loss_value, log_data, pred_y_all, physics_loss = self.train_step(self.x_physics)

            # log train loss and errors specified in logger error
            self.logger.log_train_epoch(epoch, loss_value, log_data)
            
            if epoch % 2_000 == 0:
                np.save(self.storage_path + "/plots/" + "res_"+str(epoch) + ".npy", pred_y_all)

            if epoch % 25_000 == 0:
                self.store_intermediate_result(epoch, pred_y_all, physics_loss)
        self.logger.log_train_end(self.tf_epochs, log_data)

    def predict(self, x):
        y = self.model(x)
        f_m1, f_m2 = self.f_model(x)
        return [y, f_m1, f_m2]


def get_layer_list(nr_inputs, nr_outputs, nr_hidden_layers, width):
    layers = [nr_inputs]
    for i in range(nr_hidden_layers + 1):
        layers.append(width)
    layers.append(nr_outputs)
    return layers


def main():
    #task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_id = int(sys.argv[1])
    print("task_id: ", task_id)

    # Parameters that change based on task id ############################################################################
    if task_id == 0:
        ic_points_idx = [0]
    elif task_id == 1:
        ic_points_idx = np.arange(0, 500, 10)
    elif task_id == 2:
        ic_points_idx = np.arange(0, 4000, 10)
    
    print("task id ", task_id)

    act_func = "tanh"
    af_str = "tanh"

    scale_m1 = 1
    h_layers = 5

    d_p_string = "vanilla"

    print("ic points: ", ic_points_idx)
    hidden_layers = h_layers

    weight_factor = 2

    lr = tf.Variable(1e-4)
    physics_scale = tf.Variable(0.0)
    ######################################################################################################################
    # Fixed parameters PINN
    training_epochs = 200_001
    width = 32
    layers = get_layer_list(nr_inputs=1, nr_outputs=1, nr_hidden_layers=hidden_layers, width=width)

    # save_loss_freq = how often test error is saved
    logger = Logger(save_loss_freq=1, print_freq=1_000)

    # Data simulation
    start_vec = [1, 0]
    m1 = (30000/2)*scale_m1
    css = 0.5e6 * 2
    dss = 1.5e4 * 2
    exp_len = 4000
    steps = 4000
    y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci, simul_constants = get_simulated_data_single_mass(start_vec, end_time=10, steps=4000, exp_len=exp_len, m1=m1, css=css,dss=dss, debug_data=True)
    simul_results = [y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci]
    ################################f#####

    # Getting the data
    p_start_step = 1
    p_sampling_rate = 1
    data_start = 0
    x_data = tExci[ic_points_idx]
    y_data = y_m1_simul[ic_points_idx]
    y_lbl = y_m1_simul[data_start:]
    t = tExci[data_start:]
    domain = [t[0], t[-1]]

    input_all = tf.cast(tf.concat([t],axis=-1),tf.float32)
    y_lbl_all = tf.convert_to_tensor(y_lbl, dtype=tf.float32)
    input_data = tf.cast(x_data, tf.float32)
    
    input_data_physics = tf.cast(tf.concat([t[p_start_step:exp_len:p_sampling_rate]],axis=-1),tf.float32)
    x_physics = tf.convert_to_tensor(input_data_physics, dtype=tf.float32)
    x_ic = tf.constant(input_data, dtype=tf.float32)
    y_lbl_ic = tf.constant(y_data, dtype=tf.float32)
    
    pinn_data = [x_ic, y_lbl_ic, x_physics, input_all, y_lbl_all]
    
  

    # Setting up folder structure # todo clean up
    result_folder_name = 'res'
    os.makedirs(result_folder_name, exist_ok=True)
    experiment_name = "one_mass_nn_neu_martin_tc_fixeds_h_l_" + str(hidden_layers) + "_w_" + str(
        width) + "_af_" + af_str + "_lr_" + str(lr.numpy()) + "_expl_" + str(exp_len) + "_steps_" + str(
        steps) + "_ps_" + str(physics_scale.numpy()) + "_sf_" + str(
        scale_m1) + "_dp_" + d_p_string + "_id_" + str(task_id)
    print("Config name: ", experiment_name)
    os.makedirs(result_folder_name + "/" + experiment_name, exist_ok=True)
    os.makedirs(result_folder_name + "/" + experiment_name + "/plots", exist_ok=True)
    plots_path = result_folder_name + "/" + experiment_name + "/"

    # plotting solution
    loss_scale_term = np.exp(-(dss)/(2*m1)*t)

    plot_solution(t, y_lbl, x_data, y_data, None,None, result_folder_name+"/" + experiment_name + '/exact_solution')
    plot_solution(t, y_lbl/loss_scale_term, x_data, y_data, None,None, result_folder_name+"/" + experiment_name + '/exact_solution_scaled')

    pinn = PhysicsInformedNN(layers, h_activation_function=act_func, logger=logger, simul_constants=simul_constants,
                             domain=domain, physics_scale=physics_scale, lr=lr, data=pinn_data,
                             simul_results=simul_results, storage_path=plots_path)
    pinn.tf_epochs = training_epochs
    pinn.storage_path = plots_path

    pinn.fit()

    # plot results
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)
    y_pred, f_pred = pinn.predict(input_all)
    plot_solution(t, y_lbl, x_data, y_data, y_pred, plots_path + "res_end")
    plot_terms_diff(t, f_pred, y_pred, y_lbl_all, plots_path + "res_error_all", p_plot_start=p_start_step)

    print("Finished")


if __name__ == "__main__":
    main()




