import random
import sys

#import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
#import scipy.io

from plot_wave import plot_solution, plot_comparison, plot_loss

#np.random.seed(1234)
#tf.random.set_seed(1234)



class Logger(object):
    def __init__(self, frequency=10):
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        # print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.start_time = time.time()
        self.frequency = frequency
        self.loss_over_meta = []
        self.loss_over_epoch = []
        self.epoch_counter = 0
        self.meta_epoch_counter = 0
        self.t = 0
        self.y_lbl_all = 0
        self.plot_path = ""  # result_folder_name + "/" + experiment_name

    def clear_loss_lst(self):
        self.loss_over_meta = []
        self.loss_over_epoch = []

    def __get_elapsed(self):
        # return datetime.utcfromtimestamp((time.time() - self.start_time)).strftime("%M:%S")
        return datetime.fromtimestamp((time.time() - self.start_time)).strftime("%M:%S")  # orig

    def __get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model, show_summary=False):
        print("\nTraining started")
        print("================")
        print(model.summary())

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if self.epoch_counter % self.frequency == 0:
            data_error, physics_error, y_pred, f_pred = self.__get_error_u()
            self.loss_over_epoch.append([data_error, physics_error])
            print(
                f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  data error= {data_error:.4e}  physics error= {physics_error:.4e}" + custom)

            if False:
                fig_res = plt.figure(figsize=(12, 12))
                plt.subplot(4, 1, 1)
                plt.plot(self.t, f_pred[:, 1], label="p error  m2")
                plt.legend()
                plt.subplot(4, 1, 2)
                plt.plot(self.t, y_pred[:, 1] - self.y_lbl_all[:, 1], label="d error  m2")
                plt.legend()
                # plt.show()
                plt.subplot(4, 1, 3)
                plt.plot(self.t, f_pred[:, 0], label="p error  m1")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(self.t, y_pred[:, 0] - self.y_lbl_all[:, 0], label="d error  m1")
                plt.legend()
                fig_res.savefig(self.plot_path + "/plots/" + str(self.epoch_counter + 1) + '.png')

        self.epoch_counter += 1

    def log_train_opt(self, name):
        # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        if self.meta_epoch_counter % 1000 == 0:
            print("==================")
            data_error, physics_error, y_pred, f_pred = self.__get_error_u()
            self.loss_over_meta.append([data_error, physics_error])
            print(
                f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error= {data_error:.4e}  physics error= {physics_error:.4e}  " + custom)

        self.meta_epoch_counter += 1


# Time tracking functions
global_time_list = []
global_last_time = 0


def reset_time():
    global global_time_list, global_last_time
    global_time_list = []
    global_last_time = time.perf_counter()


def record_time():
    global global_last_time, global_time_list
    new_time = time.perf_counter()
    global_time_list.append(new_time - global_last_time)
    global_last_time = time.perf_counter()


def last_time():
    """Returns last interval records in millis."""
    global global_last_time, global_time_list
    if global_time_list:
        return 1000 * global_time_list[-1]
    else:
        return 0


class PhysicsInformedNN(object):
    def __init__(self, layers, h_activation_function, optimizer, logger, velocity, n_inputs=3,
                 physics_scale=1.00, p_norm="l2"):

        self.layers = layers
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        pot_sin_layer = 0  # (len(layers[1:])-1)//2
        for count, width in enumerate(layers[1:]):
            if width > 1:
                if h_activation_function == "sine" and count == pot_sin_layer:
                    print("sine af")
                    self.model.add(tf.keras.layers.Dense(
                        width, activation=tf.math.sin,
                        kernel_initializer='glorot_normal'))
                elif h_activation_function == "sine_all":
                    print("sine af")
                    self.model.add(tf.keras.layers.Dense(
                        width, activation=tf.math.sin,
                        kernel_initializer='glorot_normal'))
                else:
                    print("soft plus af")
                    self.model.add(tf.keras.layers.Dense(
                        width, activation=tf.nn.softplus,
                        kernel_initializer='glorot_normal'))
            else:
                print("no af")
                self.model.add(tf.keras.layers.Dense(
                    width, activation=None,
                    kernel_initializer='glorot_normal'))

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))


        self.n_inputs = n_inputs
        self.optimizer = optimizer
        self.logger = logger
        self.physics_scale = tf.Variable(physics_scale, dtype=tf.float32)
        self.dtype = tf.float32
        self.p_norm = p_norm

        self.velocity = tf.constant(velocity, dtype=tf.float32)

        #  Separating the collocation coordinates
        # self.x = tf.convert_to_tensor(X, dtype=self.dtype)

    # Defining custom loss
    @tf.function
    def __loss(self, x, y_lbl, x_physics, y, x_semi_begin=None, semi_scale=None, p_norm=None, physics_scale=1.0):
        f_pred = self.f_model(x_physics)
        data_loss_semi = 0.0
        data_loss = tf.reduce_mean((y_lbl - y) ** 2)
        if p_norm == "l2":
            physics_loss = tf.reduce_mean(f_pred ** 2)
        if p_norm == "l1":
            physics_loss = tf.reduce_mean(tf.math.abs(f_pred))
        return data_loss + data_loss_semi + physics_scale * physics_loss # todo check if it wors as intened

    def __grad(self, x, y_lbl, x_physics, x_semi_begin, semi_scale):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss_value = self.__loss(x, y_lbl, x_physics, self.model(x), x_semi_begin, semi_scale, p_norm=self.p_norm, physics_scale=self.physics_scale)
        return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

    def __wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    # The actual PINN
    def f_model(self, input_p):
        #########################################
        # https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=azOBDHMoZEkn
        t = tf.expand_dims(input_p[:, 0], -1) # todo check
        x = tf.expand_dims(input_p[:, 1], -1)
        y = tf.expand_dims(input_p[:, 2], -1)
        with tf.GradientTape(persistent=True) as tape:
            # Split t (time) and x (position) to compute partial derivatives
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            input_watched_full = tf.concat([t, x, y], axis=-1)
            # Determine residual
            u = tf.expand_dims(self.model(input_watched_full), -1)

            u_dx = tape.gradient(u, x)
            u_dy = tape.gradient(u, y)
            u_dt = tape.gradient(u, t)
        u_dx2 = tape.gradient(u_dx, x)
        u_dy2 = tape.gradient(u_dy, y)
        u_dt2 = tape.gradient(u_dt, t)
        del tape
        return (u_dx2 + u_dy2) - (u_dt2 / (self.velocity ** 2)) # todo check
    

    def get_params(self, numpy=False):
        return self.k

    def get_weights(self):
        w = []
        for layer in self.model.layers:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        return tf.convert_to_tensor(w, dtype=self.dtype)

    def set_weights(self, w):
        for i, layer in enumerate(self.model.layers):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    def summary(self):
        return self.model.summary()

    # The training function
    def fit(self, x, y_lbl, x_physics, x_semi=None, y_semi=None, semi_scale=None, tf_epochs=5000, show_summary=False):
        if show_summary:
            self.logger.log_train_start(self, show_summary=show_summary)
        # Creating the tensors
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        y_lbl = tf.convert_to_tensor(y_lbl, dtype=self.dtype)
        x_physics = tf.convert_to_tensor(x_physics, dtype=self.dtype)
        x_semi_begin = None
        # self.logger.log_train_opt("Adam")
        for epoch in range(tf_epochs):
            loss_value, grads = self.__grad(x, y_lbl, x_physics, x_semi_begin, semi_scale)
            self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
            self.logger.log_train_epoch(epoch, loss_value)

        #self.logger.log_train_end(tf_epochs)

    def predict(self, x):
        y = self.model(x)
        f = self.f_model(x)  # todo change
        return y, f

    def predict_multiple_images(self, considered_times, locations):
        locs_x, locs_y = locations
        y_lst_big = []
        f_lst_big = []
        for t_step in considered_times: 
            tracked_time_steps = np.expand_dims(np.repeat(np.array(t_step), locs_x.shape[0] * locs_y.shape[0]), 1)
            input_data_pred = tf.cast(np.hstack((tracked_time_steps,
                                                 np.expand_dims(np.repeat(locs_x, locs_y.shape[0]), 1),
                                                 np.expand_dims(np.tile(locs_y, locs_x.shape[0]), 1))), tf.float32)
            # just because cant fit all into gpu:
            y_lst = []
            f_lst = []
            nr_splits = input_data_pred.shape[0] // 300
            for i in range(0, nr_splits):
                y, f = self.predict(input_data_pred[i * 300:(i + 1) * 300])
                y_lst.append(y)
                f_lst.append(f)
            y_lst_big.append(tf.concat(y_lst, axis=0))
            f_lst_big = tf.concat(f_lst, axis=0)
        return tf.squeeze(tf.concat(y_lst_big, axis=0)), tf.squeeze(tf.concat(f_lst_big, axis=0))

def get_grid_spacing(specified_nr_points, array_size):
    actual_nr_points = np.round(np.sqrt(specified_nr_points)).astype(int)
    spacing_points = np.floor((array_size)/actual_nr_points).astype(int)
    print("### Getting grid indices ###")
    print("Specified number of points: ", specified_nr_points)
    print("Actual number of points that fit with an even grid: ", actual_nr_points)
    print("Spacing between the points", spacing_points)
    space_left = (array_size - ((actual_nr_points-1)*spacing_points)).astype(int)
    if  (space_left // actual_nr_points) > 0:
        spacing_points = spacing_points + (space_left // actual_nr_points)
        print("Change spacing of points for better coverage to: ", spacing_points)
    
    return spacing_points

# from matlab script############################################################
def main():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #task_id = int(sys.argv[1])
    print("task_id: ", task_id)
    input_bool = False
    if task_id < 5:
        act_func = "soft_plus"
        af_str = "soft_plus"
    elif 5 <= task_id < 10:
        act_func = "sine"
        af_str = "sine"
    elif 10 <= task_id < 15:
        act_func = "sine_all"
        af_str = "sine_all"

    p_norm = "l1"
    p_start_step = 0

    width = 1024
    physics_scale_new = 50.0
    physics_scale = 0.0
    p_end_start = 1

    hidden_layers = 10  # layer_dic[task_id]
    layers = [3]  # input time and position x, y
    for i in range(hidden_layers + 1):
        layers.append(width)
    layers.append(1)  # result is the value at position x

    print("layers: ", layers)

    result_folder_name = 'res_wave'
    os.makedirs(result_folder_name, exist_ok=True)

    # fixed parameters
    logger = Logger(frequency=5_000)

    p_start = 0
    # parameters ########################################################################################################
    meta_epochs = 1_000_001  # todo without hard coded numbers
    tf_epochs = 1
    
    lr = tf.Variable(1e-5)
    
    #batch_size = 500

    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #############################################################################################
    experiment_name = "wave_va_schedule_" + "_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_pn_" + p_norm + "_af_" + af_str + "_ps_" + str(physics_scale_new) + "_pstart_" + str(p_start_step) + "_id_" + str(task_id)
    
    
    os.makedirs(result_folder_name + "/" + experiment_name, exist_ok=True)
    os.makedirs(result_folder_name + "/" + experiment_name + "/plots", exist_ok=True)

    # load data
    wavefields_all = np.load('wave_data/const_wave.npy')

    velocity = np.load('wave_data/velocity_00000000_const.npy')
    wavefields = wavefields_all[::4] # every fourth because data creation script uses delta_t = 0.0005 -> we want delta_t = 0.002
    print("wavefields, delta_t 0.002: ", wavefields.shape)
    time_steps_all = np.linspace(0, 1.024, num=2048) # macht sinn für 2048 samples a 0.0005 step_size todo not hard coded
    time_steps_init= time_steps_all[::4]
    
    # reduce size of data
    # set t0 t a point where the source term is negligible
    train_data_start = 100
    train_data_end = 201
    wavefields = np.float32(wavefields[train_data_start:train_data_end, :, :])
    time_steps_init = np.float32(time_steps_init[train_data_start:train_data_end])
    print("wavefields reduced, delta_t 0.002: ", wavefields.shape)

    data_start = 0
    # Getting the data
    data_sampling = 1
    num_points = 10
    t = time_steps_init[data_start:]
    x_data = t[data_start:num_points:data_sampling]
    
    print("x_data, ", x_data.shape)
    print("x_data, ", x_data)

    y_data = wavefields[data_start:num_points:data_sampling, :, :]

    y_lbl = wavefields[data_start:, :, :]
    
    locations_x = (np.arange(0, wavefields.shape[1]) * 5) + 2.5 # 300 × 300 grid with a spacing of 5 m in each direction Xmax = 1500 m in both directions + 2,5 to arrive in the middle of the grid
    locations_y = (np.arange(0, wavefields.shape[2]) * 5) + 2.5 
    
    
    print(locations_x)
    print(locations_x.shape)
    # define which timesteps should be used to calcualte the eror metric printed (all time steps might be to computionally intensive)
    considered_steps_idx = np.array([0, 5, 9, 25, 50, 75, 100])
    considered_times = t[considered_steps_idx]
    print("t considered times: ", considered_times)
    y_lbl_error = wavefields[considered_steps_idx, :, :]
    # plotting
    plots_path = result_folder_name + "/" + experiment_name + "/"
    plot_solution(t, velocity, wavefields, plots_path + "exact_solution")
    
    print("velocity")
    print(np.min(velocity), np.max(velocity))

    def error():
        # y, f = pinn.predict(error_input_data)
        y, f = pinn.predict_multiple_images(considered_times=considered_times, locations=[locations_x, locations_y])
        data_error = np.mean((y_lbl_error.flatten() - y) ** 2)  # to do check flattened correctly?
        physics_error = np.mean(f ** 2)
        return data_error, physics_error, y, f

    logger.set_error_fn(error)

    ### set data in logger for plots
    logger.t = t
    logger.y_lbl_all = y_lbl
    logger.plot_path = result_folder_name + "/" + experiment_name

    pinn = PhysicsInformedNN(layers, h_activation_function=act_func, optimizer=tf_optimizer, logger=logger,
                             velocity=velocity[0, 0], n_inputs=layers[0], physics_scale=physics_scale,
                             p_norm=p_norm)  # to do change velocity!!!
    
    ####
    #####################################################################################################################
    # create new patch for training
    d_points_per_ts = 100
    p_points_per_ts = 100
    
    # create boundary input data
    data_grid_spacing = get_grid_spacing(d_points_per_ts, locations_x.shape[0])
    idx_pos_x = np.expand_dims(np.arange(0, wavefields.shape[1])[0::data_grid_spacing], 1)
    idx_pos_y = np.expand_dims(np.arange(0, wavefields.shape[2])[0::data_grid_spacing], 1)

    idx_time = np.expand_dims(np.arange(0, x_data.shape[0]), 1)
    batch_time = x_data[idx_time]

    batch_time_steps = np.expand_dims(np.repeat(np.array(batch_time), idx_pos_x.shape[0] * idx_pos_y.shape[0]), 1)
    batch_pos_x = np.expand_dims(np.repeat(locations_x[idx_pos_x], idx_time.shape[0]*idx_pos_y.shape[0]), 1)
    batch_pos_y = np.expand_dims(np.tile(np.squeeze(locations_y[idx_pos_y]), idx_time.shape[0]*idx_pos_x.shape[0]), 1)


    train_data = tf.cast(np.hstack((batch_time_steps, batch_pos_x ,batch_pos_y)), tf.float32)
    
    batch_time_idx = np.expand_dims(np.repeat(np.array(idx_time), idx_pos_x.shape[0] * idx_pos_y.shape[0]), 1)
    batch_idx_x = np.expand_dims(np.repeat(idx_pos_x, idx_time.shape[0]*idx_pos_y.shape[0]), 1)
    batch_idx_y = np.expand_dims(np.tile(np.squeeze(idx_pos_y), idx_time.shape[0]*idx_pos_x.shape[0]), 1)
    
    train_data_lbl = y_data[batch_time_idx, batch_idx_x, batch_idx_y]

    
    # create physics input data
    grid_spacing_p = get_grid_spacing(d_points_per_ts, locations_x.shape[0])
    idx_pos_x_p = np.expand_dims(np.arange(0, wavefields.shape[1])[0::grid_spacing_p], 1)
    idx_pos_y_p = np.expand_dims(np.arange(0, wavefields.shape[2])[0::grid_spacing_p], 1)
    
    
    idx_time_p = np.expand_dims(np.arange(0, t.shape[0]), 1)
    batch_time_p = t[idx_time_p]
    
    batch_time_steps_p = np.expand_dims(np.repeat(np.array(batch_time_p), idx_pos_x_p.shape[0] * idx_pos_y_p.shape[0]), 1)
    batch_pos_x_p = np.expand_dims(np.repeat(locations_x[idx_pos_x_p], idx_time_p.shape[0]*idx_pos_y_p.shape[0]), 1)
    batch_pos_y_p = np.expand_dims(np.tile(np.squeeze(locations_y[idx_pos_y_p]), idx_time_p.shape[0]*idx_pos_x_p.shape[0]), 1)
    physics_data_input = tf.cast(np.hstack((batch_time_steps_p, batch_pos_x_p ,batch_pos_y_p)), tf.float32)
    #####################################################################################################################
    
    for i in range(0, meta_epochs, 1):
        if i % 1000 == 0:
            print(str(i) + " / " + str(meta_epochs - 1))
        if i % 25_000 == 0:
            y_pred, f_pred = pinn.predict_multiple_images(considered_times=considered_times,
                                                          locations=[locations_x, locations_y])
            plot_comparison(considered_times=considered_times, wavefields=y_lbl_error, wavefields_pred=y_pred, f_path_name = plots_path + "/plots/" + str(i)) # considered_times, wavefields, wavefields_pred,  f_path_name
            
            with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
                pickle.dump(logger.loss_over_meta, fp)
            with open(result_folder_name + "/" + experiment_name + "/loss_epoch.pkl", "wb") as fp:
                pickle.dump(logger.loss_over_epoch, fp)

            plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
            plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)
            
            plt.close('all')
            
            
        if i == 0:
            show_summary = True
        else:
            show_summary = False

        if i == meta_epochs // 2:
            pinn.physics_scale = physics_scale_new
            p_end = p_end_start
        
        if i >= meta_epochs // 2:
          if i % ((meta_epochs // 2) // (t.shape[0]+1)) == 0: # todo clean up
              p_end += 1
              p_end = min(p_end, t.shape[0]-1)
        else:
            p_end = p_end_start
            


        pinn.fit(train_data, train_data_lbl, physics_data_input, tf_epochs=tf_epochs, show_summary=show_summary)
        

    with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
        pickle.dump(logger.loss_over_meta, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_epoch.pkl", "wb") as fp:
        pickle.dump(logger.loss_over_epoch, fp)
    with open(result_folder_name + "/" + experiment_name + "/p_end.pkl", "wb") as fp:
        pickle.dump([p_start, p_end], fp)

    #model_save  = pinn.model
    #model_save.save(result_folder_name + "/" + experiment_name + "/model_end", include_optimizer=True)

    #pinn.model.save_weights(result_folder_name + "/" + experiment_name + "/_weights")

    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)

    y_pred, f_pred = pinn.predict_multiple_images(considered_times=considered_times,
                                                  locations=[locations_x, locations_y])
    
    plot_comparison(considered_times=considered_times, wavefields=y_lbl_error, wavefields_pred=y_pred, f_path_name = plots_path + "end")
    
    
    
    y_pred, f_pred = pinn.predict_multiple_images(considered_times=x_data, locations=[locations_x, locations_y])
    plot_comparison(considered_times=x_data, wavefields=y_lbl[data_start:num_points:data_sampling].flatten(), wavefields_pred=y_pred, f_path_name = plots_path + "input_data_comp")

    print("Finished")


if __name__ == "__main__":
    main()



