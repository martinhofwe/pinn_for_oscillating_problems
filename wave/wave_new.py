import random
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

from data_simulation import get_simulated_data_two_mass
from plot_wave_short import plot_solution, plot_comparison, plot_loss, plot_terms_diff


np.random.seed(12345)
tf.random.set_seed(12345)


class Logger(object):
  def __init__(self, save_loss_freq, print_freq=1_000):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    self.start_time = time.time()
    self.save_loss_freq = save_loss_freq
    self.print_freq = print_freq
    self.loss_over_epoch = []
    self.overall_epoch = 0

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
    if self.overall_epoch % self.save_loss_freq == 0:
      data_error, physics_error = [m.numpy() for m in log_data]
      self.loss_over_epoch.append([data_error, physics_error])
    if self.overall_epoch % self.print_freq == 0:
        if not self.overall_epoch % self.save_loss_freq == 0:
          data_error, physics_error = [m.numpy() for m in log_data]
        print(f"{'tf_epoch'} = {self.overall_epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error= {data_error:.4e}  physics error = {physics_error:.4e} ")
    self.overall_epoch +=1

  def log_train_end(self, epoch, log_data):
    print("==================")
    data_error, physics_error = [m.numpy() for m in log_data]
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error = {data_error:.4e}  physics error = {physics_error:.4e}  ")


class PhysicsInformedNN(object):
  def __init__(self, layers, h_activation_function, lr, logger, velocity, physics_scale=1.00, p_norm="l1"):
    
    self.model = tf.keras.Sequential()
    self.setup_layers(layers, h_activation_function)

    self.storage_path = ""
    self.dtype = tf.float32

    
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    self.logger = logger
    self.velocity = velocity
    self.physics_scale = physics_scale

  def setup_layers(self, layers, h_activation_function):
    self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    print("Input Layer + Hidden Layers")
    for count, width in enumerate(layers[1:-1]):
        if h_activation_function == "sine" and count == 0:
          print(width, ": sine af")
          self.model.add(tf.keras.layers.Dense(width, activation=tf.math.sin, kernel_initializer='glorot_normal'))
        else:
          print(width, ": soft plus af")
          self.model.add(tf.keras.layers.Dense(
                        width, activation=tf.nn.softplus,
                        kernel_initializer='glorot_normal'))

    print("Output layer:")
    print(layers[-1], ": no af")
    self.model.add(tf.keras.layers.Dense(layers[-1], activation=None,kernel_initializer='glorot_normal'))


  def store_intermediate_result(self, epoch, pred_params, physics_losses):
    y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = pred_params
    y_pred = tf.concat((y_m1, y_m2), axis=1)
    with open(self.storage_path + "loss_epoch.pkl", "wb") as fp:
      pickle.dump(self.logger.loss_over_epoch, fp)

    plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss", scaled=False)
    plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss_scaled", scaled=True)

    plot_solution(self.input_all, self.y_lbl_all[:, 1], self.y_lbl_all[:, 0], self.x_ic, self.y_lbl_ic[:,1], self.y_lbl_ic[:, 0], y_pred, self.storage_path +"/plots/" + "res_", epoch)
    plot_terms_diff(self.input_all, physics_losses[0], physics_losses[1], y_pred, self.y_lbl_all, self.storage_path + "res_error_all", p_plot_start=1)
    plot_terms_detail(self.input_all, y_m2, self.y_m2_simul, y_m1, self.y_m1_simul, y_m2_dx, self.y_m2_dx_simul, y_m1_dx, self.y_m1_dx_simul, y_m2_dx2, self.y_m2_dx2_simul, y_m1_dx2, self.y_m1_dx2_simul, f_path_name=self.storage_path + "res_error_detail_after")

    plt.close('all')
  # The actual PINN
  def f_model(self, x_):
      with tf.GradientTape() as tape:
        tape.watch(x_)
        y = self.pred_with_grad(x_)
        y_dx = y[:, 1:]
        y_dx2_all = tape.batch_jacobian(y_dx, x_)
      del tape

      u_dt2 = tf.expand_dims(y_dx2_all[:, 0, 0], -1)
      u_dx2 = tf.expand_dims(y_dx2_all[:, 1, 1], -1)
      u_dy2 = tf.expand_dims(y_dx2_all[:, 2, 2], -1)
      return (u_dx2 + u_dy2) - (u_dt2 / (self.velocity ** 2))


  def sample_collocation_points(self, nr_p_points=500):
        # 500 random samples aus t, x, y
        t_points = np.expand_dims((self.time_steps[self.p_end] - self.time_steps[self.p_start]) * np.random.rand(nr_p_points).astype('float32') + self.time_steps[self.p_start], 1)
        idx_pos_p_x = np.expand_dims(random.choices(np.arange(0, self.locations_x.shape[0]), k=nr_p_points),1)  # todo needs rework? more points?
        idx_pos_p_y = np.expand_dims(random.choices(np.arange(0, self.locations_y.shape[0]), k=nr_p_points), 1)
        return tf.convert_to_tensor(tf.concat([t_points, idx_pos_p_x, idx_pos_p_y], axis=-1), dtype=self.dtype)
  
  def sample_data_points(self, nr_points=500):
        # create new patch for training
        idx_pos_x = np.expand_dims(random.choices(np.arange(0, self.locations_x.shape[0]), k=nr_points), 1) #x = row
        idx_pos_y = np.expand_dims(random.choices(np.arange(0, self.locations_y.shape[0]), k=nr_points), 1) # y = col
        idx_time = np.expand_dims(random.choices(np.arange(0, self.x_data.shape[0]), k=nr_points), 1)

        batch_time = self.x_data[idx_time]

        input_data = tf.concat([batch_time, idx_pos_x, idx_pos_y], axis=-1)  # timestep and position

        data_lbl = self.y_data[idx_time, idx_pos_x, idx_pos_y]
        return tf.convert_to_tensor(input_data, dtype=self.dtype), tf.convert_to_tensor(data_lbl, dtype=self.dtype)
  def define_p_cuts(self, x_physics, nr_points=10_000):
    nr_cuts = int(np.ceil(x_physics.shape[0] / nr_points))
    for i in range(nr_cuts):
       self.collocation_cuts.append([i*nr_points, np.min([(i+1)*nr_points, x_physics.shape[0]])])


  def pred_with_grad(self, x_points):
    with tf.GradientTape() as t:
      t.watch(x_points)
      pred = self.model(x_points)
    dx = t.batch_jacobian(pred, x_points)
    y_pred_full = tf.concat((pred, dx[:,:, 0], dx[:,:, 1], dx[:,:, 2]), axis=1)
    return y_pred_full

  def calc_loss_ic(self,input_batch, batch_lbl):
    diff = batch_lbl - self.pred_with_grad(input_batch)[0, :]
    ic_loss = tf.reduce_mean(tf.square(diff))
    return ic_loss


  def calc_physics_loss(self, x_col):
    p_loss = self.f_model(x_col)
    p_loss_mean = tf.reduce_mean(tf.square(p_loss))
    return p_loss_mean

  @tf.function
  def train_step(self, input_batch, batch_lbl, coll_points):
    with tf.GradientTape(persistent=True) as tape:
      # data loss / initial condition
      data_loss = self.calc_loss_ic(input_batch, batch_lbl)

      # physics loss
      p_loss = self.calc_physics_loss(coll_points)
      combined_weighted_loss = data_loss + self.physics_scale * p_loss

      # retrieve gradients
    grads = tape.gradient(combined_weighted_loss, self.model.weights)
    del tape
    self.optimizer.apply_gradients(zip(grads,self.model.weights))


    #physics_losses, pred_parameters = self.f_model_detail(self.input_all)
    #m1_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_parameters[0]) - tf.squeeze(self.y_lbl_all[:, 0])))
    #m2_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_parameters[1]) - tf.squeeze(self.y_lbl_all[:, 1])))
    #m1_loss, m2_loss = physics_losses
    #f_pred_m1 = tf.reduce_mean(tf.square(m1_loss))
    #f_pred_m2 = tf.reduce_mean(tf.square(m2_loss))
    #log_data = [m1_data_loss, m2_data_loss, f_pred_m1, f_pred_m2]
    
    log_data = [data_loss, p_loss]
    return combined_weighted_loss, log_data

  def fit(self, tf_epochs, show_summary):
    if show_summary:
        self.logger.log_train_start(self)
    for epoch in range(tf_epochs):
      input_batch, batch_lbl = self.sample_data_points(500)   
      loss_value, log_data = self.train_step(input_batch, batch_lbl, self.sample_collocation_points(500)) # todo as parameter
      # log train loss and errors specified in logger error
      self.logger.log_train_epoch(epoch, loss_value, log_data)

      #if epoch % 25_000 == 0: todo!
      #  self.store_intermediate_result(epoch, pred_parameters, physics_losses)
    #self.logger.log_train_end(self.tf_epochs,  log_data)

  @tf.function
  def predict(self, x):
    y = self.model(x)
    #p_error = self.f_model(x)
    return [y, y]

  def predict_multiple_images(self, considered_steps, locs_x, locs_y, t_s):
        y_lst_big = []
        f_lst_big = []
        for step_c in tf.range(len(considered_steps)):
            step = t_s[step_c]
            
            tracked_time_steps = tf.expand_dims(tf.repeat(step, locs_x.shape[0] * locs_y.shape[0]), -1)
            l_x = tf.expand_dims(tf.repeat(locs_x, locs_y.shape[0]), 1)
            l_y = tf.expand_dims(tf.tile(locs_y, tf.constant([locs_x.shape[0]], tf.int32)), 1)
            input_data_pred = tf.concat((tracked_time_steps, l_x, l_y), axis=1)

            y, f = self.predict(input_data_pred)
            y_lst_big.append(y)
            f_lst_big.append(f)
        return tf.squeeze(tf.concat(y_lst_big, axis=0)), tf.squeeze(tf.concat(f_lst_big, axis=0))

def main():
    #task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_id = int(sys.argv[1])
    print("task_id: ", task_id)
    input_bool = False

    act_func = "soft_plus"
    af_str = "soft_plus"

    p_norm = "l1"
    width = 1024
    
    physics_scale_new = 50.0
    physics_scale = 0.0
    p_end = 1

    hidden_layers = 10  # layer_dic[task_id]
    layers = [3]  # input time and position x, y
    for i in range(hidden_layers + 1):
        layers.append(width)
    layers.append(1)  # result is the value at position x

    print("layers: ", layers)

    result_folder_name = 'res_wave'
    os.makedirs(result_folder_name, exist_ok=True)

    # fixed parameters
    logger = Logger(save_loss_freq=5_000, print_freq=10)

    p_start = 0
    # parameters ########################################################################################################
    meta_epochs = 1_000_001  # todo without hard coded numbers
    tf_epochs = 1
    
    lr = tf.Variable(1e-5)
    
    batch_size = 500

    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #############################################################################################
    experiment_name = "new_wave_vanilla_schedule_" + "_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_pn_" + p_norm + "_af_" + af_str + "_ps_" + str(physics_scale_new) + "_pstart_" + str(p_start) + "_id_" + str(task_id)
    
    
    os.makedirs(result_folder_name + "/" + experiment_name, exist_ok=True)
    os.makedirs(result_folder_name + "/" + experiment_name + "/plots", exist_ok=True)

    # load data
    wavefields_all = np.load('wave_data/const_wave.npy')

    velocity = np.load('wave_data/velocity_00000000_const.npy')
    wavefields = wavefields_all[::4] # every fourth because data creation script uses delta_t = 0.0005 -> we want delta_t = 0.002
    print("wavefields, delta_t 0.002: ", wavefields.shape)
    time_steps_all = np.linspace(0, 1.024, num=2048) # macht sinn für 2048 samples a 0.0005 step_size todo not hard coded
    time_steps = time_steps_all[::4]
    
    # reduce size of data
    # set t0 t a point where the source term is negligible
    train_data_start = 100
    train_data_end = 201
    wavefields = np.float32(wavefields[train_data_start:train_data_end, :, :])
    time_steps = np.float32(time_steps[train_data_start:train_data_end])
    print("wavefields reduced, delta_t 0.002: ", wavefields.shape)

    data_start = 0
    # Getting the data
    data_sampling = 1
    num_points = 10
    x_data = time_steps[data_start:num_points:data_sampling]
    
    print("x_data, ", x_data.shape)
    print("x_data, ", x_data)

    y_data = wavefields[data_start:num_points:data_sampling, :, :]

    y_lbl = wavefields[data_start:, :, :]
    t = time_steps[data_start:]
    locations_x = np.arange(0, wavefields.shape[1])
    locations_y = np.arange(0, wavefields.shape[2])

    # define which timesteps should be used to calcualte the eror metric printed (all time steps might be to computionally intensive)
    considered_steps = np.array([0,1,2,3,4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100])
    considered_steps = np.array([0, 5])
    print("t considered times: ", t[considered_steps])
    y_lbl_error = wavefields[considered_steps, :, :]
    # plotting
    plots_path = result_folder_name + "/" + experiment_name + "/"
    plot_solution(t, velocity, wavefields, plots_path + "exact_solution")
    
    print("velocity")
    print(np.min(velocity), np.max(velocity))
    
    pinn = PhysicsInformedNN(layers, h_activation_function=act_func, lr=lr, logger=logger, velocity=velocity[0, 0], physics_scale=physics_scale, p_norm=p_norm)
    pinn.storage_path = plots_path # todo integrate
    ### wave
    pinn.time_steps = tf.convert_to_tensor(time_steps,tf.float32)
    pinn.p_end = p_end
    pinn.p_start = p_start
    pinn.locations_x = locations_x
    pinn.locations_y = locations_y
    pinn.x_data = x_data
    pinn.y_data = y_data
    ###
    for i in range(0, 1, 1):
        if i % 1000 == 0:
            print(str(i) + " / " + str(meta_epochs - 1))
        if (i+1) % 5_000 == 0:
            y_pred, f_pred = pinn.predict_multiple_images(considered_steps=tf.convert_to_tensor(considered_steps, tf.int32),
                                                            locs_x=tf.convert_to_tensor(locations_x, tf.float32), locs_y=tf.convert_to_tensor(locations_x,tf.float32), t_s=pinn.time_steps)
            plot_comparison(t, y_lbl_error, y_pred, considered_steps, plots_path + "/plots/" + str(i))
            plt.close('all')
        if i == 0:
            show_summary = True
        else:
            show_summary = False

        if i == meta_epochs // 2:
            pinn.physics_scale = physics_scale_new
        
        if i > meta_epochs // 2:
            if i % ((meta_epochs // 2) // time_steps.shape[0]) == 0: # todo clean up
                pinn.p_end += 1
                pinn.p_end = min(p_end, time_steps.shape[0]-1)
        else:
            pinn.p_end = p_end
            
        pinn.fit(tf_epochs=meta_epochs, show_summary=show_summary)
        

    with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
        pickle.dump(logger.loss_over_meta, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_epoch.pkl", "wb") as fp:
        pickle.dump(logger.loss_over_epoch, fp)
    with open(result_folder_name + "/" + experiment_name + "/p_end.pkl", "wb") as fp:
        pickle.dump([p_start, p_end], fp)

    model_save  = pinn.model
    model_save.save(result_folder_name + "/" + experiment_name + "/model_end", include_optimizer=True)

    pinn.model.save_weights(result_folder_name + "/" + experiment_name + "/_weights")

    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)

    y_pred, f_pred = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])
    plot_comparison(t, y_lbl_error, y_pred, considered_steps, plots_path + "end")

    y_pred, f_pred = pinn.predict_multiple_images(considered_steps=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
                                                  locations=[locations_x, locations_y])
    plot_comparison(t, y_lbl[[0, 2, 4, 6, 8, 10, 12, 14, 16]].flatten(), y_pred, [0, 2, 4, 6, 8, 10, 12, 14, 16],
                    plots_path + "learned")

    print("Finished")



if __name__ == "__main__":
    main()




