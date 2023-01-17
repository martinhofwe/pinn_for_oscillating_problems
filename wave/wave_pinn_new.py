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

from data_simulation import get_simulated_data_two_mass
from plot_wave import plot_solution, plot_comparison, plot_loss, plot_terms_diff


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
          data_erro, physics_error = [m.numpy() for m in log_data]
        print(f"{'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error = {data_error:.4e} physics error = {physics_error:.4e}")

  def log_train_end(self, epoch, log_data):
    print("==================")
    data_error, physics_erro = [m.numpy() for m in log_data]
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error = {data_error:.4e} physics error = {physics_error:.4e} ")


class PhysicsInformedNN(object):
  def __init__(self, layers, h_activation_function, logger, velocity, physics_scale, data_scale, lr, storage_path, data):
    
    self.model = tf.keras.Sequential()
    self.setup_layers(layers, h_activation_function)

    self.storage_path = storage_path
    self.dtype = tf.float32
    scaling_factor = 1.0
    self.velocity =  tf.constant(velocity, dtype=self.dtype)
    self.scaling_factor = tf.constant(1.0,dtype=self.dtype)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.8, decay=0.)

    self.logger = logger
    self.physics_scale = physics_scale
    self.data_loss_scl = data_scale
    self.domain = [0, 1] 
    self.storage_path = ""
    self.collocation_cuts = []
    
    self.y_lbl_all, self.t_all, self.x_data, self.y_data, self.locations_x, self.locations_y, self.time_steps, self.p_start_idx, self.wave_fields, self.considered_steps = data 
    

  def setup_layers(self, layers, h_activation_function):
    self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    print("Input Layer + Hidden Layers")
    for count, width in enumerate(layers[1:-1]):
        if h_activation_function == "sine" and count == 0:
          print(width, ": sine af")
          self.model.add(tf.keras.layers.Dense(width, activation=tf.math.sin, kernel_initializer='glorot_normal'))
        else:
          print(width, ": tanh af")
          self.model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal'))

    print("Output layer:")
    print(layers[-1], ": no af")
    self.model.add(tf.keras.layers.Dense(layers[-1], activation=None,kernel_initializer='glorot_normal'))


  def store_intermediate_result(self, epoch, pred_params):
    
    #y_pred = tf.concat((y_m1, y_m2), axis=1)
    with open(self.storage_path + "loss_epoch.pkl", "wb") as fp:
      pickle.dump(self.logger.loss_over_epoch, fp)

    plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss", scaled=False)
    plot_loss(self.logger.loss_over_epoch, self.physics_scale, self.storage_path + "loss_scaled", scaled=True)

    #plot_solution(self.input_all, self.y_lbl_all[:, 1], self.y_lbl_all[:, 0], self.x_ic, self.y_lbl_ic[:,1], self.y_lbl_ic[:, 0], y_pred, self.storage_path +"/plots/" + "res_", epoch)
    plot_comparison(self.t_all, self.wave_fields[self.considered_steps, :, :],pred_params, self.considered_steps,self.storage_path + "plots/" +str(epoch))

    plt.close('all')
  # The actual PINN
  def f_model(self, input_data): # input of form [time, idx pos x, idx pos y]
    with tf.GradientTape() as tape: # todo possible without persistent?
        t = tf.expand_dims(input_data[..., 0], -1)
        x = tf.expand_dims(input_data[..., 1], -1)
        y = tf.expand_dims(input_data[..., 2], -1)
        tape.watch(t)
        tape.watch(x)
        tape.watch(y)
        input_watched_full = tf.concat([t, x, y], axis=-1)

        pred = self.pred_with_grad(input_watched_full)
        u_dt = pred[:, 1]
        u_dx = pred[:, 2]
        u_dy = pred[:, 3]
        #u_dx2 = tape.gradient(u_dx, x)
        #u_dy2 = tape.gradient(u_dy, y)
        #u_dt2 = tape.gradient(u_dt, t)
        u_dt2, u_dx2, u_dy2 = tape.gradient([u_dt, u_dx, u_dy], [t, x, y]) # todo check if works as intended
        del tape
    return (u_dx2+u_dy2)-(u_dt2/(self.velocity**2))

  def f_model_detail(self, input_data):
    with tf.GradientTape() as tape: # todo possible without persistent?
        t = tf.expand_dims(input_data[..., 0], -1)
        x = tf.expand_dims(input_data[..., 1], -1)
        y = tf.expand_dims(input_data[..., 2], -1)
        tape.watch(t)
        tape.watch(x)
        tape.watch(y)
        input_watched_full = tf.concat([t, x, y], axis=-1)

        pred = self.pred_with_grad(input_watched_full)
        u_dt = pred[:, 1]
        u_dx = pred[:, 2]
        u_dy = pred[:, 3]
        #u_dx2 = tape.gradient(u_dx, x)
        #u_dy2 = tape.gradient(u_dy, y)
        #u_dt2 = tape.gradient(u_dt, t)
        u_dt2, u_dx2, u_dy2 = tape.gradient([u_dt, u_dx, u_dy], [t, x, y]) # todo check if works as intended
        del tape
    loss = (u_dx2+u_dy2)-(u_dt2/(self.velocity**2))
    return loss, [pred[:, 0], u_dt, u_dx, u_dy, u_dt2, u_dx2, u_dy2]

  def sample_boundary_points(self, nr_points):
      # create new patch for training
      idx_pos_x = np.expand_dims(np.random.choice(np.arange(0, self.locations_x.shape[0]), size=nr_points), 1)
      idx_pos_y = np.expand_dims(np.random.choice(np.arange(0, self.locations_y.shape[0]), size=nr_points), 1)
      idx_time = np.expand_dims(np.random.choice(np.arange(0, self.x_data.shape[0]), size=nr_points), 1)
      batch_time = self.x_data[idx_time]

      input_data_boundary = tf.concat([batch_time, idx_pos_x, idx_pos_y], axis=-1) # timestep and position

      boundary_lbl = self.y_data[idx_time, idx_pos_x, idx_pos_y]
      
      return input_data_boundary, boundary_lbl
  
  def get_input_lable_for_timestep(self, timestep_idx):
    t_cons = self.t_all[timestep_idx]#self.t_all[self.considered_steps]
    tracked_time_steps = np.expand_dims(np.repeat(np.array(t_cons),self.locations_x.shape[0]* self.locations_y.shape[0]), 1)
    input_data = tf.cast(np.hstack((tracked_time_steps,np.expand_dims(np.repeat(self.locations_x, self.locations_y.shape[0]) ,1), np.expand_dims(np.tile( self.locations_y, self.locations_x.shape[0]),1))), tf.float32)
    lbl_flat = self.y_lbl_all[timestep_idx].flatten() # check if it works as intened
    return input_data, lbl_flat 

  def sample_collocation_points(self, nr_p_points):
      # create physics loss batch
      idx_pos_p_x = np.expand_dims(np.random.choice(np.arange(0, self.locations_x.shape[0]), size=nr_p_points), 1) # todo needs rework? more points?
      idx_pos_p_y = np.expand_dims(np.random.choice(np.arange(0, self.locations_y.shape[0]), size=nr_p_points), 1)
      idx_time_p = np.expand_dims(np.arange(self.p_start_idx, self.time_steps.shape[0]), 1)


      batch_pos_p_x = np.tile(idx_pos_p_x, (self.time_steps.shape[0]-self.p_start_idx, 1))
      batch_pos_p_y = np.tile(idx_pos_p_y, (self.time_steps.shape[0]-self.p_start_idx, 1))
      batch_time_p = np.expand_dims(np.repeat(self.time_steps[idx_time_p], nr_p_points), 1)

      input_data_physics = tf.concat([batch_time_p, batch_pos_p_x, batch_pos_p_y], axis=-1)
      
      return input_data_physics

  def define_p_cuts(self, x_physics, nr_points=10_000):
    nr_cuts = int(np.ceil(x_physics.shape[0] / nr_points))
    for i in range(nr_cuts):
       self.collocation_cuts.append([i*nr_points, np.min([(i+1)*nr_points, x_physics.shape[0]])])


  def pred_with_grad(self, input_watched_full): #todo work with batch gradient?
    with tf.GradientTape() as tape:
        t = tf.expand_dims(input_watched_full[..., 0], -1)
        x = tf.expand_dims(input_watched_full[..., 1], -1)
        y = tf.expand_dims(input_watched_full[..., 2], -1)
        tape.watch(t)
        tape.watch(x)
        tape.watch(y)
        input_watched_full = tf.concat([t, x, y], axis=-1)
        u = self.model(input_watched_full)
    u_dt, u_dx, u_dy = tape.gradient(u, [t, x, y])
    y_pred_full = tf.concat((u, u_dt, u_dx, u_dy), axis=1)
    return y_pred_full

  def calc_loss_ic(self, input_data_boundary, boundary_lbl):
    pred =  self.pred_with_grad(input_data_boundary)
    diff = boundary_lbl - self.pred_with_grad(input_data_boundary)[:, 0]
    ic_loss = tf.reduce_mean(tf.square(diff))
    return ic_loss


  def calc_physics_loss(self, x_col):
    p_loss = self.f_model(x_col)
    p_loss_mean = tf.reduce_mean(tf.square(p_loss))
    return p_loss_mean

  def split_img_into_batches(self, input_data_pred, nr_samples=500):
    nr_splits = input_data_pred.shape[0]//nr_samples
    data_lst = []
    p_lst = []
    for i in range(0,nr_splits):
        physics_loss, pred_parameters = self.f_model_detail(input_data_pred[i*nr_samples:(i+1)*nr_samples])
        data_lst.append(pred_parameters[0])
        p_lst.append(physics_loss)
    return  tf.concat(data_lst, axis=0),  tf.concat(p_lst, axis=0)

  @tf.function
  def train_step(self,input_data_boundary, boundary_lbl, x_col):
    with tf.GradientTape(persistent=True) as tape:

      # data loss / initial condition
      data_loss = self.calc_loss_ic(input_data_boundary, boundary_lbl)

      # physics loss
      p_loss = self.calc_physics_loss(x_col)
      combined_weighted_loss = self.data_loss_scl*data_loss + self.physics_scale * (p_loss)

      # retrieve gradients
    grads = tape.gradient(combined_weighted_loss, self.model.weights)
    del tape
    self.optimizer.apply_gradients(zip(grads,self.model.weights))

    # split since not all images fit into memory at once
    data_loss_metric_lst, physics_loss_metric_lst  = [], []
    pred_lst = []
    for time_step_idx in self.considered_steps:
        input_data, lbl = self.get_input_lable_for_timestep(time_step_idx)
        # process row wise so that it fits into memory
        
        data_pred, physics_pred = self.split_img_into_batches(input_data)
        pred_lst.append(data_pred)
        data_loss_metric_lst.append(tf.square(tf.squeeze(data_pred) - tf.squeeze(lbl)))
        physics_loss_metric_lst.append(tf.square(physics_pred))
        
    log_data = [tf.reduce_mean(tf.concat(data_loss_metric_lst, 0)), tf.reduce_mean(tf.concat(physics_loss_metric_lst, 0))]

    return combined_weighted_loss, log_data, tf.concat(pred_lst, 0)

  def fit(self):
    self.logger.log_train_start(self)
    for epoch in range(self.tf_epochs):
      #loss_value, log_data, pred_parameters, physics_losses = self.train_step(self.x_physics)
      input_data_boundary, boundary_lbl = self.sample_boundary_points(500)
      loss_value, log_data, pred_parameters = self.train_step(input_data_boundary, boundary_lbl, self.sample_collocation_points(500)) # todo as parameter
      # log train loss and errors specified in logger error
      self.logger.log_train_epoch(epoch, loss_value, log_data)

      if epoch % 5_000 == 0:
        self.store_intermediate_result(epoch, pred_parameters)
    self.logger.log_train_end(self.tf_epochs,  log_data)


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
  task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
  #task_id = int(sys.argv[1])
  print("task_id: ", task_id)
  input_bool = False
  if task_id % 2 == 0:
    act_func ="tanh"
    af_str = "tanh"
  else:
    act_func = "sine"
    af_str = "sin"
    
  act_func = "soft_plus"
  af_str = "soft_plus"

  exp_len = 400
  p_norm = "l1"
  p_start_step = 1

  if task_id <= 3:
    width = 128
  else:
    width = 1024

  p_scale_dic = {0: 0.0, 1: 1e-4, 2: 1e-2, 3: 1.0,
                 4: 0.0, 5: 1e-4, 6: 1e-2, 7: 1.0}

  physics_scale = p_scale_dic[task_id]

  hidden_layers = 10 #layer_dic[task_id]
  layers = get_layer_list(nr_inputs=3, nr_outputs=1, nr_hidden_layers=hidden_layers, width=width)

  print("layers: ", layers)
  training_epochs = 1_000_000
  lr = tf.Variable(1e-4)
  ######################################################################################################################
  #load data
  wavefields_all = np.load('wave_data/const_wave.npy')

  velocity = np.load('wave_data/velocity_00000000_const.npy')
  wavefields = wavefields_all[::4]
  wavefields = np.float32(wavefields[:101,:,:])
  time_steps_all = np.linspace(0, 1.024, num=2048)
  time_steps = time_steps_all[::4]

  time_steps = np.float32(time_steps[:101])
  time_step_size = 0.002


  #wavefields_flat = wavefields.reshape(wavefields.shape[0], wavefields.shape[1]*wavefields.shape[2])




  data_start = 0
  scaling_factor = 1
  # Getting the data
  data_sampling = 1
  num_points = 10#200#4000#250
  x_data = time_steps[data_start:num_points+1:data_sampling]

  y_data = wavefields[data_start:num_points+1:data_sampling, :, :]*scaling_factor

  y_lbl = wavefields[data_start:, :, :]
  t = time_steps[data_start:]
  locations_x = np.arange(0, wavefields.shape[1])
  locations_y = np.arange(0, wavefields.shape[2])

  # define which timesteps should be used to calcualte the eror metric printed (all time steps might be to computionally intensive)
   # Setting up folder structure # todo clean up
  logger = Logger(save_loss_freq=1, print_freq=1_000)
  result_folder_name = 'res_wave'
  os.makedirs(result_folder_name, exist_ok=True)
  experiment_name = "wave_vanilla_new"+ "_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_pn_" + p_norm + "_af_" + af_str + "_input_" + str(input_bool) + "_expl_" + str(exp_len) + "_ps_" + str(physics_scale) + "_pstart_" +str(p_start_step) + "_id_" + str(task_id)
  os.makedirs(result_folder_name + "/"+ experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)
  considered_steps = np.array([0, 12, 25, 37, 50, 62, 75, 87, 100])
  y_lbl_error = wavefields[considered_steps, :, :]
  # plotting
  plots_path = result_folder_name+ "/" + experiment_name+ "/"
  plot_solution(t, velocity, wavefields, plots_path + "exact_solution")

  p_start_idx = 1
  data = [y_lbl, t, x_data, y_data, locations_x, locations_y, time_steps, p_start_idx, wavefields, considered_steps]

  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, logger=logger, velocity=velocity[0, 0], physics_scale=physics_scale, data_scale=1.0, lr=lr, storage_path=plots_path, data=data)
  pinn.tf_epochs = training_epochs # todo integrate
  pinn.storage_path = plots_path # todo integrate

  pinn.fit()

  plot_loss(logger.loss_over_meta, pinn.physics_scale, plots_path + "loss", scaled=False)
  plot_loss(logger.loss_over_meta, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)


  y_pred, f_pred = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])
  plot_comparison(t, y_lbl_error,y_pred, considered_steps,plots_path + "end")

  y_pred, f_pred = pinn.predict_multiple_images(considered_steps=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]), locations=[locations_x, locations_y])
  plot_comparison(t, y_lbl[[0, 2, 4, 6, 8, 10, 12, 14, 16]].flatten(),y_pred, considered_steps,plots_path + "learned")

  print("Finished")



if __name__ == "__main__":
    main()




