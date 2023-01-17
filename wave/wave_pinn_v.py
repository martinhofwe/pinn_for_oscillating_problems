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
from plot_wave import plot_solution, plot_comparison, plot_loss, plot_terms_diff

np.random.seed(1234)
tf.random.set_seed(1234)

y_lbl_m1_g = tf.Variable(0.0)
y_lbl_m2_g = tf.Variable(0.0)
y_lbl_m1_dx_g = tf.Variable(0.0)
y_lbl_m2_dx_g = tf.Variable(0.0)
y_lbl_m1_dx2_g = tf.Variable(0.0)
y_lbl_m2_dx2_g = tf.Variable(0.0)
u_lbl_g= tf.Variable(0.0)
up_lbl_g = tf.Variable(0.0)

class Logger(object):
  def __init__(self, frequency=10):
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    #print("GPU-accerelated: {}".format(tf.test.is_gpu_available()))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    self.start_time = time.time()
    self.frequency = frequency
    self.loss_over_meta  = []
    self.loss_over_epoch = []
    self.epoch_counter = 0
    self.meta_epoch_counter = 0
    self.t = 0
    self.y_lbl_all = 0
    self.plot_path = "" #result_folder_name + "/" + experiment_name
  def clear_loss_lst(self):
    self.loss_over_meta = []
    self.loss_over_epoch = []

  def __get_elapsed(self):
    #return datetime.utcfromtimestamp((time.time() - self.start_time)).strftime("%M:%S")
    return datetime.fromtimestamp((time.time() - self.start_time)).strftime("%M:%S")# orig

  def __get_error_u(self):
    return self.error_fn()

  def set_error_fn(self, error_fn):
    self.error_fn = error_fn
  
  def log_train_start(self, model,show_summary=False):
    print("\nTraining started")
    print("================")
    self.model = model
    
    print(self.model.summary())

  def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
    if self.epoch_counter % self.frequency == 0:
      data_error, physics_error,y_pred, f_pred = self.__get_error_u()
      self.loss_over_epoch.append([data_error, physics_error])
      print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  data error= {data_error:.4e}  physics error= {physics_error:.4e}" + custom)

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
      data_error, physics_error,y_pred, f_pred = self.__get_error_u()
      self.loss_over_meta.append([data_error, physics_error])
      print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error= {data_error:.4e}  physics error= {physics_error:.4e}  " + custom)

    self.meta_epoch_counter +=1

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
  def __init__(self, layers, h_activation_function, optimizer, logger,velocity, n_inputs=3,scaling_factor=1e12,physics_scale=1.00, p_norm="l2"):

    self.layers = layers
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    pot_sin_layer = 0#(len(layers[1:])-1)//2
    for count, width in enumerate(layers[1:]):
      if width > 1:
        if h_activation_function == "sine" and count == pot_sin_layer:
          print("sine af")
          self.model.add(tf.keras.layers.Dense(
            width, activation=tf.math.sin,
            kernel_initializer='glorot_normal'))
        elif h_activation_function == "sine" and False:
          assert False
          print("tanh af")
          self.model.add(tf.keras.layers.Dense(
            width, activation=tf.nn.tanh,
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


    self.scaling_factor = tf.constant(scaling_factor,dtype=tf.float32)
    self.scaling_factor64 = tf.constant(scaling_factor,dtype=tf.float32)
    self.n_inputs = n_inputs
    self.optimizer = optimizer
    self.logger = logger
    self.physics_scale=tf.constant(physics_scale,dtype=tf.float32)
    self.dtype = tf.float32
    self.p_norm = p_norm

    self.velocity =  tf.constant(velocity, dtype=tf.float32)

    # Separating the collocation coordinates
    #self.x = tf.convert_to_tensor(X, dtype=self.dtype)
    
  # Defining custom loss
  @tf.function
  def __loss(self,x, y_lbl,x_physics, y, x_semi_begin=None, semi_scale=None, p_norm=None):

    f_pred = self.f_model(x_physics)
    if x_semi_begin is not None:
      assert False # todo fix for two mass
      data_loss =  tf.reduce_mean((y_lbl[:x_semi_begin] - y[:x_semi_begin])**2)
      data_loss_semi =  tf.reduce_mean(semi_scale*((y_lbl[x_semi_begin:] - y[x_semi_begin:])**2))
    else:
      data_loss_semi = 0.0
      # np.mean((y_lbl_m1 - y[:, 0])**2), np.mean((y_lbl_m2 - y[:, 1])**2),
      data_loss = tf.reduce_mean((y_lbl - y)**2)
    if p_norm == "l2":
      physics_loss = tf.reduce_mean(f_pred**2)
    if p_norm == "l1":
      physics_loss = tf.reduce_mean(tf.math.abs(f_pred))

    return data_loss+data_loss_semi+self.physics_scale*physics_loss
  
  def __grad(self, x, y_lbl, x_physics, x_semi_begin, semi_scale):
    with tf.GradientTape() as tape:
      tape.watch(x)
      loss_value = self.__loss(x,y_lbl, x_physics, self.model(x), x_semi_begin, semi_scale, p_norm=self.p_norm)
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.model.trainable_variables
    return var

  # The actual PINN
  def f_model(self,input):
    #########################################
    # https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=azOBDHMoZEkn
    with tf.GradientTape(persistent=True) as tape:
      # Split t (time) and x (position) to compute partial derivatives
      t = tf.expand_dims(input[..., 0], -1)
      x = tf.expand_dims(input[..., 1], -1)
      y = tf.expand_dims(input[..., 2], -1)

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
    return (u_dx2+u_dy2)-(u_dt2/(self.velocity**2))


  # The actual PINN
  def f_model_detail(self, x):
    assert False
    #########################################
    # https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=azOBDHMoZEkn
    with tf.GradientTape(persistent=True) as tape:
      # Split t and x to compute partial derivatives
      x_ = tf.expand_dims(x[..., 0], -1)
      u = tf.expand_dims(x[..., 1], -1)
      u_dx = tf.expand_dims(x[..., 2], -1)
      # ym1_lbl = tf.expand_dims(x[...,3],-1)

      # Variables t and x are watched during tape
      # to compute derivatives u_t and u_x
      tape.watch(x_)
      x_watched_full = tf.concat([x_, u, u_dx], axis=-1)

      # Determine residual
      y = self.model(x_watched_full)
      y_m1 = tf.expand_dims(y[:, 0], -1)
      y_m2 = tf.expand_dims(y[:, 1], -1)
      y_m1_dx = tape.gradient(y_m1, x_)
      y_m2_dx = tape.gradient(y_m2, x_)
    y_m1_dx2 = tape.gradient(y_m1_dx, x_)
    y_m2_dx2 = tape.gradient(y_m2_dx, x_)
    del tape

    #####################################

    y_m1_dx = tf.expand_dims(y_m1_dx[..., 0], -1)
    y_m2_dx = tf.expand_dims(y_m2_dx[..., 0], -1)

    y_m1_dx2 = tf.expand_dims(y_m1_dx2[..., 0], -1)
    y_m2_dx2 = tf.expand_dims(y_m2_dx2[..., 0], -1)

    m2_loss = ((-self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2) - (y_m2_dx2/self.scaling_factor)
    # p loss for m1
    #m1_loss = (-y_m1_dx2 / self.scaling_factor) + (self.c1 * (u - y_m1) + self.d1 * (u_dx - y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (y_m2_dx - y_m1_dx)) / self.m1 old and wrong
    m1_loss = ((self.c1 * (u - y_m1) + self.d1 * (u_dx - y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (y_m2_dx - y_m1_dx)) / self.m1) - (y_m1_dx2/self.scaling_factor)

    return tf.concat([m1_loss, m2_loss], axis=1), y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2

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
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def summary(self):
    return self.model.summary()

  # The training function
  def fit(self, x, y_lbl,x_physics,x_semi=None, y_semi=None, semi_scale=None, tf_epochs=5000, show_summary=False):
    if show_summary:
        self.logger.log_train_start(self,show_summary=show_summary)
    # Creating the tensors
    x = tf.convert_to_tensor(x, dtype=self.dtype)
    y_lbl = tf.convert_to_tensor(y_lbl, dtype=self.dtype)
    x_physics = tf.convert_to_tensor(x_physics, dtype=self.dtype)

    # todo remove
    x_semi_begin = None
    if x_semi is not None:
      x_semi = tf.convert_to_tensor(x_semi, dtype=self.dtype)
      y_lbl_semi = tf.convert_to_tensor(y_semi, dtype=self.dtype)
      semi_scale =  tf.convert_to_tensor(semi_scale, dtype=self.dtype)
      x_semi_begin = x.shape[0]
      x = tf.concat([x, x_semi], 0)
      y_lbl = tf.concat([y_lbl, y_lbl_semi], 0)

    #self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      loss_value, grads = self.__grad(x, y_lbl, x_physics, x_semi_begin, semi_scale)
      self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      #self.logger.log_train_epoch(epoch, loss_value)

    self.logger.log_train_end(tf_epochs)

  def predict(self, x):
    y = self.model(x)
    f = self.f_model(x) # todo change
    return y, f

  def predict_multiple_images(self, considered_steps, locations):
    locs_x, locs_y = locations
    '''
    if self.layers[1] == 128:
      y_lst_big = []
      f_lst_big = []
      for step in considered_steps:
        tracked_time_steps = np.expand_dims(np.repeat(np.array(step), locations.shape[0]), 1)
        input_data_pred = tf.cast(np.hstack((tracked_time_steps,  np.expand_dims(locations,1))), tf.float32)
        # just because cant fit all into gpu:
        y_lst = []
        f_lst = []
        for i in range(1,3):
          y, f = self.predict(input_data_pred[(i-1)*45_000:45_000*i])
          y_lst.append(y)
          f_lst.append(f)
        y_lst_big.append(tf.concat(y_lst, axis=0))
        f_lst_big = tf.concat(f_lst, axis=0)
    else:
    '''
    y_lst_big = []
    f_lst_big = []
    for step in considered_steps:
      tracked_time_steps = np.expand_dims(np.repeat(np.array(step),locs_x.shape[0]*locs_y.shape[0]), 1)
      input_data_pred = tf.cast(np.hstack((tracked_time_steps,np.expand_dims(np.repeat(locs_x,locs_y.shape[0]) ,1), np.expand_dims(np.tile(locs_y, locs_x.shape[0]),1))), tf.float32)
      # just because cant fit all into gpu:
      y_lst = []
      f_lst = []
      # nr_splits = input_data_pred.shape[1]//500
      # for i in range(0,nr_splits):
      #   print(input_data_pred[i*500:(i+1)*500].shape)
      #   y, f = self.predict(input_data_pred[i*500:(i+1)*500])
      #   y_lst.append(y)
      #   f_lst.append(f)
      # y_lst_big.append(tf.concat(y_lst, axis=0))
      # f_lst_big.append(tf.concat(f_lst, axis=0))


      nr_splits = input_data_pred.shape[0]//500
      for i in range(0,nr_splits):
        y, f = self.predict(input_data_pred[i*500:(i+1)*500])
        y_lst.append(y)
        f_lst.append(f)
      y_lst_big.append(tf.concat(y_lst, axis=0))
      f_lst_big = tf.concat(f_lst, axis=0)
    return  tf.squeeze(tf.concat(y_lst_big, axis=0)),  tf.squeeze(tf.concat(f_lst_big, axis=0))

    return tf.squeeze(tf.concat(y_lst_big, axis=0)),  tf.squeeze(tf.concat(f_lst_big, axis=0))







# from matlab script############################################################
def main():
  tf.keras.backend.set_floatx('float32')
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

  p_scale_dic = {0: 1e-5, 1: 1e-4, 2: 1e-2, 3: 1.0,
                 4: 1e-5, 5: 1e-4, 6: 1e-2, 7: 1.0}

  physics_scale_new = 0.0
  physics_scale = p_scale_dic[task_id]


  hidden_layers = 10 #layer_dic[task_id]
  layers = [3] # input time and position x, y
  for i in range(hidden_layers + 1):
    layers.append(width)
  layers.append(1) # result is the value at position x

  print("layers: ", layers)

  result_folder_name = 'res_wave'
  #if not os.path.exists(result_folder_name):
  os.makedirs(result_folder_name, exist_ok=True)

  # fixed parameters
  logger = Logger(frequency=1000)

  p_start = 0
  p_end = 250
  # parameters ########################################################################################################
  ppi = 0
  max_iter_overall = 1#
  if width == 1024:
    meta_epochs = 400_000  # todo without hard coded numbers
  else:
    meta_epochs = 800_000
  lr = tf.Variable(1e-4)#tf.Variable(1e-4)
  tf_epochs_warm_up = 2000
  tf_epochs_train = int(max_iter_overall / meta_epochs)
  tf_epochs = max_iter_overall
  batch_size = 500

  mode_frame = False
  tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.8, decay=0.)

  experiment_name = "wave_vanilla_ppi_" + str(ppi) + "_frame_" + str(mode_frame) + "_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_pn_" + p_norm + "_af_" + af_str + "_input_" + str(input_bool) + "_expl_" + str(exp_len) + "_ps_" + str(physics_scale_new) + "_pstart_" +str(p_start_step) + "_id_" + str(task_id)
  os.makedirs(result_folder_name + "/"+ experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)

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
  considered_steps = np.array([0, 12, 25, 37, 50, 62, 75, 87, 100])
  y_lbl_error = wavefields[considered_steps, :, :]
  # plotting
  plots_path = result_folder_name+ "/" + experiment_name+ "/"
  plot_solution(t, velocity, wavefields, plots_path + "exact_solution")



  def error():
    #y, f = pinn.predict(error_input_data)
    y, f = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])
    data_error = np.mean((y_lbl_error.flatten() - y) ** 2) #to do check flattened correctly?
    physics_error = np.mean(f ** 2)
    return data_error, physics_error, y, f
  logger.set_error_fn(error)


  print("Frame mode: ", str(mode_frame))
  print("ppi", ppi)

  ### set data in logger for plots
  logger.t = t
  logger.y_lbl_all = y_lbl
  logger.plot_path = result_folder_name + "/" + experiment_name


  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, optimizer=tf_optimizer, logger=logger,velocity=velocity[0, 0], n_inputs=layers[0],
                           scaling_factor=scaling_factor,physics_scale=physics_scale, p_norm=p_norm) # to do change velocity!!!

  try_next_semi_points = False
  input_data_semi, y_data_semi, y_data_semi_pseudo, pseudo_physics_norm, input_data_semi_new_pot, semi_candidates = None, None, None, None, None, None

  y_pred, f_pred = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])



  plot_comparison(t, y_lbl_error,y_pred, considered_steps,plots_path + "beginning")

  for i in range(meta_epochs):
      if i % 1000 == 0:
        print(str(i) + " / " +str(meta_epochs-1))
      if i % 100 == 0:
          y_pred, f_pred = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])
          plot_comparison(t, y_lbl_error,y_pred, considered_steps,plots_path + "/plots/" +str(i))
      if i==0:
          show_summary = True
      else:
          show_summary = False

      if i == meta_epochs//2:
        print("setting p scale " + str(physics_scale_new))
        '''
        f_pred, y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = pinn.f_model_detail(input_all)
        plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_m2_dx_simul, y_m1_dx, y_m1_dx_simul, y_m2_dx2,
                          y_m2_dx2_simul, y_m1_dx2, y_m1_dx2_simul, f_path_name= plots_path + "res_error_detail_before")
        '''

        pinn.physics_scale = physics_scale_new

      # create new patch for training
      idx_pos_x = np.expand_dims(random.choices(np.arange(0, locations_x.shape[0]), k=batch_size), 1)
      idx_pos_y = np.expand_dims(random.choices(np.arange(0, locations_y.shape[0]), k=batch_size), 1)
      idx_time = np.expand_dims(random.choices(np.arange(0, x_data.shape[0]), k=batch_size), 1)

      batch_time = x_data[idx_time]



      input_data = tf.concat([batch_time, idx_pos_x, idx_pos_y], axis=-1) # timestep and position

      data_lbl = y_data[idx_time, idx_pos_x, idx_pos_y]



      # create physics loss batch
      idx_pos_p_x = np.expand_dims(random.choices(np.arange(0, locations_x.shape[0]), k=5), 1) # todo needs rework? more points?
      idx_pos_p_y = np.expand_dims(random.choices(np.arange(0, locations_y.shape[0]), k=5), 1)
      idx_time_p = np.expand_dims(np.arange(p_start_step, time_steps.shape[0]), 1)


      batch_pos_p_x = np.tile(idx_pos_p_x, (100, 1))
      batch_pos_p_y = np.tile(idx_pos_p_y, (100, 1))
      batch_time_p = np.expand_dims(np.repeat(time_steps[idx_time_p], 5), 1)

      input_data_physics = tf.concat([batch_time_p, batch_pos_p_x, batch_pos_p_y], axis=-1)

      pinn.fit(input_data, data_lbl,input_data_physics,input_data_semi, y_data_semi_pseudo, pseudo_physics_norm, tf_epochs, show_summary=show_summary)

  with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_meta, fp)
  with open(result_folder_name+ "/" + experiment_name +"/loss_epoch.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_epoch, fp)
  with open(result_folder_name + "/" + experiment_name +"/p_end.pkl", "wb") as fp:
    pickle.dump([p_start, p_end], fp)

  pinn.model.save_weights(result_folder_name + "/" + experiment_name +"/_weights")

  plot_loss(logger.loss_over_meta, pinn.physics_scale, plots_path + "loss", scaled=False)
  plot_loss(logger.loss_over_meta, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)


  y_pred, f_pred = pinn.predict_multiple_images(considered_steps=considered_steps, locations=[locations_x, locations_y])
  plot_comparison(t, y_lbl_error,y_pred, considered_steps,plots_path + "end")

  y_pred, f_pred = pinn.predict_multiple_images(considered_steps=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]), locations=[locations_x, locations_y])
  plot_comparison(t, y_lbl[[0, 2, 4, 6, 8, 10, 12, 14, 16]].flatten(),y_pred, considered_steps,plots_path + "learned")
  #y_pred, f_pred = pinn.predict(input_all)

  #plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, y_pred, plots_path + "res")

  #plot_terms_diff(t, f_pred, y_pred, y_lbl_all, plots_path + "res_error_all", p_plot_start=p_start_step)

  #f_pred, y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = pinn.f_model_detail(input_all)

  #plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_m2_dx_simul, y_m1_dx, y_m1_dx_simul, y_m2_dx2,y_m2_dx2_simul, y_m1_dx2, y_m1_dx2_simul, f_path_name=plots_path + "res_error_detail_after")

  print("Finished")



if __name__ == "__main__":
    main()




