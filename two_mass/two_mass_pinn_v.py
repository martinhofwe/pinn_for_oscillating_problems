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
from plot_two_mass import plot_solution, plot_terms_detail, plot_loss, plot_terms_diff

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
      data_error_m1, data_error_m2, physics_error_m1, physics_error_m2,y_pred, f_pred = self.__get_error_u()
      self.loss_over_epoch.append([data_error_m1, data_error_m2, physics_error_m1, physics_error_m2])
      print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e} " + custom)

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
    print("==================")
    data_error_m1, data_error_m2, physics_error_m1, physics_error_m2,_,_ = self.__get_error_u()
    self.loss_over_meta.append([data_error_m1, data_error_m2, physics_error_m1, physics_error_m2])
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e}  " + custom)

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
  def __init__(self, layers, h_activation_function, optimizer, logger,c1, d1, c2, d2,m1,m2, n_inputs=3,scaling_factor=1e12,physics_scale=1.00, p_norm="l2"):
    
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    pot_sin_layer = 0#(len(layers[1:])-1)//2
    for count, width in enumerate(layers[1:]):
        if width >1:
          if h_activation_function == "sine" and count == pot_sin_layer:
            print("sine af")
            self.model.add(tf.keras.layers.Dense(
              width, activation=tf.math.sin,
              kernel_initializer='glorot_normal'))
          else:
            print("tanh af")
            self.model.add(tf.keras.layers.Dense(
              width, activation=tf.nn.tanh,
              kernel_initializer='glorot_normal'))
        else:
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

    self.c1 = tf.constant(c1/scaling_factor,dtype=tf.float64)
    self.d1 = tf.constant(d1/scaling_factor,dtype=tf.float64)
    self.c2 = tf.constant(c2/scaling_factor,dtype=tf.float64)
    self.d2 = tf.constant(d2/scaling_factor,dtype=tf.float64)
    self.m1 = tf.constant(m1 / scaling_factor, dtype=tf.float64)
    self.m2 = tf.constant(m2 / scaling_factor, dtype=tf.float64)

    self.c164 = tf.constant(c1/scaling_factor,dtype=tf.float64)
    self.d164 = tf.constant(d1/scaling_factor,dtype=tf.float64)
    self.c264 = tf.constant(c2/scaling_factor,dtype=tf.float64)
    self.d264 = tf.constant(d2/scaling_factor,dtype=tf.float64)
    self.m164 = tf.constant(m1 / scaling_factor, dtype=tf.float64)
    self.m264 = tf.constant(m2 / scaling_factor, dtype=tf.float64)


    self.scaling_factor = tf.constant(scaling_factor,dtype=tf.float64)
    self.scaling_factor64 = tf.constant(scaling_factor,dtype=tf.float64)
    self.n_inputs = n_inputs
    self.optimizer = optimizer
    self.logger = logger
    self.physics_scale=physics_scale
    self.dtype = tf.float64
    self.p_norm = p_norm

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
      data_loss_semi = 0
      # np.mean((y_lbl_m1 - y[:, 0])**2), np.mean((y_lbl_m2 - y[:, 1])**2),
      data_loss = tf.reduce_mean((y_lbl[:, 0] - y[:, 0])**2) + tf.reduce_mean((y_lbl[:, 1] - y[:, 1])**2)
    if p_norm == "l2":
      physics_loss = tf.reduce_mean(f_pred[:, 0]**2) + tf.reduce_mean(f_pred[:, 1]**2)
    if p_norm == "l1":
      physics_loss = tf.reduce_mean(tf.math.abs(f_pred[:, 0])) + tf.reduce_mean(tf.math.abs(f_pred[:, 1]))

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
  def f_model(self,x):
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

    # Buidling the PINNs
    # where 𝑥1, 𝑥3 is position 𝑚1 and 𝑚2. 𝑥2, 𝑥4 is velocity of 𝑚1 and 𝑚2.
    # p loss for m2 (upper mass!)
    global y_lbl_m1_g, y_lbl_m2_g, y_lbl_m1_dx_g, y_lbl_m2_dx_g, y_lbl_m1_dx2_g, y_lbl_m2_dx2_g, u_lbl_g, up_lbl_g

    #m2_loss = (-y_m2_dx2 / self.scaling_factor) - (self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2 old and wrong
    #m2_loss = ((-self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2) - (y_m2_dx2/self.scaling_factor) richtig
    m2_loss = (((-self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2) - (y_m2_dx2 / self.scaling_factor))
    # p loss for m1
    #m1_loss = (-y_m1_dx2 / self.scaling_factor) + (self.c1 * (u - y_m1) + self.d1 * (u_dx - y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (y_m2_dx - y_m1_dx)) / self.m1 old and wrong
    m1_loss = ((self.c1 * (u - y_m1) + self.d1 * (u_dx - y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (y_m2_dx - y_m1_dx)) / self.m1) - (y_m1_dx2/self.scaling_factor)

    return tf.concat([m1_loss, m2_loss], axis=1)
  # The actual PINN
  def f_model_detail(self, x):
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
      self.logger.log_train_epoch(epoch, loss_value)

    self.logger.log_train_end(tf_epochs)

  def predict(self, x):
    y = self.model(x)
    f = self.f_model(x)
    return y, f

# from matlab script############################################################
def main():
  tf.keras.backend.set_floatx('float64')
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

  exp_len = 400
  p_norm = "l2"
  p_start_step = 50

  if task_id <= 13:
    width = 128
  else:
    width = 256

  #layer_dic = {0: 34, 1: 34, 2: 64, 3: 64, 4: 128, 5: 128, 6: 256, 7: 256, 8: 512, 9: 512, 10: 1024, 11: 1024}
  p_scale_dic = {0: 1, 1: 1, 2: 1e-1, 3: 1e-1, 4: 1e-2, 5: 1e-2, 6: 1e-3, 7: 1e-3, 8: 1e-4, 9: 1e-4, 10: 1e-5, 11: 1e-5, 12: 0, 13:0,
                 14: 1, 15: 1, 16: 1e-1, 17: 1e-1, 18: 1e-2, 19: 1e-2, 20: 1e-3, 21: 1e-3, 22: 1e-4, 23: 1e-4, 24: 1e-5, 25: 1e-5, 26: 0, 27:0}

  physics_scale_new = p_scale_dic[task_id]
  physics_scale = 0


  hidden_layers = 10 #layer_dic[task_id]
  layers = [3]
  for i in range(hidden_layers + 1):
    layers.append(width)
  layers.append(2) # for pos m1 and m2

  print("layers: ", layers)

  result_folder_name = 'res'
  #if not os.path.exists(result_folder_name):
  os.makedirs(result_folder_name, exist_ok=True)

  # fixed parameters
  logger = Logger(frequency=1000)

  p_start = 0
  p_end = 250
  # parameters ########################################################################################################
  ppi = 0
  max_iter_overall = 300_000#
  meta_epochs = 1 # todo without hard coded numbers
  lr = tf.Variable(1e-4)#tf.Variable(1e-4)
  tf_epochs_warm_up = 2000
  tf_epochs_train = int(max_iter_overall / meta_epochs)
  tf_epochs = max_iter_overall

  mode_frame = False
  tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.8, decay=0.)

  experiment_name = "two_mass_vanilla_ppi_" + str(ppi) + "_frame_" + str(mode_frame) + "_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_pn_" + p_norm + "_af_" + af_str + "_input_" + str(input_bool) + "_expl_" + str(exp_len) + "_ps_" + str(physics_scale_new) + "_pstart_" +str(p_start_step) + "_id_" + str(task_id)
  os.makedirs(result_folder_name + "/"+ experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)

  start_vec = [1.0, 0.0, 0.5, 0.0]
  y_m2_simul, y_m2_dx_simul, y_m2_dx2_simul, y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci, simul_constants = get_simulated_data_two_mass(start_vec, end_time=20, steps=4001, exp_len=exp_len, debug_data=True)
  m2, m1, c1, d1, c2, d2 = simul_constants

  data_start = 0
  scaling_factor = 1
  # Getting the data
  data_sampling = 10
  num_points = 200#200#4000#250
  x_data = tExci[data_start:num_points+1:data_sampling]
  y_data_m1 = y_m1_simul[data_start:num_points+1:data_sampling]*scaling_factor
  y_data_m2 = y_m2_simul[data_start:num_points+1:data_sampling]*scaling_factor
  y_data_all = np.hstack((y_data_m1, y_data_m2))

  y_lbl_m1 = y_m1_simul[data_start:]*scaling_factor
  y_lbl_m2 = y_m2_simul[data_start:] * scaling_factor
  y_lbl_all = np.hstack((y_lbl_m1, y_lbl_m2))

  t = tExci[data_start:]

  u_lbl = u_simul[data_start:]*scaling_factor
  up_lbl = up_simul[data_start:]*scaling_factor

  u_data = u_simul[data_start:num_points+1:data_sampling]*scaling_factor
  u_dx_data = up_simul[data_start:num_points+1:data_sampling]*scaling_factor

  input_all = tf.cast(tf.concat([t,u_lbl,up_lbl],axis=-1),tf.float64)
  input_data_physics = tf.cast(tf.concat([t[p_start_step:],u_lbl[p_start_step:],up_lbl[p_start_step:]],axis=-1),tf.float64)
  input_data = tf.cast(tf.concat([x_data,u_data,u_dx_data],axis=-1),tf.float64)

  # plotting
  plots_path = result_folder_name+ "/" + experiment_name+ "/"

  plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, None, plots_path + "exact_solution")

  fig = plt.figure()
  plt.plot(t[:min(exp_len, 500)], u_lbl[:min(exp_len, 500)]*scaling_factor, label="u close-up")
  plt.scatter(x_data,u_data,c="r")
  plt.legend()
  fig.savefig(result_folder_name + "/" + experiment_name + '/u_close_up.png')

  def error():
    y, f = pinn.predict(input_all)
    return np.mean((y_lbl_all[:, 0] - y[:, 0])**2), np.mean((y_lbl_all[:, 1] - y[:, 1])**2), np.mean(f[:, 0]**2), np.mean(f[:, 1]**2), y, f
  logger.set_error_fn(error)


  print("Frame mode: ", str(mode_frame))
  print("ppi", ppi)

  ### set data in logger for plots
  logger.t = t
  logger.y_lbl_all = y_lbl_all
  logger.plot_path = result_folder_name + "/" + experiment_name


  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, optimizer=tf_optimizer, logger=logger,c1=c1,d1=d1,c2=c2, d2=d2,m1=m1,m2=m2, n_inputs=layers[0],
                           scaling_factor=scaling_factor,physics_scale=physics_scale, p_norm=p_norm)

  try_next_semi_points = False
  input_data_semi, y_data_semi, y_data_semi_pseudo, pseudo_physics_norm, input_data_semi_new_pot, semi_candidates = None, None, None, None, None, None

  for i in range(2):
      print(str(i) + " / " +str(meta_epochs-1))
      if i==0:
          show_summary = True
      else:
          show_summary = False

      if i == 1:
        print("setting p scale")
        f_pred, y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = pinn.f_model_detail(input_all)
        plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_m2_dx_simul, y_m1_dx, y_m1_dx_simul, y_m2_dx2,
                          y_m2_dx2_simul, y_m1_dx2, y_m1_dx2_simul, f_path_name= plots_path + "res_error_detail_before")

        pinn.physics_scale = physics_scale_new

      pinn.fit(input_data, y_data_all,input_data_physics,input_data_semi, y_data_semi_pseudo, pseudo_physics_norm, tf_epochs, show_summary=show_summary)

  with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_meta, fp)
  with open(result_folder_name+ "/" + experiment_name +"/loss_epoch.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_epoch, fp)
  with open(result_folder_name + "/" + experiment_name +"/p_end.pkl", "wb") as fp:
    pickle.dump([p_start, p_end], fp)

  pinn.model.save_weights(result_folder_name + "/" + experiment_name +"/_weights")

  plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
  plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)

  y_pred, f_pred = pinn.predict(input_all)

  plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, y_pred, plots_path + "res")

  plot_terms_diff(t, f_pred, y_pred, y_lbl_all, plots_path + "res_error_all", p_plot_start=p_start_step)

  f_pred, y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = pinn.f_model_detail(input_all)

  plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_m2_dx_simul, y_m1_dx, y_m1_dx_simul, y_m2_dx2,
                          y_m2_dx2_simul, y_m1_dx2, y_m1_dx2_simul, f_path_name=plots_path + "res_error_detail_after")

  print("Finished")



if __name__ == "__main__":
    main()




