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
from plot_two_mass import plot_solution, plot_terms_detail, plot_loss, plot_terms_diff


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
      data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
      self.loss_over_epoch.append([data_error_m1, data_error_m2, physics_error_m1, physics_error_m2])
    if epoch % self.print_freq == 0:
        if not epoch % self.save_loss_freq == 0:
          data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
        print(f"{'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e} ")

  def log_train_end(self, epoch, log_data):
    print("==================")
    data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e}  ")


class PhysicsInformedNN(object):
  def __init__(self, layers, h_activation_function, logger,simul_constants ,domain, physics_scale,data_scale, lr, data, simul_results, storage_path, task_id):
    
    inputs, outputs = self.setup_layers2(layers, h_activation_function)
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="concat_model")
    self.storage_path = storage_path
    self.dtype = tf.float32
    m2, m1, c1, d1, c2, d2 = simul_constants
    scaling_factor = 1.0
    self.c1 = tf.constant(c1/scaling_factor,dtype=self.dtype)
    self.d1 = tf.constant(d1/scaling_factor,dtype=self.dtype)
    self.c2 = tf.constant(c2/scaling_factor,dtype=self.dtype)
    self.d2 = tf.constant(d2/scaling_factor,dtype=self.dtype)
    self.m1 = tf.constant(m1 / scaling_factor, dtype=self.dtype)
    self.m2 = tf.constant(m2 / scaling_factor, dtype=self.dtype)

    self.x_ic, self.y_lbl_ic, self.x_physics, self.input_all, self.y_lbl_all = data
    self.y_m2_simul, self.y_m2_dx_simul, self.y_m2_dx2_simul, self.y_m1_simul, self.y_m1_dx_simul, self.y_m1_dx2_simul, _, _, _ = simul_results
    self.scaling_factor = tf.constant(1.0,dtype=self.dtype)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.8, decay=0.)

    self.logger = logger
    self.physics_scale = physics_scale
    self.data_loss_scl = data_scale
    self.domain = domain
    self.storage_path = ""
    self.collocation_cuts = []
    
    self.curr_epoch = 0
    self.term_scaling = tf.Variable(0.0, dtype=self.dtype)

  def setup_layers2(self, layers, h_activation_function):
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for count, width in enumerate(layers[1:-1]):
        if h_activation_function == "sine":
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
  def f_model(self, x_, epoch):
      with tf.GradientTape() as tape:
        tape.watch(x_)
        y = self.pred_with_grad(x_)
        y_dx = y[:, 2:]
        y_dx2_all = tape.batch_jacobian(y_dx, x_)[:, :, 0]
      del tape

      y_m1 = tf.expand_dims(y[:, 0], -1)
      y_m2 = tf.expand_dims(y[:, 1], -1)
      y_m1_dx = tf.expand_dims(y_dx[..., 0], -1)
      y_m2_dx = tf.expand_dims(y_dx[..., 1], -1)
      y_m1_dx2 = tf.expand_dims(y_dx2_all[..., 0], -1)
      y_m2_dx2 = tf.expand_dims(y_dx2_all[..., 1], -1)

      if epoch >= self.terms_stop or epoch == -1:
          term_scaling = tf.math.minimum(tf.cast(((epoch+1-self.terms_stop)/(self.terms_stop2+1-self.terms_stop)), tf.float32), 1.0)
      else:
          term_scaling = 0.0 

      m1_loss = ((self.c1 * (- y_m1) + self.d1 * (- y_m1_dx) + term_scaling*(self.c2 * (y_m2 - y_m1)) + term_scaling*(self.d2 * (y_m2_dx - y_m1_dx))) / self.m1) - (y_m1_dx2 / self.scaling_factor)  # for input 0
      m2_loss = (( (-self.c2 * (y_m2 - term_scaling*y_m1) -  self.d2 * (y_m2_dx - term_scaling*(y_m1_dx))) / self.m2) - (y_m2_dx2 / self.scaling_factor))

      return [m1_loss, m2_loss]

  def f_model_detail(self, x_):
      with tf.GradientTape() as tape:
        tape.watch(x_)
        y = self.pred_with_grad(x_)
        y_dx = y[:, 2:]
        y_dx2_all = tape.batch_jacobian(y_dx, x_)[:, :, 0]
      del tape

      y_m1 = tf.expand_dims(y[:, 0], -1)
      y_m2 = tf.expand_dims(y[:, 1], -1)
      y_m1_dx = tf.expand_dims(y_dx[..., 0], -1)
      y_m2_dx = tf.expand_dims(y_dx[..., 1], -1)
      y_m1_dx2 = tf.expand_dims(y_dx2_all[..., 0], -1)
      y_m2_dx2 = tf.expand_dims(y_dx2_all[..., 1], -1)

      m1_loss = ((self.c1 * (- y_m1) + self.d1 * (- y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (y_m2_dx - y_m1_dx)) / self.m1) - (y_m1_dx2/self.scaling_factor) # for input 0 needs to be substituted with input and dx input
      m2_loss = (((-self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2) - (y_m2_dx2 / self.scaling_factor))

      return [m1_loss, m2_loss], [y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2]

  def sample_collocation_points(self, nr_p_points):
    t_points = np.expand_dims((self.domain[1] - self.domain[0]) * random(nr_p_points) + self.domain[0], 1)
    return tf.cast(t_points, tf.float32)

  def define_p_cuts(self, x_physics, nr_points=10_000):
    nr_cuts = int(np.ceil(x_physics.shape[0] / nr_points))
    for i in range(nr_cuts):
       self.collocation_cuts.append([i*nr_points, np.min([(i+1)*nr_points, x_physics.shape[0]])])


  def pred_with_grad(self, x_points):
    with tf.GradientTape() as t:
      t.watch(x_points)
      pred = self.model(x_points)
    dx = t.batch_jacobian(pred, x_points)[:, :, 0]
    y_pred_full = tf.concat((pred, dx), axis=1)
    return y_pred_full

  def calc_loss_ic(self):
    diff = self.y_lbl_ic - self.pred_with_grad(self.x_ic)
    ic_loss =tf.reduce_mean(tf.square(diff))
    return ic_loss


  def calc_physics_loss(self, x_col, epoch):
    m1_loss, m2_loss = self.f_model(x_col, epoch)
    m1_loss_mean = tf.reduce_mean(tf.square(m1_loss))
    m2_loss_mean = tf.reduce_mean(tf.square(m2_loss))
    return [m1_loss_mean, m2_loss_mean]

  @tf.function
  def train_step(self, x_col, epoch, physics_scale):
    with tf.GradientTape(persistent=True) as tape:

      # data loss / initial condition
      data_loss = self.calc_loss_ic()

      # physics loss
      m1_p_loss, m2_p_loss = self.calc_physics_loss(x_col, epoch)
      combined_weighted_loss = self.data_loss_scl*data_loss + physics_scale * (m1_p_loss + m2_p_loss)

      # retrieve gradients
    grads = tape.gradient(combined_weighted_loss, self.model.weights)
    del tape
    self.optimizer.apply_gradients(zip(grads,self.model.weights))


    physics_losses, pred_parameters = self.f_model_detail(self.input_all)
    m1_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_parameters[0]) - tf.squeeze(self.y_lbl_all[:, 0])))
    m2_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_parameters[1]) - tf.squeeze(self.y_lbl_all[:, 1])))
    m1_loss, m2_loss = physics_losses
    f_pred_m1 = tf.reduce_mean(tf.square(m1_loss))
    f_pred_m2 = tf.reduce_mean(tf.square(m2_loss))
    log_data = [m1_data_loss, m2_data_loss, f_pred_m1, f_pred_m2]

    return combined_weighted_loss, log_data, pred_parameters, physics_losses

  def fit(self):
    self.logger.log_train_start(self)
    for epoch in range(self.tf_epochs):
      if epoch < self.terms_stop:
          physics_scale = self.physics_scale
      else:
          physics_scale = self.physics_scale2
          print("Switching p loss: ", physics_scale)
      loss_value, log_data, pred_parameters, physics_losses = self.train_step(self.x_physics, tf.cast(epoch, tf.int32), physics_scale)
      #loss_value, log_data, pred_parameters, physics_losses = self.train_step(self.sample_collocation_points(1_000)) # todo as parameter
      # log train loss and errors specified in logger error
      self.logger.log_train_epoch(epoch, loss_value, log_data)

      if epoch % 25_000 == 0:
        self.store_intermediate_result(epoch, pred_parameters, physics_losses)
    self.logger.log_train_end(self.tf_epochs,  log_data)


  def predict(self, x):
    y = self.model(x)
    f_m1, f_m2 = self.f_model(x, tf.cast(-1, tf.int32))
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

  act_func = "sine"
  af_str = "sin"

  ic_points_idx = [0]
  d_p_string = "vanilla"
  '''
  if task_id <= 1:
    ic_points_idx = [7499]
    d_p_string = "end"
  elif 1 < task_id <= 3:
    ic_points_idx = [0, 7499]
    d_p_string = "start_end"
  elif 3 < task_id <= 5:
    ic_points_idx = [x for x in range(0, 7499, 500)]
    d_p_string = "cont"
  '''
  lr = tf.Variable(1e-4)
  print("ic points: ", ic_points_idx)
  hidden_layers = 7
  
  if task_id == 0:
    terms_stop = 100_000
    terms_stop2 = 1_200_000
  if task_id == 1:
     terms_stop = 100_000
     terms_stop2 = 200_000
  if task_id == 2:
      terms_stop = 500_000
      terms_stop2 = 1_200_000
  if task_id == 3:
      terms_stop = 500_000
      terms_stop2 = 850_000


  weight_factor = 2

  physics_scale = tf.Variable(1e-6)
  data_loss_scl = tf.Variable(1.0)
  ######################################################################################################################
  # Fixed parameters PINN
  training_epochs = 1_200_000
  width = 128
  layers = get_layer_list(nr_inputs=1, nr_outputs=2, nr_hidden_layers=hidden_layers, width=width)

  # save_loss_freq = how often test error is saved
  logger = Logger(save_loss_freq=1, print_freq=1_000)

  # Data simulation
  #weight_factor = 0.0002
  c2_in = (0.5e8 * 2)
  d2_in = (1.5e3 * 2)
  m2 = 3_000
  m1 = 15_000 * 2
  x_d = [0, 20]  # time domain of simulation
  exp_len = 7_500  # number of points are used in training / testing the PINN
  steps = 100_000  # number of steps in time domain

  ###
  # weight_factor = 20
  # x_d = [0, 6]
  # c2_in = 0.5e6 * 2  # (0.5e8 * 2)  # orig: 0.5e6 * 2
  # d2_in = 1.5e4 * 2  # (1.5e3 * 2)  # orig: 1.5e4 * 2
  # m2 = 15_000  # 3_000  # orig : 15_000 * weight_factor
  # m1 = 3_000  # 15_000 * weight_factor
  # end = 6
  # exp_len = 750
  # steps = 4_001

  ###

  start_vec = [1.0, 0.0, 0.5, 0.0] # pos m2, dx pos m2, pos m1, dx pos m1
  simul_results, simul_constants = get_simulated_data_two_mass(start_vec, end_time=x_d[1], steps=steps, exp_len=exp_len, m1=m1, m2=m2, css=c2_in,dss=d2_in,debug_data=True)
  y_m2_simul, y_m2_dx_simul, y_m2_dx2_simul, y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci = simul_results
  m2, m1, c1, d1, c2, d2 = simul_constants
  #####################################

  # Getting the data
  p_start_step = 1
  p_sampling_rate = 1
  data_start = 0
  t = tExci[data_start:]
  domain = [t[0], t[-1]]

  # ic condition
  x_data = tExci[ic_points_idx]
  y_data_m1 = y_m1_simul[ic_points_idx]
  y_data_m2 = y_m2_simul[ic_points_idx]


  y_data_all = np.hstack((y_data_m1, y_data_m2, y_m1_dx_simul[ic_points_idx], y_m2_dx_simul[ic_points_idx]))# pos m1 (lower mass), pos m2, dx pos m1, dx pos m2
  input_data = tf.cast(x_data, tf.float32)

  # define data sets
  input_data_physics = tf.cast(t[p_start_step:exp_len:p_sampling_rate],tf.float32)
  x_ic = tf.constant(input_data, dtype=tf.float32)
  y_lbl_ic = tf.constant(y_data_all, dtype=tf.float32)
  x_physics = tf.convert_to_tensor(input_data_physics, dtype=tf.float32) # not used in the moment since we randomly sample physic points

  # test set, used to calcualte logged loss not used in training
  y_lbl_m1 = y_m1_simul[data_start:]
  y_lbl_m2 = y_m2_simul[data_start:]
  y_lbl_all = np.hstack((y_lbl_m1, y_lbl_m2))
  y_lbl_all = tf.convert_to_tensor(y_lbl_all, dtype=tf.float32)
  input_all = tf.convert_to_tensor(t, dtype=tf.float32)
  pinn_data = [x_ic, y_lbl_ic, x_physics, input_all, y_lbl_all]

  # Setting up folder structure # todo clean up
  result_folder_name = 'res'
  os.makedirs(result_folder_name, exist_ok=True)
  experiment_name = "two_mass_vanilla_terms_big_0_h_l_" + str(hidden_layers) + "_w_" + str(width) + "_af_" + af_str + "_lr_" + str(lr.numpy())+ "_expl_" + str(exp_len)+ "_steps_" + str(steps) +"_ds_" + str(data_loss_scl.numpy())+ "_ps_" + str(physics_scale.numpy())+ "_wf_" + str(weight_factor) + "_dp_" + d_p_string + "_id_" + str(task_id)
  print("Config name: ", experiment_name)
  os.makedirs(result_folder_name + "/" + experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)
  plots_path = result_folder_name + "/" + experiment_name+ "/"
    
  ###ä#########
  m1_loss_hand = ((c1 * (- y_m1_simul) + d1 * (- y_m1_dx_simul) + c2 * (y_m2_simul - y_m1_simul) + d2 * (y_m2_dx_simul - y_m1_dx_simul)) / m1) - (y_m1_dx2_simul / 1.0)
  m2_loss_hand = (((-c2 * (y_m2_simul - y_m1_simul) - d2 * (y_m2_dx_simul - y_m1_dx_simul)) / m2) - (y_m2_dx2_simul / 1.0))
  fig_res = plt.figure(figsize=(12, 12))
  plt.subplot(2, 1, 1)
  plt.plot(tExci[1:-1], m2_loss_hand[1:-1], label="m2 hand")
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.plot(tExci[1:-1], m1_loss_hand[1:-1], label="m1 hand")
  plt.legend()
  fig_res.savefig(plots_path + "hand_solution" + '.svg', format='svg', dpi=1200)
  #####
  # plotting solution
  plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, None, plots_path + "exact_solution")

  fig_res = plt.figure(figsize=(16, 9))
  plt.subplot(2, 1, 1)
  plt.scatter(t[p_start_step:], y_lbl_m2[1:], label="p points", c="r", s=1)
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.scatter(t[p_start_step:],  y_lbl_m1[1:], label="p points", c="r", s=1)
  plt.legend()
  fig_res.savefig(result_folder_name + "/" + experiment_name+'/p_points_dist.svg', format='svg', dpi=1200)



  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, logger=logger, simul_constants=simul_constants, domain=domain, physics_scale=physics_scale,data_scale=data_loss_scl, lr=lr, data=pinn_data, simul_results=simul_results, storage_path=plots_path, task_id=task_id)
  pinn.tf_epochs = training_epochs # todo integrate
  pinn.storage_path = plots_path # todo integrate
  pinn.terms_stop = tf.cast(terms_stop, tf.int32)
  pinn.terms_stop2 = tf.cast(terms_stop2, tf.int32)
  pinn.physics_scale2 = tf.cast(1e-6, tf.int32)

  pinn.fit()

  # plot results
  plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss", scaled=False)
  plot_loss(logger.loss_over_epoch, pinn.physics_scale, plots_path + "loss_scaled", scaled=True)
  y_pred, f_m1, f_m2 = pinn.predict(input_all)
  plot_solution(t, y_lbl_m2, y_lbl_m1, x_data, y_data_m2, y_data_m1, y_pred, plots_path + "res_end")
  plot_terms_diff(t, f_m1, f_m2, y_pred, y_lbl_all, plots_path + "res_error_all", p_plot_start=p_start_step)
  _,  params = pinn.f_model_detail(input_all)
  y_m1, y_m2, y_m1_dx, y_m2_dx, y_m1_dx2, y_m2_dx2 = params
  plot_terms_detail(t, y_m2, y_lbl_m2, y_m1, y_lbl_m1, y_m2_dx, y_m2_dx_simul, y_m1_dx, y_m1_dx_simul, y_m2_dx2,y_m2_dx2_simul, y_m1_dx2, y_m1_dx2_simul, f_path_name=plots_path + "res_error_detail_after")

  print("Finished")



if __name__ == "__main__":
    main()



