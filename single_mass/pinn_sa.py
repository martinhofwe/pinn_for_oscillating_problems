import sys

import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle

from data_simulation_single import get_simulated_data_single_mass
from plot_single_mass import plot_solution, plot_loss, plot_terms_diff


np.random.seed(1234)
tf.random.set_seed(1234)


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
  def clear_loss_lst(self):
    self.loss_over_meta = []
    self.loss_over_epoch = []

  def __get_elapsed(self):
    return datetime.utcfromtimestamp((time.time() - self.start_time)).strftime("%M:%S")
    #return datetime.fromtimestamp((time.time() - self.start_time)).strftime("%M:%S")# orig

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
    data_error, physics_error, data_error_scaled, physics_error_scaled = self.__get_error_u()
    total_error_mean = np.mean([data_error, physics_error])
    total_error_data = np.mean([data_error])
    if self.epoch_counter % self.frequency == 0:
      self.loss_over_epoch.append([data_error, physics_error, data_error_scaled, physics_error_scaled])
      print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  data error = {data_error:.4e} physics error = {physics_error:.4e}  " + custom)
    self.epoch_counter += 1
    return total_error_mean, total_error_data

  def log_train_opt(self, name):
    # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
    print(f"—— Starting {name} optimization ——")

  def log_train_end(self, epoch, custom=""):
    print("==================")
    data_error,physics_error, data_error_scaled, physics_error_scaled = self.__get_error_u()
    self.loss_over_meta.append([data_error, physics_error, data_error_scaled, physics_error_scaled])
    print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  data error = {data_error:.4e} physics error = {physics_error:.4e}  " + custom)

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
  #print("step: %.2f"%(global_time_list[-1]*1000))

def last_time():
  """Returns last interval records in millis."""
  global global_last_time, global_time_list
  if global_time_list:
    return 1000 * global_time_list[-1]
  else:
    return 0

def dot(a, b):
  """Dot product function since TensorFlow doesn't have one."""
  return tf.reduce_sum(a*b)

def verbose_func(s):
  print(s)

final_loss = None
times = []
def lbfgs(opfunc, x, config, state, do_verbose, log_fn):
  """port of lbfgs.lua, using TensorFlow eager mode.
  """

  if config.maxIter == 0:
    return

  global final_loss, times
  
  maxIter = config.maxIter
  maxEval = config.maxEval or maxIter*1.25
  tolFun = config.tolFun or 1e-5
  tolX = config.tolX or 1e-19
  nCorrection = config.nCorrection or 100
  lineSearch = config.lineSearch
  lineSearchOpts = config.lineSearchOptions
  learningRate = config.learningRate or 1
  isverbose = config.verbose or False

  # verbose function
  if isverbose:
    verbose = verbose_func
  else:
    verbose = lambda x: None

    # evaluate initial f(x) and df/dx
  f, g = opfunc(x)

  f_hist = [f]
  currentFuncEval = 1
  state.funcEval = state.funcEval + 1
  p = g.shape[0]

  # check optimality of initial point
  tmp1 = tf.abs(g)
  if tf.reduce_sum(tmp1) <= tolFun:
    verbose("optimality condition below tolFun")
    return x, f_hist

  # optimize for a max of maxIter iterations
  nIter = 0
  times = []
  while nIter < maxIter:
    start_time = time.time()
    
    # keep track of nb of iterations
    nIter = nIter + 1
    state.nIter = state.nIter + 1

    ############################################################
    ## compute gradient descent direction
    ############################################################
    if state.nIter == 1:
      d = -g
      old_dirs = []
      old_stps = []
      Hdiag = 1
    else:
      # do lbfgs update (update memory)
      y = g - g_old
      s = d*t
      ys = dot(y, s)
      
      if ys > 1e-10:
        # updating memory
        if len(old_dirs) == nCorrection:
          # shift history by one (limited-memory)
          del old_dirs[0]
          del old_stps[0]

        # store new direction/step
        old_dirs.append(s)
        old_stps.append(y)

        # update scale of initial Hessian approximation
        Hdiag = ys/dot(y, y)

      # compute the approximate (L-BFGS) inverse Hessian 
      # multiplied by the gradient
      k = len(old_dirs)

      # need to be accessed element-by-element, so don't re-type tensor:
      ro = [0]*nCorrection
      for i in range(k):
        ro[i] = 1/dot(old_stps[i], old_dirs[i])
        

      # iteration in L-BFGS loop collapsed to use just one buffer
      # need to be accessed element-by-element, so don't re-type tensor:
      al = [0]*nCorrection

      q = -g
      for i in range(k-1, -1, -1):
        al[i] = dot(old_dirs[i], q) * ro[i]
        q = q - al[i]*old_stps[i]

      # multiply by initial Hessian
      r = q*Hdiag
      for i in range(k):
        be_i = dot(old_stps[i], r) * ro[i]
        r += (al[i]-be_i)*old_dirs[i]
        
      d = r
      # final direction is in r/d (same object)

    g_old = g
    f_old = f
    
    ############################################################
    ## compute step length
    ############################################################
    # directional derivative
    gtd = dot(g, d)

    # check that progress can be made along that direction
    if gtd > -tolX:
      verbose("Can not make progress along direction.")
      break

    # reset initial guess for step size
    if state.nIter == 1:
      tmp1 = tf.abs(g)
      t = min(1, 1/tf.reduce_sum(tmp1))
    else:
      t = learningRate


    # optional line search: user function
    lsFuncEval = 0
    if lineSearch and isinstance(lineSearch) == types.FunctionType:
      # perform line search, using user function
      f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
      f_hist.append(f)
    else:
      # no line search, simply move with fixed-step

      x += t*d
      
      if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
        f, g = opfunc(x)
        lsFuncEval = 1
        f_hist.append(f)


    # update func eval
    currentFuncEval = currentFuncEval + lsFuncEval
    state.funcEval = state.funcEval + lsFuncEval

    ############################################################
    ## check conditions
    ############################################################
    if nIter == maxIter:
      break

    if currentFuncEval >= maxEval:
      # max nb of function evals
      verbose('max nb of function evals')
      break

    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <=tolFun:
      # check optimality
      verbose('optimality condition below tolFun')
      break
    
    tmp1 = tf.abs(d*t)
    if tf.reduce_sum(tmp1) <= tolX:
      # step size below tolX
      verbose('step size below tolX')
      break

    if tf.abs(f-f_old) < tolX:
      # function value changing less than tolX
      verbose('function value changing less than tolX'+str(tf.abs(f-f_old)))
      break

    if do_verbose:
      log_fn(nIter, f.numpy(), True)
      #print("Step %3d loss %6.5f msec %6.3f"%(nIter, f.numpy(), last_time()))
      record_time()
      times.append(last_time())

    if nIter == maxIter - 1:
      final_loss = f.numpy()


  # save state
  state.old_dirs = old_dirs
  state.old_stps = old_stps
  state.Hdiag = Hdiag
  state.g_old = g_old
  state.f_old = f_old
  state.t = t
  state.d = d

  return x, f_hist, currentFuncEval

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

class PhysicsInformedNN(object):
  def __init__(self, layers, h_activation_function, optimizer, logger,c,d,m, nr_p_points, nr_d_points, n_inputs=3,scaling_factor=1e12,physics_scale=1.0, p_norm="l2"):
    
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    for width in layers[1:]:
        if width >1:
            self.model.add(tf.keras.layers.Dense(
              width, activation=tf.nn.tanh,
              kernel_initializer='glorot_normal'))
            #self.model.add(tf.keras.layers.Dropout(rate=0.1))
        else:
            self.model.add(tf.keras.layers.Dense(
              width, activation=h_activation_function,
              kernel_initializer='glorot_normal'))


    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        self.sizes_w.append(int(width * layers[1]))
        self.sizes_b.append(int(width if i != 0 else layers[1]))

    self.m = tf.constant(m / scaling_factor, dtype=tf.float32)
    self.c = tf.constant(c/scaling_factor,dtype=tf.float32)
    self.d = tf.constant(d/scaling_factor,dtype=tf.float32)

    self.scaling_factor = tf.constant(scaling_factor,dtype=tf.float32)
    self.n_inputs = n_inputs
    self.optimizer = optimizer
    self.logger = logger
    self.physics_scale=physics_scale
    self.dtype = tf.float32
    self.p_norm = p_norm

    self.physics_weights = tf.Variable(tf.random.uniform([nr_p_points, 1], dtype=tf.float32))
    self.data_weights = tf.Variable(tf.random.uniform([nr_d_points, 1], dtype=tf.float32))  # todo not hard coded
    self.optimizer_sa = tf.keras.optimizers.Adam(lr=0.005, beta_1=.90)

    # Separating the collocation coordinates
    #self.x = tf.convert_to_tensor(X, dtype=self.dtype)
    # for early stopping implementation
    self.ea_elements = 200
    self.ea_overlap = 0
    self.ea_error_lst = []
    self.ea_error_data_lst = []
    self.ea_mean_lst = []
    self.ea_median_lst = []
    self.ea_mean_data_lst = []
    self.ea_median_data_lst = []

  def __get_save_intermediate(self):
    return self.save_intermediate()

  def set_save_intermediate(self, save_f):
    self.save_intermediate = save_f
    
  # Defining custom loss
  @tf.function
  def __loss(self,x, y_lbl,x_physics, y,p1_weight, d1_weight, x_semi_begin=None, semi_scale=None, p_norm=None):
    f_pred = self.f_model(x_physics)
    if x_semi_begin is not None:
      data_loss =  tf.reduce_mean((y_lbl[:x_semi_begin] - y[:x_semi_begin])**2)
      #print("###")
      #print(semi_scale.shape)
      #print(y_lbl[x_semi_begin:].shape)
      #print(y[x_semi_begin:].shape)
      #print("####")
      data_loss_semi =  tf.reduce_mean(semi_scale*((y_lbl[x_semi_begin:] - y[x_semi_begin:])**2))
    else:
      data_loss_semi = 0
      data_loss =  tf.reduce_mean((d1_weight*(y_lbl - y))**2)
    if p_norm == "l2":
      physics_loss = tf.reduce_mean(p1_weight*(f_pred**2))
    if p_norm == "l1":
      physics_loss = tf.reduce_mean(tf.math.abs(p1_weight*f_pred))

    return data_loss+data_loss_semi+self.physics_scale*physics_loss

  def __grad_data(self, x, y_lbl):
    with tf.GradientTape() as tape:
      tape.watch(x)  
      y = self.model(x)
      data_loss =  tf.reduce_mean(tf.square(y_lbl - y)) 
      print("___________")
      print(data_loss)
    return data_loss, tape.gradient(data_loss, self.__wrap_training_variables())
  
  @tf.function
  def __grad(self, x, y_lbl, x_physics, x_semi_begin, semi_scale,p1_weight, d1_weight):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)  
      loss_value = self.__loss(x,y_lbl, x_physics, self.model(x),p1_weight, d1_weight, x_semi_begin, semi_scale, p_norm=self.p_norm)
      grads_data = tape.gradient(loss_value, d1_weight)
      grads_p = tape.gradient(loss_value, p1_weight)
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables()), grads_data, grads_p

  def __wrap_training_variables(self):
    var = self.model.trainable_variables
    return var

  # The actual PINN
  @tf.function
  def f_model(self,x):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime

    # old version
    # with tf.GradientTape() as tape:
    #   # Watching the two inputs we’ll need later, x and t
    #   tape.watch(x)
    #   # Packing together the inputs
    #   #X_f = tf.stack([self.x_f[:,0], self.t_f[:,0]], axis=1)
    #   with tf.GradientTape() as tape2:
    #   # Getting the prediction
    #       tape2.watch(x)
    #       y = self.model(x)
    #   # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
    #   y_dx = tape2.gradient(y, x)
    # # Getting the other derivatives
    # y_dx2 = tape.gradient(y_dx, x)
    #
    # y_dx = tf.expand_dims(y_dx[...,0],-1)
    # y_dx2 = tf.expand_dims(y_dx2[...,0],-1)

    #########################################
    # https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb#scrollTo=azOBDHMoZEkn
    with tf.GradientTape(persistent=True) as tape:
      # Split t and x to compute partial derivatives
      x_ = tf.expand_dims(x[...,0],-1)
      u = tf.expand_dims(x[...,1],-1)
      u_dx = tf.expand_dims(x[...,2],-1)

      # Variables t and x are watched during tape
      # to compute derivatives u_t and u_x
      tape.watch(x_)
      x_watched_full = tf.concat([x_,u,u_dx],axis=-1)

      # Determine residual
      y = self.model(x_watched_full)
      y_dx = tape.gradient(y, x_)
    y_dx2 = tape.gradient(y_dx, x_)

    del tape

    #####################################
    
    y_dx = tf.expand_dims(y_dx[...,0],-1)
    y_dx2 = tf.expand_dims(y_dx2[...,0],-1)
    # Letting the tape go
    #del tape
    #print("____dx_______")
    #print(self.k*y+y_dx2+self.mu*y_dx)

    # Buidling the PINNs
    return (-y_dx2/self.scaling_factor) -(self.d/self.m)*y_dx - (self.c/self.m)*y + (self.d/self.m)*u_dx + (self.c/self.m)*u

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
    x_physics =  tf.convert_to_tensor(x_physics, dtype=self.dtype)

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
      # Optimization step
      #xy = tf.random.shuffle(xy)
      if epoch <0:
        loss_value, grads = self.__grad_data(x, y_lbl, x_physics)
        self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      else:
        loss_value, grads, grads_data, grads_p = self.__grad(x, y_lbl, x_physics, x_semi_begin, semi_scale, self.physics_weights, self.data_weights)
        self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
        self.optimizer_sa.apply_gradients(zip([-grads_data], [self.data_weights]))
        self.optimizer_sa.apply_gradients(zip([-grads_p], [self.physics_weights]))

      # Early detection
      total_error_mean, data_error_mean = self.logger.log_train_epoch(epoch, loss_value)
      self.ea_error_lst.append(total_error_mean)
      self.ea_error_data_lst.append(data_error_mean)

      if len(self.ea_error_lst) == (self.ea_elements + (self.ea_elements - self.ea_overlap)):
        prev_error_mean = np.mean(self.ea_error_lst[:self.ea_elements])
        curr_error_mean = np.mean(self.ea_error_lst[(self.ea_elements - self.ea_overlap)::])

        prev_error_mean_data = np.mean(self.ea_error_data_lst[:self.ea_elements])
        curr_error_mean_data = np.mean(self.ea_error_data_lst[(self.ea_elements - self.ea_overlap)::])

        prev_error_median = np.median(self.ea_error_lst[:self.ea_elements])
        curr_error_median = np.median(self.ea_error_lst[(self.ea_elements - self.ea_overlap)::])

        prev_error_median_data = np.median(self.ea_error_data_lst[:self.ea_elements])
        curr_error_median_data = np.median(self.ea_error_data_lst[(self.ea_elements - self.ea_overlap)::])

        if curr_error_mean > prev_error_mean:
          self.ea_mean_lst.append([prev_error_mean, curr_error_mean, epoch])
        if curr_error_median > prev_error_median:
          self.ea_median_lst.append([prev_error_median, curr_error_median, epoch])
        if curr_error_mean_data > prev_error_mean_data:
          self.ea_mean_data_lst.append([prev_error_mean_data, curr_error_mean_data, epoch])
        if curr_error_median_data > prev_error_median_data:
          self.ea_median_data_lst.append([prev_error_median_data, curr_error_median_data, epoch])

        del self.ea_error_lst[:(self.ea_elements - self.ea_overlap)]

      # store res if computation time runs out so that you have something
      if epoch % 50_000 == 0:
        print("store save")
        self.save_intermediate()
        print("store save end")


    self.logger.log_train_end(tf_epochs)

    return self.ea_mean_lst, self.ea_median_lst, self.ea_mean_data_lst, self.ea_median_data_lst

  def predict(self, x):
    y = self.model(x)
    f = self.f_model(x)
    return y, f

# from matlab script############################################################
def main():
  task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
  #task_id = int(sys.argv[1])
  print("task_id: ", task_id)
  if task_id % 2 == 0:
    act_func = tf.nn.tanh
    af_str = "tanh"
  else:
    act_func = tf.math.sin
    af_str = "sin"

  p_norm = "l2"

  scl_dic = {0: 100, 1: 100, 2: 10, 3: 10, 4:1, 5:1, 6:0.1, 7:0.1}

  hidden_layers = 5
  layers = [3]
  for i in range(hidden_layers + 1):
    layers.append(32) # todo test
  layers.append(1)

  print("layers: ", layers)

  result_folder_name = 'res'
  if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

  # fixed parameters
  logger = Logger(frequency=1000)
  physics_scale = 1e-4
  p_start = 1
  p_end = 3999#p_end_dic[task_id]
  # parameters ########################################################################################################
  ppi = 0
  max_iter_overall = 3_000_000#3_000_000#3000000
  meta_epochs = 1 # todo without hard coded numbers
  lr = tf.Variable(1e-4)
  tf_epochs_warm_up = 2000
  tf_epochs_train = int(max_iter_overall / meta_epochs)
  tf_epochs = max_iter_overall

  mode_frame = False
  tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.8, decay=0.)

  experiment_name = "single_mass_sa_ppi_" + str(ppi) + "_frame_" + str(mode_frame) + "_h_l_" + str(hidden_layers) + "_pn_" + p_norm + "_af_" + af_str + "_p_end_" +str(p_end) +"_scl_" + str(scl_dic[task_id]) + "_" + str(task_id)
  os.makedirs(result_folder_name + "/"+ experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)
  # data creation##########################################################################################################


  start_vec = [1, 0]
  m1 = (30000/2)*scl_dic[task_id]
  css = 0.5e6 * 2
  dss = 1.5e4 * 2
  exp_len = 4000
  y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci, simul_constants = get_simulated_data_single_mass(start_vec, end_time=10, steps=4000, exp_len=exp_len, m1=m1, css=css,dss=dss, debug_data=True)

  # Getting the data
  data_sampling = 10
  scaling_factor = 1#1e12
  num_points = 250
  data_start = 0


  # Getting the data

  x_data = np.expand_dims(tExci[data_start], 1)
  y_data = np.expand_dims(y_m1_simul[data_start]*scaling_factor, 1)
  y_lbl = y_m1_simul[data_start:]*scaling_factor
  u_lbl = u_simul[data_start:]*scaling_factor
  up_lbl = up_simul[data_start:]*scaling_factor
  t = tExci[data_start:]
  u_data = np.expand_dims(u_simul[data_start], 1)*scaling_factor
  u_dx_data = np.expand_dims(up_simul[data_start], 1) *scaling_factor

  input_all = tf.cast(tf.concat([t,u_lbl,up_lbl],axis=-1),tf.float32)
  input_data_physics = tf.cast(tf.concat([t[p_start:p_end],u_lbl[p_start:p_end],up_lbl[p_start:p_end]],axis=-1),tf.float32)
  input_data = tf.cast(tf.concat([x_data,u_data,u_dx_data],axis=-1),tf.float32)

  loss_scale_term = np.exp(-(dss)/(2*m1)*t)

  plot_solution(t, y_lbl, x_data, y_data, None,p_end, result_folder_name+"/" + experiment_name + '/exact_solution')
  plot_solution(t, y_lbl/loss_scale_term, x_data, y_data, None,p_end, result_folder_name+"/" + experiment_name + '/exact_solution_scaled')




  def error():
    y, f = pinn.predict(input_all)
    return np.mean((y_lbl - y)**2),np.mean(f**2), np.mean(((y_lbl - y)**2)/loss_scale_term), np.mean((f**2)/loss_scale_term)
  logger.set_error_fn(error)


  def store_intermediate():
    with open(result_folder_name + "/" + experiment_name + "/ea_mean_lst.pkl", "wb") as fp:
      pickle.dump(pinn.ea_mean_lst, fp)

    with open(result_folder_name + "/" + experiment_name + "/ea_median_lst.pkl", "wb") as fp:
      pickle.dump(pinn.ea_median_lst, fp)

    with open(result_folder_name + "/" + experiment_name + "/ea_mean_data_lst.pkl", "wb") as fp:
      pickle.dump(pinn.ea_mean_data_lst, fp)

    with open(result_folder_name + "/" + experiment_name + "/ea_median_data_lst.pkl", "wb") as fp:
      pickle.dump(pinn.ea_median_data_lst, fp)

    with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
      pickle.dump(logger.loss_over_meta, fp)
    with open(result_folder_name + "/" + experiment_name + "/loss_epoch.pkl", "wb") as fp:
      pickle.dump(logger.loss_over_epoch, fp)
    with open(result_folder_name + "/" + experiment_name + "/p_end.pkl", "wb") as fp:
      pickle.dump([p_start, p_end], fp)

    pinn.model.save_weights(result_folder_name + "/" + experiment_name + "/_weights")

    plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name + "/" + experiment_name + '/loss', False)
    plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name + "/" + experiment_name + '/loss_scaled', True)
    plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name + "/" + experiment_name + '/loss_exp', False,
              True)
    plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name + "/" + experiment_name + '/loss_exp_scaled',
              True, True)

    y_pred, f_pred = pinn.predict(input_all)

    plot_solution(t, y_lbl, x_data, y_data, y_pred, p_end, result_folder_name + "/" + experiment_name + '/res')
    plot_terms_diff(t, f_pred, y_pred, y_lbl, result_folder_name + "/" + experiment_name + '/res_error_all',
                    p_plot_start=0)
    plt.close('all')

  print("Frame mode: ", str(mode_frame))
  print("ppi", ppi)


  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, optimizer=tf_optimizer, logger=logger, m=simul_constants[0], c=simul_constants[1], d=simul_constants[2], nr_d_points=input_data.shape[0],nr_p_points=input_data_physics.shape[0], n_inputs=layers[0],
                           scaling_factor=scaling_factor,physics_scale=physics_scale, p_norm=p_norm)

  pinn.set_save_intermediate(store_intermediate)


  try_next_semi_points = False
  input_data_semi, y_data_semi, y_data_semi_pseudo, pseudo_physics_norm, input_data_semi_new_pot, semi_candidates = None, None, None, None, None, None

  for i in range(meta_epochs):
      print(str(i) + " / " +str(meta_epochs-1))
      if i==0:
          show_summary = True
      else:
          show_summary = False

      ea_mean_lst, ea_median_lst, ea_mean_data_lst, ea_median_data_lst = pinn.fit(input_data, y_data,input_data_physics,input_data_semi, y_data_semi_pseudo, pseudo_physics_norm, tf_epochs, show_summary=show_summary)




  with open(result_folder_name + "/" + experiment_name + "/ea_mean_lst.pkl", "wb") as fp:
    pickle.dump(ea_mean_lst, fp)

  with open(result_folder_name + "/" + experiment_name + "/ea_median_lst.pkl", "wb") as fp:
    pickle.dump(ea_median_lst, fp)

  with open(result_folder_name + "/" + experiment_name + "/ea_mean_data_lst.pkl", "wb") as fp:
    pickle.dump(ea_mean_data_lst, fp)

  with open(result_folder_name + "/" + experiment_name + "/ea_median_data_lst.pkl", "wb") as fp:
    pickle.dump(ea_median_data_lst, fp)

  with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_meta, fp)
  with open(result_folder_name+ "/" + experiment_name +"/loss_epoch.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_epoch, fp)
  with open(result_folder_name + "/" + experiment_name +"/p_end.pkl", "wb") as fp:
    pickle.dump([p_start, p_end], fp)

  pinn.model.save_weights(result_folder_name + "/" + experiment_name +"/_weights")




  plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name+"/" + experiment_name + '/loss', False)
  plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name+"/" + experiment_name + '/loss_scaled', True)
  plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name+"/" + experiment_name + '/loss_exp', False, True)
  plot_loss(logger.loss_over_epoch, physics_scale, result_folder_name+"/" + experiment_name + '/loss_exp_scaled', True, True)


  y_pred, f_pred = pinn.predict(input_all)

  plot_solution(t, y_lbl, x_data, y_data, y_pred,p_end, result_folder_name+"/" + experiment_name + '/res')
  plot_terms_diff(t, f_pred, y_pred, y_lbl, result_folder_name+"/" + experiment_name + '/res_error_all' , p_plot_start=0)


  print("Finished")



if __name__ == "__main__":
    main()




