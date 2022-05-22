import scipy.signal as sig
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
import sys

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
    if self.epoch_counter % self.frequency == 0:
      data_error, physics_error = self.__get_error_u()
      self.loss_over_epoch.append([data_error, physics_error])
      print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e}  data error = {data_error:.4e} physics error = {physics_error:.4e}  " + custom)
    self.epoch_counter += 1
  def log_train_opt(self, name):
    # print(f"tf_epoch =      0  elapsed = 00:00  loss = 2.7391e-01  error = 9.0843e-01")
    print(f"—— Starting {name} optimization ——")

  def log_train_end(self, epoch, custom=""):
    print("==================")
    data_error,physics_error = self.__get_error_u()
    self.loss_over_meta.append([data_error, physics_error])
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
  def __init__(self, layers, h_activation_function, optimizer, logger,c,d,n_inputs=3,scaling_factor=1e12,physics_scale=1.0, p_norm="l2"):
    
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

    self.c = tf.constant(c/scaling_factor,dtype=tf.float32)
    self.d = tf.constant(d/scaling_factor,dtype=tf.float32)

    self.scaling_factor = tf.constant(scaling_factor,dtype=tf.float32)
    self.n_inputs = n_inputs
    self.optimizer = optimizer
    self.logger = logger
    self.physics_scale=physics_scale
    self.dtype = tf.float32
    self.p_norm = p_norm

    # semi supervised
    self.semi_mean = 0
    self.semi_s = 0
    self.semi_count = 0
    self.semi_count_thresh = 1000
    self.cond_var = tf.Variable([0.0])

    # Separating the collocation coordinates
    #self.x = tf.convert_to_tensor(X, dtype=self.dtype)
    
  # Defining custom loss
  @tf.function
  def __loss(self,x, y_lbl,x_physics, y, x_semi_begin=None, semi_scale=None, p_norm=None, semi_mean=None, semi_s=None, semi_count=None, bool_var=None):
    f_pred = self.f_model(x_physics)
    if x_semi_begin is not None:
      data_loss =  tf.reduce_mean((y_lbl[:x_semi_begin] - y[:x_semi_begin])**2)
      # calc for semi points
      semi_pred = tf.squeeze(y[x_semi_begin:])
      semi_pred_len = semi_pred.shape[0]
      pred_padded = tf.pad(semi_pred, [[ 0 , semi_count.shape[0]-semi_pred_len]])
      # https://www.johndcook.com/blog/standard_deviation/
      x_minus_mean_old = (pred_padded-semi_mean)
      semi_mean = semi_mean + x_minus_mean_old / semi_count
      semi_s = semi_s + x_minus_mean_old * (pred_padded - semi_mean)

      selector = tf.math.greater_equal(semi_count, 1000)


      if bool_var is not None:
        print("Adding semi points")
        var = semi_s[:semi_pred_len][selector[:semi_pred_len]] / (semi_count[:semi_pred_len][selector[:semi_pred_len]] - tf.ones_like(semi_pred[:semi_pred_len][selector[:semi_pred_len]]))
        mean_minus_pred_sq = (
                  (semi_mean[:semi_pred_len][selector[:semi_pred_len]] - semi_pred[selector[:semi_pred_len]]) ** 2)
        data_loss_semi = tf.reduce_mean(mean_minus_pred_sq * (
                  1 - tf.math.exp(-(mean_minus_pred_sq / (2 * var[:semi_pred_len][selector[:semi_pred_len]])))))

      else:
        data_loss_semi = 0


      semi_count = semi_count + tf.pad(tf.ones_like(semi_pred), [[ 0 , semi_count.shape[0]-semi_pred_len]]) # needs to be at the end to avoid division with zero therefor semi_count init to 1
    else:
      data_loss_semi = 0
      data_loss =  tf.reduce_mean((y_lbl - y)**2)

    if p_norm == "l2":
      physics_loss = tf.reduce_mean(f_pred**2)
    if p_norm == "l1":
      physics_loss = tf.reduce_mean(tf.math.abs(f_pred))

    return data_loss+data_loss_semi+self.physics_scale*physics_loss, semi_mean, semi_s, semi_count

  def __grad_data(self, x, y_lbl):
    with tf.GradientTape() as tape:
      tape.watch(x)  
      y = self.model(x)
      data_loss =  tf.reduce_mean(tf.square(y_lbl - y)) 
      print("___________")
      print(data_loss)
    return data_loss, tape.gradient(data_loss, self.__wrap_training_variables())
  
  def __grad(self, x, y_lbl, x_physics, x_semi_begin, semi_scale):

    if tf.math.reduce_any(tf.math.greater_equal(self.semi_count, 1000)):
      test_var = 0
    else:
      test_var = None

    with tf.GradientTape() as tape:

      tape.watch(x)  
      loss_value, self.semi_mean, self.semi_s, self.semi_count = self.__loss(x,y_lbl, x_physics, self.model(x), x_semi_begin, semi_scale, p_norm=self.p_norm, semi_mean=self.semi_mean, semi_s=self.semi_s, semi_count=self.semi_count, bool_var=test_var)
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.model.trainable_variables
    return var

  # The actual PINN
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
      ym1_lbl = tf.expand_dims(x[...,3],-1)

      # Variables t and x are watched during tape
      # to compute derivatives u_t and u_x
      tape.watch(x_)
      x_watched_full = tf.concat([x_,u,u_dx,ym1_lbl],axis=-1)

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
    return (-y_dx2/self.scaling_factor) -self.d*y_dx - self.c*y + self.d*u_dx + self.c*u

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
      #y_lbl_semi = tf.convert_to_tensor(y_semi, dtype=self.dtype)
      #semi_scale =  tf.convert_to_tensor(semi_scale, dtype=self.dtype)
      x_semi_begin = x.shape[0]
      x = tf.concat([x, x_semi], 0)
      #y_lbl = tf.concat([y_lbl, y_lbl_semi], 0)

    #self.logger.log_train_opt("Adam")
    for epoch in range(tf_epochs):
      # Optimization step
      #xy = tf.random.shuffle(xy)
      if epoch <0:
        loss_value, grads = self.__grad_data(x, y_lbl, x_physics)
        self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
      else:
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
  task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
  #task_id = int(sys.argv[1])
  print("task_id: ", task_id)
  if task_id % 2 == 0:
    act_func = tf.nn.tanh
    af_str = "tanh"
  else:
    act_func = tf.math.sin
    af_str = "sin"
  if task_id <= 7:
    p_norm = "l1"
  else:
    p_norm = "l2"

  layer_dic = {0: 3, 1: 3, 2: 7, 3: 7, 4: 12, 5: 12, 6: 20, 7: 20, 8: 3, 9: 3, 10: 7, 11: 7, 12: 12, 13: 12, 14: 20,
               15: 20}

  hidden_layers = layer_dic[task_id]
  layers = [4]
  for i in range(hidden_layers + 1):
    layers.append(34)
  layers.append(1)

  print("layers: ", layers)
  result_folder_name = 'res'
  if not os.path.exists(result_folder_name):
    os.makedirs(result_folder_name)

  # fixed parameters
  logger = Logger(frequency=1000)
  physics_scale = 1e-4
  p_start = 0
  p_end = 250
  # parameters ########################################################################################################
  ppi = 75
  max_iter_overall = 3000000
  meta_epochs = int((3750/(ppi/10))) # todo without hard coded numbers
  lr = tf.Variable(1e-4)
  tf_epochs_warm_up = 2000
  tf_epochs_train = int(max_iter_overall / meta_epochs)
  tf_epochs = tf_epochs_warm_up

  mode_frame = False
  tf_optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr,
    beta_1=0.8, decay=0.)

  experiment_name = "semi_ppi_" + str(ppi) + "_frame_" + str(mode_frame) + "_h_l_" + str(hidden_layers) + "_pn_" + p_norm + "_af_" + af_str
  os.makedirs(result_folder_name + "/"+ experiment_name, exist_ok=True)
  os.makedirs(result_folder_name + "/" + experiment_name+ "/plots", exist_ok=True)
  # data creation##########################################################################################################
  scale_factor = 1

  cPS = 1e6 * 4 # Stiffness Primary Suspension
  dPS = 1e4 * 4 # Damping Primary Suspension
  mDG = 3000
  cSS = 0.5e6 * 2 # Stiffness Secondary Suspension
  dSS = 1.5e4 * 2 # Stiffness Primary Suspension
  mWK = 30000/2

  cSF = cSS;
  dSF = dSS
  c = (cSF/mWK)
  d = (dSF/mWK)

  Av = 5.9233e-7;
  Omc = 0.8246;
  Omr = 0.0206;
  Om = np.logspace(-2,1,100)
  Fs = 200
  tExci = np.linspace(0, 10, 4000) #time
  Fs_Exci = 5*Fs;
  tExci_raw = np.linspace(0, 25, 25001)
  frequ = np.hstack((np.linspace(0.05,1.95,40), np.linspace(2,15,100)))
  #for f1 in frequ:
  f1 = 15.
  Sz = (Av*Omc**2.)/ ((f1**2. +Omr**2)*(f1**2+Omc**2))
  z_raw = Sz * np.sin(tExci_raw*f1*2*np.pi)
  z_raw = z_raw * scale_factor

  u_raw = z_raw
  up_raw = np.diff(u_raw,prepend=0)
  #up_raw = np.gradient(u_raw)# gradient vs diff?
  u_orig = np.expand_dims(u_raw,-1)
  up_orig = np.expand_dims(np.diff(np.squeeze(u_raw),prepend=0),-1)

  u_orig = np.expand_dims(np.interp(tExci,tExci_raw,u_raw),1) #todo why interpolation?
  up_orig = np.expand_dims(np.interp(tExci, tExci_raw,up_raw),1)

  tExci =  np.expand_dims(tExci, 1)
  u_orig = np.zeros_like(tExci)
  up_orig = np.zeros_like(u_orig)
  ################################################################################

  A =  np.array([
      [0, 1],
      [-cSF/mWK, -dSF/mWK]])

  B =  np.array(
    [[0, 0],
      [cSF/mWK, dSF/mWK]])


  C = np.array([[1, 0]]) # extract x1 = position
  D = np.zeros((1, 2))

  OneMassOsci = sig.StateSpace(A,B,C,D)

  sys_input = np.hstack((u_orig, up_orig))


  tsim_nom_orig,y_orig,xsim_nom_orig = sig.lsim(OneMassOsci,sys_input,np.squeeze(tExci),X0=[1,0])

  dy_orig = np.diff(y_orig,axis=0,prepend=0)
  dy2_orig = np.diff(dy_orig,axis=0,prepend=0)
  y_orig = np.expand_dims(y_orig, 1)
  dy_orig = np.expand_dims(dy_orig, 1)
  dy2_orig = np.expand_dims(dy2_orig, 1)

  # Getting the data
  data_sampling = 10
  scaling_factor = 1#1e12
  num_points = 250
  x_data = tExci[1:num_points+1:data_sampling]
  y_data = y_orig[1:num_points+1:data_sampling]*scaling_factor
  ym1_data = u_orig[0:num_points:data_sampling]*scaling_factor#np.zeros_like(x_data)#y_orig[0:num_points:data_sampling]*scaling_factor
  y_lbl = y_orig[1:]*scaling_factor
  ym1_lbl = u_orig[:-1]*scaling_factor#np.zeros_like(y_lbl)#y_orig[:-1]*scaling_factor
  u_lbl = u_orig[1:]*scaling_factor
  up_lbl = up_orig[1:]*scaling_factor
  t = tExci[1:]
  u_data = u_orig[1:num_points+1:data_sampling]*scaling_factor
  u_dx_data = up_orig[1:num_points+1:data_sampling]*scaling_factor

  input_all = tf.cast(tf.concat([t,u_lbl,up_lbl,ym1_lbl],axis=-1),tf.float32)
  input_data_physics = input_all

  input_data = tf.cast(tf.concat([x_data,u_data,u_dx_data,ym1_data],axis=-1),tf.float32)

  # plot and save data used
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  plt.plot(t, y_lbl, label="Exact solution")
  plt.scatter(x_data,y_data,c="r")
  plt.legend()
  #plt.show()
  fig.savefig(result_folder_name+"/" + experiment_name + '/exact_solution.png')


  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  plt.plot(t[:500], y_lbl[:500], label="Exact solution close-up")
  plt.plot(t[:500], ym1_lbl[:500], label="Previous step close-up")
  plt.scatter(x_data,y_data,c="r")
  plt.legend()
  #plt.show()
  fig.savefig(result_folder_name + "/" + experiment_name + '/exact_solution_close_up.png')

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  plt.plot(t[:500], u_orig[:500]*scaling_factor, label="u close-up")
  plt.scatter(x_data,u_data,c="r")
  plt.legend()
  #plt.show()
  fig.savefig(result_folder_name + "/" + experiment_name + '/u_close_up.png')

  def error():
    y, f = pinn.predict(input_all)
    return np.mean((y_lbl - y)**2),np.mean(f**2),
  logger.set_error_fn(error)

  print("Frame mode: ", str(mode_frame))
  print("ppi", ppi)


  pinn = PhysicsInformedNN(layers, h_activation_function= act_func, optimizer=tf_optimizer, logger=logger,c=c,d=d,n_inputs=layers[0],
                           scaling_factor=scaling_factor,physics_scale=physics_scale, p_norm=p_norm)

  # todo as function
  print("input_all shape:")
  print(y_data.shape)
  print(input_all.shape)
  print(input_all[0::data_sampling].shape)
  pot_semi_points = (input_all[0::data_sampling])[y_data.shape[0]::] # first point at index 251
  pinn.semi_mean = tf.zeros_like(pot_semi_points[:,0])
  pinn.semi_s =tf.zeros_like(pinn.semi_mean)
  pinn.semi_count = tf.ones_like(pinn.semi_mean)
  print(pot_semi_points.shape)
  print("shapes")
  print(pinn.semi_mean.shape)
  print(pinn.semi_s.shape)
  print(pinn.semi_count.shape)
  # todo end

  y_pred, f_pred = pinn.predict(input_all)
  print(input_all.shape)

  try_next_semi_points = False
  input_data_semi, y_data_semi, y_data_semi_pseudo, pseudo_physics_norm, input_data_semi_new_pot, semi_candidates = None, None, None, None, None, None

  for i in range(meta_epochs):
      print(str(i) + " / " +str(meta_epochs-1))
      if i==0:
          show_summary = True
      else:
          show_summary = False

      if (i + 1) % 10 == 0 and ppi != 0:
        tf_epochs = tf_epochs_train
        if mode_frame:
          p_start = p_end
        p_end += ppi
        if p_end > 3998:
          p_end = 3998
        print("new p_end: ", p_end)

      # add semi points one step behind physic points
      if (i + 1) % 10 == 0 and (i + 1) >= 20:
          nr_new_semipoints = ppi//data_sampling
          input_data_semi = pot_semi_points[:(nr_new_semipoints*(((i + 1)-10)//10))]

      pinn.fit(input_data, y_data,input_data_physics[p_start:p_end],input_data_semi, y_data_semi_pseudo, pseudo_physics_norm, tf_epochs, show_summary=show_summary)
      if (i + 1) % 10 == 0 or i == 0:
        y_pred, f_pred = pinn.predict(input_all)
        fig_res = plt.figure()
        plt.plot(t, y_pred)
        plt.scatter(x_data, y_data, c='r')
        selector = tf.math.greater_equal(pinn.semi_count, 1000)

        if tf.math.reduce_any(selector):
          plt_len = int(tf.reduce_sum(tf.cast(selector, tf.float32)))
          t_all = tExci[1::data_sampling]
          x_data_semi = t_all[:plt_len]
          plt.scatter(x_data_semi+t[251], pinn.semi_mean[:plt_len], c='c')
        plt.plot(t, y_lbl)
        plt.axvline(t[p_start], c='g')
        plt.axvline(t[p_end], c='r')
        plt.scatter(x_data, y_data, c='r')
        fig_res.savefig(result_folder_name + "/" + experiment_name + "/plots/" +str(i+1)+ '.png')
        plt.close(fig_res)
        with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
          pickle.dump(logger.loss_over_meta, fp)
        with open(result_folder_name + "/" + experiment_name + "/loss_epoch.pkl", "wb") as fp:
          pickle.dump(logger.loss_over_epoch, fp)
        with open(result_folder_name + "/" + experiment_name + "/p_end.pkl", "wb") as fp:
          pickle.dump([p_start, p_end], fp)

  with open(result_folder_name + "/" + experiment_name + "/loss.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_meta, fp)
  with open(result_folder_name+ "/" + experiment_name +"/loss_epoch.pkl", "wb") as fp:
    pickle.dump(logger.loss_over_epoch, fp)
  with open(result_folder_name + "/" + experiment_name +"/p_end.pkl", "wb") as fp:
    pickle.dump([p_start, p_end], fp)

  pinn.model.save_weights(result_folder_name + "/" + experiment_name +"/_weights")


  fig_l = plt.figure()
  plt.title("Loss over meta epochs")
  plt.plot([l[0] for l in logger.loss_over_meta], label='data_error')
  plt.plot([l[1] for l in logger.loss_over_meta], label='physics_error')
  plt.yscale("log")
  plt.legend()
  fig_l.savefig(result_folder_name + "/" + experiment_name + '/_loss.png')
  #plt.show()

  fig_s = plt.figure()
  plt.title("Loss over meta epochs scaled")
  plt.plot([l[0] for l in logger.loss_over_meta], label='data_error')
  plt.plot([l[1]*physics_scale for l in logger.loss_over_meta], label='physics_error')
  plt.yscale("log")
  plt.legend()
  fig_s.savefig(result_folder_name + "/" + experiment_name + '/_loss_s.png')

  y_pred, f_pred = pinn.predict(input_all)

  fig_res = plt.figure()
  plt.plot(t, y_pred)
  plt.scatter(x_data, y_data, c='r')
  plt.plot(t, y_lbl)
  plt.axvline(t[p_start], c='g')
  plt.axvline(t[p_end], c='r')
  plt.scatter(x_data, y_data, c='r')
  fig_res.savefig(result_folder_name + "/" + experiment_name + '/_res.png')

  print("Finished")



if __name__ == "__main__":
    main()




