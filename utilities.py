from torch import normal
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.stats import multivariate_normal as mvn

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from tqdm.auto import tqdm

import seaborn as sns
import sys
## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import time
import subprocess
import glob
import os
import json
import csv
from pathlib import Path

from custom_models import custom_models
from matplotlib.ticker import FormatStrFormatter




def visualise_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)

def custom_collate(batch):
    
    inputs = [b['X'] for b in batch]
    labels = [b['Y'] for b in batch]
    k_vec = [b['K']['vector'] for b in batch]
    k_lab = [b['K']['label'] for b in batch]
    k_mag = [b['K']['magnitude'] for b in batch]

    inputs = jnp.stack(inputs)
    labels = jnp.stack(labels)
    knowledge = {'vector':k_vec,'label':k_lab,'magnitude':k_mag}

    return{'X':inputs,'Y':labels,'K':knowledge}


def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    r = np.sqrt((x*x).sum())
    return x / r


def get_unit_vec(p1,p2):
      x1 = p1[0]
      x2 = p2[0]
      y1 = p1[1]
      y2 = p2[1]

      vec_x = x2 - x1
      vec_y = y2 - y1
      
      distance = np.sqrt(abs(vec_x)**2 + abs(vec_y)**2)

      return np.array([vec_x,vec_y])/distance, distance



def expand_data(dataclass):
        n_vec = dataclass.n_vec
        x = []
        y = []
        du = []
        dv = []
        dl = []
        dd = []

        for i,d in enumerate(dataclass.data):
            x.extend([d[0]]*n_vec)
            y.extend([d[1]]*n_vec)
            try:
              du.extend(dataclass.directions[i][:,0])
              dv.extend(dataclass.directions[i][:,1])          #  = np.append(d,[dataclass.directions[i][:,0],dataclass.directions[i][:,1]])
              dl.extend(dataclass.direction_label[i][:])
              dd.extend(dataclass.direction_distance[i][:])
            except:
              du.extend([0])
              dv.extend([0])
              dl.extend([0])
              dd.extend([0])
            
        
        data = jnp.column_stack((x,y)) 
        directions = jnp.column_stack((du,dv))
        direction_labels = jnp.array(dl)
        direction_distance = jnp.array(dd)

        return data, directions, direction_labels, direction_distance

def visualise_classes(dataset,knowledge=True):

  scale = 1
  fig = plt.figure(figsize = (5,5))
  ax = fig.add_subplot(111)
  # fig.set_size_inches(18.5, 10.5)

  x,y = dataset.X[:,0],dataset.X[:,1]
  labels = np.unique(dataset.Y)

  n = len(labels)
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])
  
  handles = []

  if knowledge:  
    normal_pal = sns.color_palette("Set1",(n+1)*2)
    pastel_pal = sns.color_palette("Pastel1",(n+1)*2)
    normal_pal.as_hex()
    pastel_pal.as_hex()

    for c_i,c in enumerate([int(l) for l in labels]):
      
      c_i_n = int(abs(c_i - 1))

      normal_patch = mpatches.Patch(color=normal_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = -1$')
      pastel_patch = mpatches.Patch(color=pastel_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = 1$')      
      handles.extend((normal_patch,pastel_patch))

    u = dataset.K['vector'][:,0]*dataset.K['magnitude'][:]
    v = dataset.K['vector'][:,1]*dataset.K['magnitude'][:]

    ax.quiver(x,y,u,v,angles='xy', scale_units = 'xy',
                                          color=[pastel_pal[c] for c in dataset.Y],width=1/200,alpha=1.0,headlength=4,headwidth=4,scale=1)
  ax.legend(handles=handles)

  ax.scatter(x,y,c=dataset.Y, cmap=cm_bright, edgecolors='k')
  plt.show()

  return fig, ax


def visualise_classes_archive(data):

  scale = 0.5
  fig = plt.figure(figsize = (5,5))
  ax = fig.add_subplot(111)
  # fig.set_size_inches(18.5, 10.5)  
  
  normal_pal = sns.color_palette("Set1",(len(data.classes)+1)*2)
  pastel_pal = sns.color_palette("Pastel1",(len(data.classes)+1)*2)
  normal_pal.as_hex()
  pastel_pal.as_hex()

  handles = []
  for c_i,c in enumerate(data.classes):
    
    x   = []
    y   = []
    u   = []
    v   = []
    col = []
    c_i_n = int(abs(c_i - 1))
    cols = [normal_pal[int(c_i_n)],pastel_pal[int(c_i_n)]]
    normal_patch = mpatches.Patch(color=normal_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = -1$')
    pastel_patch = mpatches.Patch(color=pastel_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = 1$')
    
    handles.extend((normal_patch,pastel_patch))
    if len(c.directions)==0:
      n = 0
    else:
      n = len(c.directions[0])

    # go through each label, if -1, change color?
    for i,d in enumerate(c.data):
      x.extend([d[0]]*n)
      y.extend([d[1]]*n)
      u.extend(c.directions[i][:,0])
      v.extend(c.directions[i][:,1])
      col.extend([cols[0] if dl < 0 else cols[int(dl)] for dl in c.direction_label[i][:]])
    
    ax.quiver(x,y,np.array(u)*scale,np.array(v)*scale,color=col,width=1/200, scale=10,alpha=0.5,headlength=4,headwidth=4)
  ax.legend(handles=handles)
  # plt.show()

  return fig, ax


def draw_classifier(predict, state, X_train, y_train, lims=None):
                    # ({'params': loaded_model_state.params}, data)
  fig, ax = plt.subplots()
  h = 0.1
  if lims is None:
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
  xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], h),
                      np.arange(lims[1][0], lims[1][1], h))
  logits = predict({'params': state.params}, np.stack([xx.ravel(), yy.ravel()]).T)
  probabilities = sigmoid(logits)
  Z = probabilities[:,1]
  Z_r = Z.reshape(xx.shape)
  x_max = []
  y_max = []
  x_p5 = []
  y_p5 = []
  
  
  for i in range(len(xx)):
    max_idx = np.argmax(Z_r[i])
    nearest_idx = (np.abs(Z_r[i] - 0.5)).argmin()
    x_max.append(xx[i][max_idx])
    y_max.append(yy[i][max_idx])
    x_p5.append(xx[i][nearest_idx])
    y_p5.append(yy[i][nearest_idx])
    
  cm = plt.cm.RdBu
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])
  im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)
  fig.colorbar(im, ax=ax)
  
  ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
  ax.scatter(x_max,y_max,facecolor='k',label='Max Probability',marker='1')
  ax.scatter(x_p5,y_p5,facecolor='orange',label = 'Probability=0.5',marker='2')
  ax.legend()
  ax.set_xlim(lims[0])
  ax.set_ylim(lims[1])
  plt.show()

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def compute_metrics(logits, labels):
    
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
    pred_labels = (logits > 0).astype(np.float32)
    
    accuracy = (pred_labels == labels).mean()
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def create_train_state(hyperparams,init_rng,opt):

    key = jax.random.PRNGKey(hyperparams['seed'])
    model = custom_models[hyperparams['model']](*hyperparams['model_size'])
    
    # params = simpleModel.init(key, np.ones([1,*model_size]))['params']
    params = model.init(key, jax.random.normal(init_rng, hyperparams['model_io']))['params']
    
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt), model


def print_dict(dct):
  for item, values in dct.items():  # dct.iteritems() in Python 2
        print("{} ({})".format(item, values))


def boundary_filter(dataset):
    len_td = len(dataset.data.X)
    count=0

    for i in range(0,len_td):      
        idx = i - count
       
        if dataset.data.K['magnitude'][idx] > 0.45:   
            dataset.drop(idx)
            count+=1

    print(f'Dataset reduced from {len_td} to {len(dataset)} boundary points.')
    return dataset


def predict_wrapper(model, params, data):
    y = model.apply({'params': params}, data)
    return y[0]

def close_event():
            plt.close() #timer calls this function after 3 seconds and closes the window 


def plotEpoch(hyperparams, X, y, states, plot_type = False):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
    
    for epoch,state in enumerate(states):
      model = custom_models[hyperparams['model']](*hyperparams['model_size'])
      Z = model.apply({'params': state.params}, points)

      grad_map = jax.vmap(jax.grad(predict_wrapper, argnums=2), in_axes=(None, None, 0), out_axes=0)
      grads = grad_map(model, state.params, points)
      magnitude = np.linalg.norm(grads, axis=1)

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
      timer = fig.canvas.new_timer(interval = 500) #creating a timer object and setting an interval of 3000 milliseconds
      timer.add_callback(close_event)

      cm = plt.cm.RdBu
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      cm2 = plt.cm.PuOr
      
      im = ax1.contourf(xx, yy, magnitude.reshape(xx.shape), cmap=cm2, alpha=.8)
      fig.colorbar(im, ax=ax1)
      ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      ax1.set_xlim(lims[0])
      ax1.set_ylim(lims[1])

      im = ax2.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
      fig.colorbar(im, ax=ax2)
      
      ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      ax2.set_xlim(lims[0])
      ax2.set_ylim(lims[1])
      

      if plot_type == 'video':
        
        plt.savefig(os.getcwd() + "/video/file%02d.png" % epoch)  
        plt.close()
      else:  
        timer.start()
        plt.show()
        # plt.show()
        # time.sleep(0.2)
        # print('supposed to close')
        # plt.clf()
      
    if plot_type == 'video':
      subprocess.call([
              'ffmpeg', '-framerate', '3', '-i',os.getcwd() + "/video/file%02d.png", '-r', '30', '-pix_fmt', 'yuv420p',
              os.getcwd() + "/video/video_name.mp4"])
      
      for file_name in glob.glob(os.getcwd() + "/video/*.png" ):
          os.remove(file_name)
    

def generate_figure(hyperparams, X, y, state):
    
    # plt.close('all')
    if plt.get_fignums():
      print(f'there are {len(plt.get_fignums())} figures')
      print(f'figure {plt.get_fignums()[0]}')
      gui_fig = plt.figure(plt.get_fignums()[0])
      plt.figure(gui_fig.number)
      # plt.clf()
      # gui_fig.clear()
      fignums = plt.get_fignums()
      print(fignums)
      print(len(plt.get_fignums()))
      
      # gui_fig = plt.gcf()
      print('EXISTS')
    else:
       print("EMPTY")
       gui_fig = plt.figure()
    
    rect = gui_fig.patch
    rect.set_facecolor('lightslategray')

    ax = gui_fig.add_axes([0.1,0.1,0.75,0.8])
    cax = gui_fig.add_axes([0.85,0.1,
                            0.05,0.8])
    
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
  
    model = custom_models[hyperparams['model']](*hyperparams['model_size'])
    Z = model.apply({'params': state.params}, points)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k')
    
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    gui_fig.colorbar(im,cax=cax)
    gui_fig.show()
    

    return gui_fig, ax, cax
    

def generate_figure_gui(hyperparams, X, y, state):

    gui_fig = plt.figure()

    ax = gui_fig.add_axes([0.1,0.1,0.75,0.8])
    cax = gui_fig.add_axes([0.85,0.1,
                            0.05,0.8])
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
  
    model = custom_models[hyperparams['model']](*hyperparams['model_size'])
    Z = model.apply({'params': state.params}, points)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k')
    
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    gui_fig.colorbar(im,cax=cax)
    

    return gui_fig, ax, scatter


def update_figure():
   None

def interactivePlot2():
   plt.scatter([0,1,2,903],[0,1,2,3])
   plt.show()



def gen_knowledge(dataset, knowledge_func):
    dataset.data.K['vector'],dataset.data.K['label'],dataset.data.K['magnitude'] = (
                                            knowledge_func(dataset))

def save_stats(dict,name, path = os.getcwd()+'/results'):
  print(name)
  with open(name,"w") as fp:
    json.dump(dict,fp) 

def plot_stats(paths):

  fig,axs = plt.subplots(1)
  ax = axs
  ax.set_ylim((0,110))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  ax.set_ylabel('Accuracy (%)')
  ax.set_xlabel('Epochs')
  for path in paths:
    stats = json.load(path)
    ax.plot(range(0,len(stats['accuracy'])),stats['accuracy'],label="Counterfactual Vectors") #,alpha=1.0,marker='+',linestyle='dashed')
  plt.show()

def gen_savepath(data_params,hyperparams):
  dataset = str(data_params['dataset'])
  knowledge = str(data_params['knowledge_func'])
  loss = str(hyperparams['loss_function'])
  size = 'size_'+str(data_params['size'])
  epochs = 'epochs_'+str(hyperparams['epochs'])

  path_order = [dataset, knowledge, loss, size, epochs]

  savepath = os.getcwd()+"/results/" + "/".join(path_order) + '/'
  Path(savepath).mkdir(parents=True, exist_ok=True)
  
  return savepath



@jax.jit  # Jit the function for efficiency
def train_step(state, batch, hyperparams, model, loss_function):

    (_, logits), grads = jax.value_and_grad(loss_function, has_aux=True, argnums=2)(hyperparams, model, state.params, batch)

    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits=logits, labels=batch['Y'])


    return state, metrics

def train_one_epoch(state, data_loader, hyperparams, model, loss_function):

    batch_metrics = []
    # 
    for cnt, batch in enumerate(data_loader):
        # from custom datsets - getitem: batch -> X, y, direction, direction label, direction distance 
        state, metrics = train_step(state, batch, hyperparams, model, loss_function)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }   
        
    return state, epoch_metrics_np