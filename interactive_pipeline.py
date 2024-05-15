# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]

## Standard libraries

#ML Libraries
import jax
import numpy as np

import optax

import torch.utils.data as data

# Custom Libraries
from custom_datasets import customDataset, datasets
from loss_functions import loss_functions
from knowledge_functions import knowledge_functions
from utilities import  custom_collate, compute_metrics, create_train_state, boundary_filter

import yaml



"""
Initialise Parameters
"""
class Pipeline():
    def __init__(self):
        
        # global hyperparams, data_params, visualisation, trained_state, model, train_dataset, train_data_loader
        
        with open("config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            yamlfile.close()
        

        self.hyperparams   = config[0]['hyperparams']
        self.data_params   = config[0]['data_params']
        self.visualisation = config[0]['visualisation']

        # 1) Initialise the model
        rng = jax.random.PRNGKey(self.hyperparams['seed'])
        rng, inp_rng, init_rng = jax.random.split(rng, 3)
        inp = jax.random.normal(inp_rng, self.hyperparams['model_io'])


        sgd_opt = optax.sgd(self.hyperparams['learning_rate'] ,self.hyperparams['momentum'] )
        adam_opt = optax.adam(self.hyperparams['learning_rate'])
        self.state, self.model = create_train_state(self.hyperparams,init_rng,adam_opt)

        # Training loop
        self.states = np.empty([0])
        self.metrics   = np.empty([0])
        self.losses = np.empty([0])
        self.accs   = np.empty([0])

        self.train_dataset = customDataset(datasets[self.data_params['dataset']],self.data_params['size'],knowledge_functions[self.data_params['knowledge_func']],
                                                            train=True, visualise=self.visualisation['visualise'],seed=self.hyperparams['seed'])

        if self.data_params['boundary_only']:
                    self.train_dataset = boundary_filter(self.train_dataset)

        self.train_data_loader = data.DataLoader(self.train_dataset, batch_size=self.hyperparams['batch_size'], shuffle=True, collate_fn=custom_collate)


    # @jax.jit  # Jit the function for efficiency
    def train_step(self, batch):

        (_, logits), grads = jax.value_and_grad(loss_functions[self.hyperparams['loss_function']], 
                                                has_aux=True, argnums=2)(self.hyperparams, self.model, self.state.params, batch)
        metrics = compute_metrics(logits=logits, labels=batch['Y'])

        self.state = self.state.apply_gradients(grads=grads)
        self.states = np.append(self.states,self.state)
        self.metrics = np.append(self.metrics, metrics)
        return metrics


    def train_one_epoch(self):

        batch_metrics = []
        # 
        for cnt, batch in enumerate(self.train_data_loader):
            # from custom datsets - getitem: batch -> X, y, direction, direction label, direction distance 
            metrics = self.train_step(batch)
            batch_metrics.append(metrics)

        batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }   
        loss = epoch_metrics_np['loss']
        accuracy = epoch_metrics_np['accuracy']
        self.losses = np.append(self.losses,loss)
        self.accs = np.append(self.accs,accuracy*100)
    
    def add_knowledge(self,selected_point,vector):
        
        index = [i for i, coor in enumerate(self.train_dataset.data.X) if coor[0]==selected_point[0] 
                 and coor[1] == selected_point[1]]
 
        try:
            self.train_dataset.data.K['vector'][index[0]] = np.vstack((self.train_dataset.data.K['vector'][index[0]],vector))
        except:
             self.train_dataset.data.K['vector'][index[0]] = [vector]
 


    def reset(self):
         self.__init__()
         
    