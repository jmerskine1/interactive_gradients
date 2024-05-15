

from utilities import predict_wrapper, get_unit_vec
import numpy as np

import jax.numpy as jnp
from jax import grad, vmap
from jax.config import config as jax_config
jax_config.update("jax_debug_nans", True)

import optax

# model = hyperparams['model'] #(*hyperparams['model_size'])

# def direction(hyperparams, model, params, batch):

#     X,Y,K = batch['X'],batch['Y'],batch['K']

#     logits = model.apply({'params': params}, X).squeeze(axis=-1)  
#     loss = jnp.empty((0,1))

    
#     for i,d in enumerate(X):
#         g = grad(predict_wrapper,argnums=2)(model, params, d)
#         directional_derivative = g @ K['vector'][i].T 
#         sign = jnp.tanh(20.0*directional_derivative)

#         # label = 1 or 0 - need to convert sign accordingly?
#         # if label = 1, sign stays the same, if label = 0, sign flipped
#         ###
#         sign = sign*Y[i]*2 - sign

#         # print("LABEL: ",label[i])
#         # print("Direction Label: ",direction_label[i])
#         # print("SIGN: ", sign.primal)
#         # print("LOSS: ", np.abs(sign.primal-direction_label[i]))
#         loss = jnp.append(loss,jnp.abs(sign-K['label'][i]))

#     return jnp.mean(loss), logits


def cross_entropy(hyperparams, model, params, batch):
    X,Y,K = batch['X'],batch['Y'],batch['K']
    logits = model(*hyperparams['model_size']).apply({'params': params}, X).squeeze(axis=-1)    
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    return loss, logits


def direction(hyperparams, model, params, batch):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X).squeeze(axis=-1)  

    loss1 = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    loss2 = jnp.empty((0,1))

    for i,d in enumerate(X):
        g = grad(predict_wrapper,argnums=2)(model, params, d)

        directional_derivative = g @ K['vector'][i].T 

        sign = jnp.tanh(20.0*directional_derivative)
        sign = sign*Y[i]*2 - sign

        loss2 = jnp.append(loss2,jnp.abs(sign-K['label'][i]))

    loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*jnp.mean(loss2)
    # loss = loss1
    return loss, logits

def direction_interactive(hyperparams, model, params, batch):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X).squeeze(axis=-1)  

    loss1 = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    loss2 = jnp.empty((0,1))

    for i,d in enumerate(X):
        for j,k in enumerate(K['vector'][i]):
            g = grad(predict_wrapper,argnums=2)(model, params, d)

            directional_derivative = g @ k.T 

            sign = jnp.tanh(20.0*directional_derivative)
            # print(f"LABEL: {Y[i]}")
            # print(f"PRECONV: {sign.primal}")
            sign = sign*Y[i]*2 - sign
            # print(f"POSTCONV: {sign.primal}")

            loss2 = jnp.append(loss2,jnp.abs(sign+1))
            # print(f"LOSS: {jnp.abs(sign+1).primal}")
        #     print("LOSS2: ",loss2.primal)
        #     print(jnp.mean(loss2).primal)

    try:
        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*jnp.mean(loss2)
    except:

        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*0
    
    return loss, logits


def direction_interactive2(hyperparams, model, params, batch):


    
    # model = custom_models['simple'](*(8,1))
    
    # model.init(key, jax.random.normal(key, (8,1000)))['params']
    # params = model.init(key, jax.random.normal(init_rng, (8,1000)))['params']


    loss_mix = 0.75


    X, Y, K = batch['X'], batch['Y'], batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)
    loss1 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))

    loss2 = jnp.empty((1,))

    Kvec = jnp.array(K['vector'])
    
    for i in range(len(X)):
        # np.where instead of if
        if jnp.any(jnp.isnan(jnp.array(K['vector'][i]))):
            continue

        g = grad(predict_wrapper, argnums=2)(model, params, X[i])
        
        directional_derivative = jnp.dot(g, K['vector'][i].T)
        
        
        sign = jnp.tanh(20.0 * directional_derivative)
        sign = sign * Y[i] * 2 - sign
        # print('shape: ',jnp.shape(jnp.abs(sign+1)))
        loss2 = jnp.vstack((loss2.T, jnp.abs(sign+1))).T #extend(jnp.abs(sign + 1))

    loss2 = jnp.mean(loss2)
    loss = (1 - loss_mix) * loss1 + loss_mix * jnp.mean(loss2)
    # print('Loss: ',loss)
    return loss, logits

def direction_interactive3(hyperparams, model, params, batch):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X).squeeze(axis=-1)  

    loss1 = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    loss2 = jnp.empty((0,1))

    for i,d in enumerate(X):
        for j,k in enumerate(K['vector'][i]):
            g = grad(predict_wrapper,argnums=2)(model, params, d)

            directional_derivative = jnp.dot(g,k.T) 

            sign = jnp.tanh(20.0*directional_derivative)
            sign = sign * Y[i] * 2 - sign
            

            loss2 = jnp.append(loss2,100*jnp.abs(sign+1))

            # print(f"LABEL: {Y[i]}")
            # print(f"PRECONV: {sign.primal}")
            # print(f"POSTCONV: {sign.primal}")
            # print(f"LOSS: {jnp.abs(sign+1).primal}")
        

    try:
        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*jnp.mean(loss2)
        
    except:

        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*0
    
    # print(f"LOSS: {loss.primal}")

    return loss, logits

def gradient_supervision(hyperparams, model, params, batch):

    X,Y,K = batch['X'],batch['Y'],batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)  
    
    loss_ce = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    loss_gs = jnp.empty((0,1))     

    # loss = 1 - (gi.gˆi) / (||gi|| ||gˆi||)

    for i,d in enumerate(X):
        
        g_x = K['vector'][i] * K['magnitude'][i]
        g_y = grad(predict_wrapper,argnums=2)(model, params, d)
        # make unit vectors, inspect
        # loss_i = 1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)+1e-5)
        # track cross product over time
        
        dot = jnp.dot(g_x,g_y)
        cross = jnp.cross(g_x,g_y)

        # print(f'Dot: {dot.primal}\nCross: {cross.primal}')
        # loss_i = jnp.where(jnp.isnan(loss_i),0,loss_i)
        # loss_gs = jnp.append(loss_gs,1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)))
        try:
            loss_gs = jnp.append(loss_gs,1-dot/cross)
        except:
            print(f'Dot: {dot.primal}\nCross: {cross.primal}')
            print(f'G_X: {g_x}\nG_Y: {g_y.primal}')
    
    loss = (1-hyperparams['loss_mix'])*loss_ce + hyperparams['loss_mix']*jnp.mean(loss_gs)
    
    return loss, logits
    # return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=label)), logits

loss_functions = {'direction':direction,
             'cross_entropy':cross_entropy,
             'gradient_supervision':gradient_supervision,
             'direction_interactive': direction_interactive,
             'direction_interactive2': direction_interactive2,
             'direction_interactive3': direction_interactive3}