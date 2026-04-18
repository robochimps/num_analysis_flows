# Define target functions and their gradients for testing the optimization
# algorithms.

import jax.numpy as jnp 
import jax 

def f_algebraic(x, w=0.):
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = 1/(1.+x_**2)**8
    out /= jnp.trapezoid(out, x=x_)    
    return out 

def full_f_algebraic(x, w=0.):
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = 1/(1.+x_**2)**4    
    out /= jnp.sqrt(jnp.trapezoid(out**2, x=x_))    
    out *= jnp.sin(x_+1.2)
    return out 
    
def f_super_Gaussian(x, w=0.):    
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = jnp.exp(-x_**4 + x_**2)
    out /= jnp.trapezoid(out, x=x_)
    return out

def full_f_super_Gaussian(x, w=0.):    
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = jnp.exp(-x_**4/2 + x_**2/2)
    out /= jnp.sqrt(jnp.trapezoid(out**2, x=x_))
    out *= jnp.sin(x_+1.2)
    return out


def f_tanh(x, w=0.):
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = jnp.exp(-x_**2/2)
    out /= jnp.trapezoid(out, x=x_)
    out *= jnp.tanh(x_)
    return out

def full_f_tanh(x, w=0.):
    try:
        x_ = x[:,0]
    except:
        x_ = x[0]
    out = jnp.exp(-x_**2/2)
    out /= jnp.trapezoid(out, x=x_)
    out *= jnp.tanh(x_)
    return out


f_algebraic_grad = jax.vmap(jax.grad(f_algebraic))
f_super_Gaussian_grad = jax.vmap(jax.grad(f_super_Gaussian))
f_tanh_grad = jax.vmap(jax.grad(f_tanh))
    
full_f_algebraic_grad = jax.vmap(jax.grad(full_f_algebraic))
full_f_super_Gaussian_grad = jax.vmap(jax.grad(full_f_super_Gaussian))
full_f_tanh_grad = jax.vmap(jax.grad(full_f_tanh))
